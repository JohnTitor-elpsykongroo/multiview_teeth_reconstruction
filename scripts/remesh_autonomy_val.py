"""CPU val grid diagnosis using exact saved post-fit codes, without refitting.

Old artifacts without codes require fresh evaluation. This tool excludes test,
does not select checkpoints, and does not grant anatomical quality acceptance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/DMM'))
from inspect_autonomy_latents import original_rows, probe_binding, sha
from remesh_autonomy_train import parity
from summarize_autonomy_run import checked_evaluation


def val_artifact(folder, case_id):
    folder = Path(folder).resolve()
    provenance = json.loads((folder/'provenance.json').read_text())
    if provenance.get('split') != 'val':
        raise ValueError('saved fitted-code remeshing accepts only val before checkpoint/metric access')
    evaluation, provenance = checked_evaluation(folder)
    matches = [case for case in evaluation['cases'] if case['case_id'] == case_id]
    if len(matches) != 1 or matches[0]['status'] != 'EVALUATED_NOT_QUALITY_ACCEPTED':
        raise ValueError('requires one successful original val case')
    record = matches[0]
    fixed = evaluation['budget'].get('gum_code_mode') == 'fixed_train_mean'
    mode = 'temporary_teeth_fixed_train_mean_gum' if fixed else 'temporary_fitted'
    if record.get('code_mode') != mode:
        raise ValueError('unexpected saved val code mode')
    if evaluation.get('fit_protocol') != provenance.get('fit_protocol'):
        raise ValueError('fit protocol binding mismatch')
    if fixed and (not isinstance(provenance.get('fit_protocol'), dict)
                  or provenance['fit_protocol'].get('gum_code_mode') != 'fixed_train_mean'):
        raise ValueError('fixed gum protocol binding mismatch')
    artifact = (folder/record['artifact_directory']).resolve()
    if not artifact.is_relative_to(folder):
        raise ValueError('artifact outside evaluation')
    paths = {name:(artifact/name).resolve() for name in ('mesh.npz','field.json','codes.npz')}
    if any(not path.is_relative_to(folder) for path in paths.values()):
        raise ValueError('artifact file outside evaluation')
    if not paths['codes.npz'].exists():
        raise ValueError('no saved codes: fresh evaluation required; old fit cannot be recovered')
    field = json.loads(paths['field.json'].read_text())
    if field.get('codes_sha256') != sha(paths['codes.npz']):
        raise ValueError('field/code SHA mismatch')
    if field.get('resolution') != evaluation['budget']['mesh_resolution']:
        raise ValueError('field/original grid mismatch')
    hashes = {str(path):sha(path) for path in
              (folder/'evaluation.json',folder/'provenance.json',*paths.values())}
    with np.load(paths['mesh.npz'],allow_pickle=False) as mesh:
        if mesh['model_unit_mm'].item() != 50:
            raise ValueError('observed-point units mismatch')
        points = mesh['observed_points_model'].copy()
        labels = mesh['observed_labels'].copy()
    if points.dtype != np.float32 or points.shape != (len(labels),3) or not np.isfinite(points).all():
        raise ValueError('invalid frozen observed points')
    if labels.ndim != 1 or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('invalid frozen observed labels')
    return evaluation,provenance,record,points,labels,paths,hashes


def remesh(checkpoint,evaluation,case_id,output,resolutions=(48,64,80),probe_config=None):
    from dmm.provenance import validate_decoder_binding, decoder_contract, source_fingerprint
    from dmm.validation import write_json, ContractError
    from networks.dmm_net import DMM
    from training.validation import mesh_metrics
    output,checkpoint = Path(output).resolve(),Path(checkpoint).resolve()
    if output.exists() or not resolutions or len(set(resolutions)) != len(resolutions) or any(
            type(n) is not int or not 8 <= n <= 128 for n in resolutions):
        raise ValueError('fresh output and unique grid resolutions 8..128 required')
    original,provenance,record,points,labels,paths,hashes = val_artifact(evaluation,case_id)
    if original['budget']['mesh_resolution'] != resolutions[0]:
        raise ValueError('first grid must reproduce original resolution')
    hashes[str(checkpoint)] = sha(checkpoint)
    if hashes[str(checkpoint)] != provenance['checkpoint_sha256']:
        raise ValueError('checkpoint/evaluation SHA mismatch')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    rows = [row for row in saved['case_rows'] if row['case_id']==case_id and row['split']=='val'
            and row['augmentation_parent_id'] is None and row['case_id'] not in saved['training_case_ids']]
    if len(rows) != 1 or rows[0]['patient_id'] != record['patient_id']:
        raise ValueError('case is not an original eligible val checkpoint row')
    row = rows[0]
    hashes.update(probe_binding(provenance,saved,probe_config))
    validate_decoder_binding(dict(local_source_sha256=saved['source_sha256'],decoder_contract=saved['decoder_contract']))
    present = {0,*[int(k) for k,v in row['presence'].items() if v]}
    if set(np.unique(labels)) != present or set(map(int,record['components'])) != present:
        raise ValueError('frozen point/checkpoint presence mismatch')
    with np.load(paths['codes.npz'],allow_pickle=False) as arrays:
        if arrays['model_unit_mm'].item()!=50 or set(arrays.files)!={'model_unit_mm',*[f'code_{label}' for label in present]}:
            raise ValueError('code artifact units/presence mismatch')
        codes = {}
        for label in sorted(present):
            value = arrays[f'code_{label}'].copy()
            table = saved['system_state'][f'latents.{label}.weight']
            if value.dtype != np.float32 or value.shape != (table.shape[1],) or not np.isfinite(value).all():
                raise ValueError('invalid saved code dimension/dtype/numeric')
            codes[label] = torch.from_numpy(value)
        if original['budget'].get('gum_code_mode') == 'fixed_train_mean':
            support = original_rows(saved)
            mean = saved['system_state']['latents.0.weight'][[r['embedding_row'] for r in support]].numpy().astype(np.float64).mean(0).astype(np.float32)
            if not np.array_equal(codes[0].numpy(),mean):
                raise ValueError('saved fixed gum differs from exported training mean')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = DMM(saved['specs'],arch=saved['arch']).cpu().eval().requires_grad_(False)
    state = {key.removeprefix('model.'):value for key,value in saved['system_state'].items() if key.startswith('model.')}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('nonfinite decoder weights')
    model.load_state_dict(state,strict=True)
    if not present.issubset(set(model.labels)):
        raise ValueError('saved codes outside decoder labels')
    sample = dict(points=torch.from_numpy(points),labels=torch.from_numpy(labels),components=tuple(sorted(present)))
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('frozen input drift before CPU grid diagnostic')
    result = dict(schema='dmm_saved_val_code_grid_diagnostic_v1',case_id=case_id,arch=saved['arch'],epoch=saved['epoch'],
        checkpoint_sha256=hashes[str(checkpoint)],quality_accepted=False,code_mode=record['code_mode'],
        fit_protocol=original.get('fit_protocol'),input_sha256=hashes,source_script_sha256=sha(__file__),
        helper_sha256={name:sha(Path(__file__).with_name(name)) for name in
                      ('inspect_autonomy_latents.py','summarize_autonomy_run.py','remesh_autonomy_train.py')},
        decoder_contract=decoder_contract(),evaluation_source_sha256=source_fingerprint(),
        runtime=dict(device='cpu',torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads()),
        original_budget=original['budget'],grids=[],scope='Exact saved post-fit codes and observed points; no fitting. Grid alone varies; fixed seed does not remove sampling uncertainty.')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'study_partial.json',result)
    for index,resolution in enumerate(resolutions):
        began = time.perf_counter()
        grid = dict(resolution=resolution,status='FAILED',score_mm=None)
        try:
            geometry = mesh_metrics(model,codes,sample,saved['sampling_domain_model'],
                                    dict(original['budget'],mesh_resolution=resolution),artifact_dir=output/f'grid_{resolution}')
            if index == 0: parity(record['geometry'],geometry)
            grid.update(status='REMESHED_NOT_QUALITY_ACCEPTED',score_mm=geometry['score_mm'],geometry=geometry,
                        parity='PASS_1e-5mm' if index==0 else None)
        except (ContractError,RuntimeError,ValueError) as exc:
            grid['error'] = str(exc)
        grid['seconds'] = time.perf_counter()-began
        result['grids'].append(grid)
        write_json(output/'study_partial.json',result)
        if any(sha(path)!=digest for path,digest in hashes.items()):
            raise ValueError('frozen input drift during CPU grid diagnostic')
        if grid['status']=='FAILED':
            raise ValueError('grid failed; stop diagnostic and preserve partial state')
    result['failed_grid_count'] = 0
    write_json(output/'study.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','evaluation','case-id','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--probe-config')
    parser.add_argument('--resolutions',nargs='+',type=int,default=[48,64,80])
    parser.add_argument('--threads',type=int,default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8: parser.error('CPU thread budget must be 1..8')
    torch.set_num_threads(args.threads)
    result = remesh(args.checkpoint,args.evaluation,args.case_id,args.output,tuple(args.resolutions),args.probe_config)
    print(json.dumps(dict(case_id=result['case_id'],failed_grids=result['failed_grid_count'],output=args.output)))
