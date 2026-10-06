"""CPU grid diagnosis with checkpoint learned codes and saved train observations.

No fitting, val/test access, checkpoint selection or anatomical quality approval.
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
from summarize_autonomy_run import checked_evaluation


def train_artifact(folder, case_id):
    folder = Path(folder).resolve()
    # Reject held-out data before loading geometry or code-bearing checkpoints.
    provenance = json.loads((folder/'provenance.json').read_text())
    if provenance.get('split') != 'train':
        raise ValueError('fixed learned-code remeshing accepts only train')
    evaluation, provenance = checked_evaluation(folder)
    matches = [case for case in evaluation['cases'] if case['case_id'] == case_id]
    if len(matches) != 1 or matches[0]['status'] != 'EVALUATED_NOT_QUALITY_ACCEPTED':
        raise ValueError('requires one successful original learned-train case')
    case = matches[0]
    if case.get('code_mode') != 'learned_train':
        raise ValueError('artifact must use learned train codes')
    path = (folder/case['artifact_directory']/'mesh.npz').resolve()
    if not path.is_relative_to(folder):
        raise ValueError('artifact outside evaluation')
    with np.load(path, allow_pickle=False) as mesh:
        if mesh['model_unit_mm'].item() != 50:
            raise ValueError('fixed-code diagnostic units mismatch')
        points = np.asarray(mesh['observed_points_model'], dtype=np.float32)
        labels = np.asarray(mesh['observed_labels'])
    if points.shape != (len(labels), 3) or not np.isfinite(points).all():
        raise ValueError('invalid frozen observed points')
    return evaluation, provenance, case, points, labels, path


def parity(expected, actual):
    if set(expected['per_tooth']) != set(actual['per_tooth']):
        raise ValueError('grid parity omits a present tooth')
    for label, metrics in expected['per_tooth'].items():
        for key, value in metrics.items():
            if not np.isfinite(actual['per_tooth'][label][key]) or abs(value-actual['per_tooth'][label][key]) > 1e-5:
                raise ValueError(f'original grid parity failed: {label}/{key}')
    if abs(expected['score_mm']-actual['score_mm']) > 1e-5:
        raise ValueError('original equal-tooth score parity failed')


def remesh(checkpoint, evaluation, case_id, output, resolutions=(48,64,80), probe_config=None):
    from dmm.provenance import validate_decoder_binding, decoder_contract, source_fingerprint
    from dmm.validation import write_json, ContractError
    from networks.dmm_net import DMM
    from training.validation import mesh_metrics
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if output.exists() or not resolutions or len(set(resolutions)) != len(resolutions) or any(type(n) is not int or n < 8 or n > 128 for n in resolutions):
        raise ValueError('fresh output and unique grid resolutions 8..128 required')
    original, provenance, record, points, labels, mesh = train_artifact(evaluation, case_id)
    if original['budget']['mesh_resolution'] != resolutions[0]:
        raise ValueError('first grid must reproduce original resolution')
    if sha(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('checkpoint/evaluation SHA mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    rows = original_rows(saved)
    row = next((row for row in rows if row['case_id'] == case_id), None)
    if row is None:
        raise ValueError('case is not original trained checkpoint row')
    validate_decoder_binding(dict(local_source_sha256=saved['source_sha256'], decoder_contract=saved['decoder_contract']))
    bindings = probe_binding(provenance, saved, probe_config)
    present = {0, *[int(k) for k,v in row['presence'].items() if v]}
    if set(np.unique(labels)) != present or set(map(int, record['components'])) != present:
        raise ValueError('frozen point/checkpoint presence mismatch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = DMM(saved['specs'], arch=saved['arch']).cpu().eval().requires_grad_(False)
    state = {key.removeprefix('model.'): value for key,value in saved['system_state'].items() if key.startswith('model.')}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('nonfinite decoder weights')
    model.load_state_dict(state, strict=True)
    codes = {label: saved['system_state'][f'latents.{label}.weight'][row['embedding_row']].detach().clone() for label in present}
    sample = dict(points=torch.tensor(points), labels=torch.tensor(labels), components=tuple(sorted(present)))
    sources = {str(path):sha(path) for path in (checkpoint, Path(evaluation)/'evaluation.json', Path(evaluation)/'provenance.json', mesh)}
    sources.update(bindings)
    result = dict(schema='dmm_learned_train_grid_diagnostic_v1', case_id=case_id, arch=saved['arch'],
                  checkpoint_sha256=sources[str(checkpoint)], epoch=saved['epoch'], quality_accepted=False,
                  input_sha256=sources, source_script_sha256=sha(__file__), decoder_contract=decoder_contract(),
                  evaluation_source_sha256=source_fingerprint(),
                  helper_sha256={name:sha(Path(__file__).with_name(name)) for name in
                                 ('inspect_autonomy_latents.py','summarize_autonomy_run.py')},
                  runtime=dict(device='cpu', torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads()),
                  original_budget=original['budget'], grids=[],
                  scope='same learned codes and observed points; grid varies; fixed seed does not eliminate mesh-sampling uncertainty')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'study_partial.json', result)
    for index, resolution in enumerate(resolutions):
        budget = dict(original['budget'], mesh_resolution=resolution)
        began = time.perf_counter()
        grid = dict(resolution=resolution, status='FAILED', score_mm=None)
        try:
            geometry = mesh_metrics(model, codes, sample, saved['sampling_domain_model'], budget,
                                    artifact_dir=output/f'grid_{resolution}')
            if index == 0:
                parity(record['geometry'], geometry)
            grid.update(status='REMESHED_NOT_QUALITY_ACCEPTED', score_mm=geometry['score_mm'], geometry=geometry,
                        parity='PASS_1e-5mm' if index == 0 else None)
        except (ContractError, RuntimeError, ValueError) as exc:
            grid['error'] = str(exc)
        grid['seconds'] = time.perf_counter()-began
        result['grids'].append(grid)
        write_json(output/'study_partial.json', result)
        if index == 0 and grid['status'] == 'FAILED':
            raise ValueError('original grid failed; stop diagnostic and preserve partial state')
        if any(sha(path) != digest for path,digest in sources.items()):
            raise ValueError('frozen input drift during CPU grid diagnostic')
    result['failed_grid_count'] = sum(row['status'] == 'FAILED' for row in result['grids'])
    write_json(output/'study.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-config')
    parser.add_argument('--resolutions', nargs='+', type=int, default=[48,64,80])
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error('CPU thread budget must be 1..8')
    torch.set_num_threads(args.threads)
    result = remesh(args.checkpoint,args.evaluation,args.case_id,args.output,tuple(args.resolutions),args.probe_config)
    print(json.dumps(dict(case_id=result['case_id'], failed_grids=result['failed_grid_count'], output=args.output)))
