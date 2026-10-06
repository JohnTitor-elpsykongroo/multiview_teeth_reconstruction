"""CPU diagnostics of original, present, trained codes and fixed-point response.

No fitting, meshing, selection, export, or held-out data access. A small code norm
does not establish functional collapse: the hypernetwork can amplify it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'DMM'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def original_rows(saved):
    trained = saved['training_case_ids']
    rows = saved['case_rows']
    ids = [row['case_id'] for row in rows]
    indices = [row['embedding_row'] for row in rows]
    if len(set(ids)) != len(ids) or len(set(indices)) != len(indices):
        raise ValueError('duplicate checkpoint case IDs or embedding rows')
    if not trained or len(set(trained)) != len(trained) or not set(trained).issubset(ids):
        raise ValueError('invalid checkpoint training IDs')
    selected = [row for row in rows if row['case_id'] in trained]
    if any(row['split'] != 'train' for row in selected):
        raise ValueError('checkpoint training subset includes held-out rows')
    original = [row for row in selected if row['augmentation_parent_id'] is None]
    if not original:
        raise ValueError('no original trained rows')
    return original


def latent_statistics(saved):
    rows = original_rows(saved)
    components, values = {}, {}
    for label in saved['specs']['labels']:
        present = [row for row in rows if label == 0 or row['presence'][str(label)]]
        if not present:
            raise ValueError(f'no original trained support for {label}')
        table = saved['system_state'][f'latents.{label}.weight'].detach().cpu().numpy()
        indices = [row['embedding_row'] for row in present]
        if min(indices) < 0 or max(indices) >= len(table):
            raise ValueError('embedding row outside checkpoint table')
        z = table[indices].astype(np.float64)
        if z.ndim != 2 or not np.isfinite(z).all():
            raise ValueError('nonfinite/invalid original present embeddings')
        mu = z.mean(0)
        centered = z - mu
        covariance = centered.T @ centered / max(1, len(z) - 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        variance = float(eigenvalues.sum())
        participation = variance ** 2 / float(np.square(eigenvalues).sum()) if variance else 0.
        values[label] = dict(mean=mu, z=z, rows=present)
        components[str(label)] = dict(original_present_count=len(z), dimension=z.shape[1],
            case_ids=[row['case_id'] for row in present], mean=mu.tolist(),
            mean_code_squared=float(np.square(z).mean()),
            mean_squared_centered_code=float(np.square(centered).mean()),
            code_norm_quantiles=np.quantile(np.linalg.norm(z, axis=1), [0, .5, .95, 1]).tolist(),
            dimension_std_ddof1=np.sqrt(np.diag(covariance)).tolist(),
            covariance_eigenvalues_ascending=eigenvalues.tolist(),
            covariance_participation_ratio=participation,
            covariance_condition_number=None if eigenvalues[0] == 0 else float(eigenvalues[-1]/eigenvalues[0]))
    return components, values


def fixed_train_probes(folder, case_ids, quota, rows):
    provenance = read(folder / 'provenance.json')
    if provenance.get('split') != 'train':
        raise ValueError('latent probe accepts only learned-train artifacts')
    evaluation = read(folder / 'evaluation.json')
    if evaluation['split'] != 'train':
        raise ValueError('probe split mismatch')
    original = {row['case_id']: row for row in rows}
    records = {case['case_id']: case for case in evaluation['cases']}
    if len(records) != len(evaluation['cases']) or len(set(case_ids)) != len(case_ids):
        raise ValueError('duplicate probe case')
    result, hashes = [], {str(folder/'provenance.json'): sha(folder/'provenance.json'),
                          str(folder/'evaluation.json'): sha(folder/'evaluation.json')}
    for case_id in case_ids:
        if case_id not in original or case_id not in records:
            raise ValueError('probe case must be an original trained case with saved artifact')
        record = records[case_id]
        if record['split'] != 'train' or record['status'] != 'EVALUATED_NOT_QUALITY_ACCEPTED':
            raise ValueError('probe requires successful learned-train artifact')
        relative = Path(record['artifact_directory']) / 'mesh.npz'
        path = (folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('probe artifact outside evaluation directory')
        with np.load(path, allow_pickle=False) as mesh:
            if mesh['model_unit_mm'].item() != 50:
                raise ValueError('probe units mismatch')
            points, labels = mesh['observed_points_model'], mesh['observed_labels']
            row = original[case_id]
            present = [0, *[int(k) for k,v in row['presence'].items() if v]]
            # Presence contains teeth only; each stored component must have points.
            present = sorted(set(present))
            if set(np.unique(labels)) != set(present):
                raise ValueError('probe point labels disagree with checkpoint presence')
            indices = np.concatenate([np.flatnonzero(labels == label)[:quota] for label in present])
            selected = np.asarray(points[indices], dtype=np.float32)
            if selected.shape != (len(indices), 3) or not np.isfinite(selected).all():
                raise ValueError('nonfinite/invalid probe points')
            result.append(dict(case_id=case_id, row=row, points=selected, labels=labels[indices],
                point_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                points_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
        hashes[str(path)] = sha(path)
    return provenance, result, hashes


def field_response(model, codes, mean_codes, points, truth):
    points = torch.as_tensor(points, dtype=torch.float32, device='cpu')
    with torch.no_grad():
        native = model.query(points, codes)
        variants = {'original_present_mean': mean_codes,
                    'zero': {label: torch.zeros_like(code) for label,code in codes.items()}}
        result = {}
        for name, alternate in variants.items():
            query = model.query(points, alternate)
            if any(not bool(torch.isfinite(value).all()) for prediction in (native,query)
                   for value in (prediction['sdf'],prediction['weights'])):
                raise ValueError('nonfinite probe response')
            difference = (query['sdf']-native['sdf']).abs().numpy()
            warped = torch.cat([(query['components'][label]['warped']-native['components'][label]['warped']).norm(dim=-1)
                                for label in codes]).numpy() * 50
            native_labels = np.asarray(native['labels'])[native['weights'].argmax(-1).numpy()]
            predicted = np.asarray(query['labels'])[query['weights'].argmax(-1).numpy()]
            result[name] = dict(sdf_absolute_change_model_quantiles=np.quantile(difference,[0,.5,.95,1]).tolist(),
                all_component_query_warped_point_change_mm_quantiles=np.quantile(warped,[0,.5,.95,1]).tolist(),
                semantic_prediction_change_fraction=float(np.mean(predicted != native_labels)),
                semantic_error=float(np.mean(predicted != truth)),
                surface_sdf_abs_mean_model=float(query['sdf'].abs().mean()))
            own_components = {}
            for label in codes:
                own = np.asarray(truth) == label
                before,after = native['components'][label],query['components'][label]
                own_warp = (after['warped'][own]-before['warped'][own]).norm(dim=-1).numpy()*50
                if not np.isfinite(own_warp).all():
                    raise ValueError('nonfinite own-component warp response')
                own_components[str(label)] = dict(points=int(own.sum()),
                    own_warped_point_change_mm_quantiles=np.quantile(own_warp,[0,.5,.95,1]).tolist(),
                    own_component_sdf_absolute_change_model_mean=float((after['sdf'][own]-before['sdf'][own]).abs().mean()))
            result[name]['own_components'] = own_components
        result['learned_train'] = dict(surface_sdf_abs_mean_model=float(native['sdf'].abs().mean()),
                                      semantic_error=float(np.mean(native_labels != truth)))
    return result


def inspect(checkpoint, probe_folder=None, case_ids=(), quota=16):
    from dmm.provenance import validate_decoder_binding, decoder_contract
    from networks.dmm_net import DMM
    started = time.perf_counter()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    stats, values = latent_statistics(saved)
    result = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha(checkpoint),
        epoch=saved['epoch'], arch=saved['arch'], training_source_sha256=saved['source_sha256'],
        training_project_revision=saved.get('project_revision'), manifest_sha256=saved['manifest_sha256'],
        canonical_reference_id=saved['canonical_reference_id'], recipe=saved['resolved_config']['recipe'],
        components=stats, probes=[], scope='original_present_trained_rows_only; diagnostic_not_quality_acceptance')
    if probe_folder is not None:
        validate_decoder_binding(dict(local_source_sha256=saved['source_sha256'], decoder_contract=saved['decoder_contract']))
        provenance, probes, hashes = fixed_train_probes(Path(probe_folder), case_ids, quota, original_rows(saved))
        if provenance['manifest_sha256'] != saved['manifest_sha256'] or provenance['canonical_reference_id'] != saved['canonical_reference_id']:
            raise ValueError('probe/checkpoint data binding mismatch')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = DMM(saved['specs'], arch=saved['arch']).cpu().eval().requires_grad_(False)
        state = {key.removeprefix('model.'): tensor for key,tensor in saved['system_state'].items() if key.startswith('model.')}
        model.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError('nonfinite decoder weights')
        for probe in probes:
            codes = {label: saved['system_state'][f'latents.{label}.weight'][probe['row']['embedding_row']].detach().clone()
                     for label in model.labels if label == 0 or probe['row']['presence'][str(label)]}
            mean_codes = {label: torch.tensor(values[label]['mean'],dtype=torch.float32) for label in codes}
            result['probes'].append(dict(case_id=probe['case_id'], points=len(probe['points']),
                point_index_sha256=probe['point_index_sha256'], points_sha256=probe['points_sha256'],
                response=field_response(model,codes,mean_codes,probe['points'],probe['labels'])))
        result.update(probe_input_sha256=hashes, runtime_decoder_contract=decoder_contract())
    result['seconds'] = time.perf_counter()-started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--probe-evaluation', type=Path)
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--points-per-component', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.points_per_component < 1 or bool(args.case_id) != bool(args.probe_evaluation):
        parser.error('fresh output, positive quota, and paired probe/case arguments required')
    results = [inspect(path,args.probe_evaluation,args.case_id,args.points_per_component) for path in args.checkpoint]
    result = dict(schema='dmm_original_latent_diagnostic_v1', script_sha256=sha(__file__),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device='cpu'),points_per_component=args.points_per_component,
        limitations='Fixed learned-train points; SDF change is not mesh distance. No clinical or latent-collapse threshold. No fit/grid/patient uncertainty measured.',
        checkpoints=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
