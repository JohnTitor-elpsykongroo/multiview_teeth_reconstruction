"""Versioned post-fit re-evaluation; preserve the original metric and fit files."""
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
from evaluate_shape_only import sampled_points
from run_forward_check import PROJECT_ROOT, load_ply, sha256
from surface_metric_utils import finite_nearest_triangle_distances


def metrics(first, second):
    diagnostics = [{}, {}]
    distances = np.r_[finite_nearest_triangle_distances(sampled_points(first,6000,410), second, diagnostics[0]),
                      finite_nearest_triangle_distances(sampled_points(second,6000,411), first, diagnostics[1])]
    return {'symmetric_sampled_surface_mean_dmm': float(distances.mean()),
            'symmetric_sampled_surface_p95_dmm': float(np.percentile(distances,95)),
            'samples_per_direction': 6000, 'nearest_triangle_candidates': 32,
            'distance_diagnostics': diagnostics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('only project runs may be re-evaluated')
    original = root/'evaluation.json'
    evaluation = json.loads(original.read_text())
    if not evaluation['truth_read_after_fit']:
        raise ValueError('original evaluation did not follow fit completion')
    output = root/'surface_review_v1'
    output.mkdir(exist_ok=False)
    fit = Path(json.loads((root/'provenance.json').read_text())['fit_input'])
    truth_path = fit.parent/'truth/manifest.json'
    if sha256(truth_path) != evaluation['truth_manifest_sha256']:
        raise ValueError('truth manifest changed since original evaluation')
    source = Path(json.loads(truth_path.read_text())['source_run'])
    surfaces, input_hashes = {}, {}
    unchanged_finite_metrics = True
    for label in evaluation['active_labels']:
        paths = [root/'meshes/initial'/f'tooth{label}.ply', root/'meshes/final'/f'tooth{label}.ply',
                 source/'meshes/training_case'/f'tooth{label}.ply']
        for path in paths:
            input_hashes[str(path)] = sha256(path)
        initial, final, target = map(load_ply, paths)
        a, b = metrics(initial, target), metrics(final, target)
        surfaces[str(label)] = {'initial': a, 'final': b, 'mean_improvement_fraction':
            1-b['symmetric_sampled_surface_mean_dmm']/a['symmetric_sampled_surface_mean_dmm']}
        for stage, value in [('initial', a), ('final', b)]:
            for key in ['symmetric_sampled_surface_mean_dmm', 'symmetric_sampled_surface_p95_dmm']:
                old = evaluation['surface_per_tooth'][str(label)][stage][key]
                if np.isfinite(old) and abs(old-value[key])>1e-14:
                    unchanged_finite_metrics = False
    initial_mean = float(np.mean([v['initial']['symmetric_sampled_surface_mean_dmm'] for v in surfaces.values()]))
    final_mean = float(np.mean([v['final']['symmetric_sampled_surface_mean_dmm'] for v in surfaces.values()]))
    gain = 1-final_mean/initial_mean
    limits = evaluation['acceptance_limits']
    checks = dict(evaluation['checks'])
    checks['surface_improvement'] = gain>=limits['min_surface_mean_improvement_fraction']
    checks['each_tooth_surface_nondegradation'] = all(v['mean_improvement_fraction']>=-limits['max_individual_surface_degradation_fraction'] for v in surfaces.values())
    checks['each_tooth_surface_p95_nondegradation'] = all(v['final']['symmetric_sampled_surface_p95_dmm']<=v['initial']['symmetric_sampled_surface_p95_dmm']*(1+limits['max_individual_surface_p95_degradation_fraction']) for v in surfaces.values())
    evaluation.update({'status': 'SHAPE_ONLY_SUBSET_PASS' if all(checks.values()) else 'SHAPE_ONLY_SUBSET_FAIL',
        'evaluated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'surface_per_tooth': surfaces, 'initial_surface_mean_dmm': initial_mean, 'final_surface_mean_dmm': final_mean,
        'surface_mean_improvement_fraction': gain, 'checks': checks,
        'surface_metric_revision': 'v1: same samples and nearest-32 candidates; nonfinite distances use exact segment/plane fallback',
        'original_evaluation_sha256': sha256(original), 'original_evaluation_preserved': True,
        'acceptance_limits_unchanged': True, 'finite_original_metrics_unchanged': unchanged_finite_metrics,
        'mesh_input_sha256': input_hashes,
        'metric_code_sha256': {name: sha256(PROJECT_ROOT/'scripts'/name) for name in ['review_shape_surface_metrics.py','surface_metric_utils.py']}})
    (output/'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    print(json.dumps({'status': evaluation['status'], 'run_root': str(root), 'surface_improvement': gain,
        'finite_original_metrics_unchanged': unchanged_finite_metrics,
        'failed_checks': [k for k,v in checks.items() if not v]}))


if __name__ == '__main__':
    main()
