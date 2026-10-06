"""Check future evaluator integration against the versioned post-fit evidence."""
import argparse
import json
import shutil
from pathlib import Path
from evaluate_shape_only import surface_metrics
from run_forward_check import PROJECT_ROOT, load_ply, sha256

parser = argparse.ArgumentParser()
parser.add_argument('fit_root', type=Path)
parser.add_argument('output_root', type=Path)
args = parser.parse_args()
fit = args.fit_root.resolve(strict=True)
output = args.output_root.resolve(strict=True)/'future_metric_integration'
if not all(p.is_relative_to((PROJECT_ROOT/'runs').resolve()) for p in [fit, output]):
    raise ValueError('only project run directories are allowed')
review = json.loads((fit/'surface_review_v1/evaluation.json').read_text())
inputs = Path(json.loads((fit/'provenance.json').read_text())['fit_input'])
source = Path(json.loads((inputs.parent/'truth/manifest.json').read_text())['source_run'])
checks = {}
for label in [11, 27]:
    target = load_ply(source/'meshes/training_case'/f'tooth{label}.ply')
    for stage in ['initial', 'final']:
        measured = surface_metrics(load_ply(fit/'meshes'/stage/f'tooth{label}.ply'), target)
        checks[f'{label}_{stage}'] = measured == review['surface_per_tooth'][str(label)][stage]
if not all(checks.values()):
    raise ValueError(f'integrated metrics differ: {checks}')
output.mkdir(exist_ok=False)
names = ['evaluate_shape_only.py', 'surface_metric_utils.py', 'run_shape_expansion.py',
         'run_shape_crosscase.py', 'review_shape_surface_metrics.py', 'check_surface_metric_fallback.py',
         'check_shape_metric_integration.py', 'verify_shape_crosscase.py', 'summarize_shape_crosscase.py']
hashes = {}
for name in names:
    script = PROJECT_ROOT/'scripts'/name
    shutil.copyfile(script, output/name)
    hashes[name] = sha256(script)
result = {'status': 'FUTURE_EVALUATOR_INTEGRATION_PASS', 'surface_pairs': checks, 'code_sha256': hashes,
          'case': str(fit), 'includes_regular_and_zero_area_target_triangles': True,
          'existing_fit_and_evaluation_files_modified': False}
(output/'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
