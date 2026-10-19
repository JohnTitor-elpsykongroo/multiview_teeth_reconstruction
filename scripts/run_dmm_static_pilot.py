"""A bounded all-view GPU pilot with declared observation-only initialization."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

project = Path(__file__).resolve().parents[1]
run = Path((project / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
plan = json.loads((run / 'plan.json').read_text('utf-8'))
source = Path(plan['source'])
package = project / '.runtime/dmm_r4_best_val_20261015/package'
spec = json.loads((source / 'specs/L428SD7J.json').read_text('utf-8'))
diagnostic = json.loads((run / 'initialization_diagnostic.json').read_text('utf-8'))
spec['scene_id'] += '_pilot256'
spec['initialization']['method'] = 'independent_estimate'
spec['initialization']['origin'] = 'Observation-only visible-FDI mask centroid ray triangulation and mean-model Kabsch proposal; candidate 0 from saved initializer diagnostic; no target 3D or truth pose. Not anatomical correspondences.'
for arch in ('upper', 'lower'):
    spec['initialization']['arches'][arch]['T_world_from_arch'] = diagnostic['proposals'][arch][0]
assets = run / 'pilot_assets'
assets.mkdir()
for row in spec['views']:
    for kind in ('rgb', 'mask', 'valid_mask'):
        target = assets / (row['view_id'] + '_' + kind + '.png')
        with Image.open(row[kind]) as image:
            image.resize((256, 256), Image.Resampling.LANCZOS if kind == 'rgb' else Image.Resampling.NEAREST).save(target)
        row[kind] = str(target)
    row.update(width=256, height=256, source_width=512, source_height=512)
    row['A_fit_from_source_pixels'] = np.diag([.5, .5, 1.]).tolist()
    row['K'] = (np.diag([.5, .5, 1.]) @ np.array(row['K_source'])).tolist()
spec_path = run / 'pilot_scene_spec.json'
spec_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
scene = run / 'scenes/L428SD7J_pilot256'
builder = [sys.executable, '-u', '-B', str(package / 'tools/build_scene_input.py'), '--spec', str(spec_path), '--output', str(scene)]
with (run / 'pilot_build_scene.log').open('w', encoding='utf-8') as log:
    subprocess.run(builder, stdout=log, stderr=subprocess.STDOUT, check=True)
fit = json.loads((run / 'configs/fit_static_v1.json').read_text('utf-8'))
fit.update(pose_iterations=10, shape_iterations=10, joint_iterations=20)
fit_path = run / 'configs/fit_pilot40.json'
fit_path.write_text(json.dumps(fit, indent=2), encoding='utf-8')
render = json.loads((run / 'configs/render_512.json').read_text('utf-8'))
render['max_pixels'] = 1048576
render_path = run / 'configs/render_pilot256.json'
render_path.write_text(json.dumps(render, indent=2), encoding='utf-8')
cli = [sys.executable, '-u', '-B', str(run / 'runtime_source/DMM/dmm_cli.py')]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(project / 'tmp'), TMP=str(project / 'tmp'))
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
validate = cli + ['validate-scene', str(scene / 'fit_input/manifest.json'), '--device', 'cpu']
with (run / 'pilot_validate_scene.log').open('w', encoding='utf-8') as log:
    subprocess.run(validate, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
command = cli + ['fit-scene', str(scene / 'fit_input/manifest.json'), '--device', 'cuda:0',
    '--fit-config', str(fit_path), '--surface-config', str(run / 'configs/surface_static_v1.json'),
    '--render-config', str(render_path), '--matching-config', str(run / 'configs/semanticxy_static_v1.json'),
    '--output', str(run / 'fit_pilot_003')]
(run / 'pilot_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
state = dict(status='PILOT_RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
             budget=dict(pose=10, shape=10, joint=20), views=15, fit_resolution=[256, 256],
             initialization='Saved observation-only mean-model Kabsch candidate, no 3D truth',
             derivative='float64 local implicit roots/field; float32 raster; frozen model values')
(run / 'pilot_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
start = time.monotonic()
with (run / 'fit_pilot_003.log').open('w', encoding='utf-8') as log:
    proc = subprocess.Popen(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['child_pid'] = proc.pid
    (run / 'pilot_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    state['returncode'] = proc.wait()
state.update(status='PILOT_FINISHED' if state['returncode'] == 0 else 'PILOT_FAILED_RETAINED', seconds=time.monotonic()-start)
(run / 'pilot_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps(state), flush=True)
sys.exit(state['returncode'])
