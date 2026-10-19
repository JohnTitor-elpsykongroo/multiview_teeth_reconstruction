"""Test the declared mean surface at finer sampling, with unchanged root gates."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / '.runtime/dmm_r4_best_val_20261015/package'
sys.path.insert(0, str(PACKAGE / 'third_party/DMM'))
import torch
from dmm.scene import load_scene
from dmm.surface import SurfaceConfig, extract_chart, attach_chart
from dmm.validation import write_json

run = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
scene = load_scene(json.loads((run / 'plan.json').read_text('utf-8'))['manifest'], 'cuda:0')
config = replace(SurfaceConfig.load(run / 'configs/surface_static_v1.json'), resolution=96)
report = dict(scope='Mean surfaces, q=0; no target mesh or truth consumed', config=config.resolved(), arches={})
for name, state in scene.arches.items():
    started = time.monotonic()
    try:
        with torch.no_grad():
            chart = extract_chart(state, config)
            points, semantics, detail = attach_chart(state, chart, config)
        report['arches'][name] = dict(passed=True, seconds=time.monotonic()-started, diagnostic=detail)
    except Exception as error:
        report['arches'][name] = dict(passed=False, seconds=time.monotonic()-started, error=str(error))
    print(name, report['arches'][name], flush=True)
write_json(run / 'surface_resolution96_diagnostic.json', report)
if all(row['passed'] for row in report['arches'].values()):
    write_json(run / 'configs/surface_resolution96.json', config.resolved())
