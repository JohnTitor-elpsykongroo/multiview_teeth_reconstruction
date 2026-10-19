"""Retain initializer locals on failure, without editing the deployed source."""
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / '.runtime/dmm_r4_best_val_20261015/package'
sys.path.insert(0, str(PACKAGE / 'third_party/DMM'))
from dmm.scene import load_scene
from dmm.surface import SurfaceConfig
from dmm.rendering import RenderConfig
from dmm.semanticxy import SemanticXYConfig
from dmm.initialization import initialize_mean_scene
from dmm.validation import write_json

run = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
plan = json.loads((run / 'plan.json').read_text('utf-8'))
scene = load_scene(plan['manifest'], 'cuda:0')
surface = SurfaceConfig.load(run / 'configs/surface_static_v1.json')
render = RenderConfig.load(run / 'configs/render_512.json')
matching = SemanticXYConfig.load(run / 'configs/semanticxy_static_v1.json')
try:
    report = initialize_mean_scene(scene, surface, render, matching)
except Exception as error:
    report = dict(error=str(error), traceback=traceback.format_exc())
    tb = error.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name == 'initialize_mean_scene':
            values = tb.tb_frame.f_locals
            report.update(arch=values.get('arch'), candidates=values.get('records'), reports=values.get('reports'),
                proposals={a: [p.tolist() for p in poses] for a, poses in values.get('proposals', {}).items()})
            for name, chart in values.get('charts', {}).items():
                import numpy as np
                np.savez_compressed(run / ('mean_canonical_' + name + '.npz'),
                    vertices_model=chart.anchors.detach().cpu().numpy(),
                    normals=chart.normals.detach().cpu().numpy(), faces=chart.faces.detach().cpu().numpy())
        tb = tb.tb_next
write_json(run / 'initialization_diagnostic.json', report)
print(json.dumps(report, indent=2))
