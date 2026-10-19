"""Verify mean surface root gates in the isolated safeguarded runtime."""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
run = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
sys.path.insert(0, str(run / 'runtime_source/DMM'))
import torch
from dmm.scene import load_scene
from dmm.surface import SurfaceConfig, extract_chart, attach_chart
from dmm.provenance import source_fingerprint, decoder_contract
from dmm.validation import write_json

scene = load_scene(json.loads((run / 'plan.json').read_text('utf-8'))['manifest'], 'cuda:0')
config = SurfaceConfig.load(run / 'configs/surface_static_v1.json')
report = dict(source_sha256=source_fingerprint(), decoder_contract=decoder_contract(),
              scope='Original residual/domain/derivative/trust gates; damped and bracketed roots', arches={})
for name, state in scene.arches.items():
    started = time.monotonic()
    try:
        with torch.no_grad():
            chart = extract_chart(state, config)
            points, semantics, detail = attach_chart(state, chart, config)
        report['arches'][name] = dict(passed=True, seconds=time.monotonic()-started, diagnostic=detail)
    except Exception as error:
        report['arches'][name] = dict(passed=False, seconds=time.monotonic()-started, error=str(error))
        tb = error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name == 'attach_chart':
                local = tb.tb_frame.f_locals
                if 'phi' in local:
                    values = local['phi'].detach().abs()
                    report['arches'][name]['residual'] = dict(max=float(values.max()),
                        failed=int((values > config.root_tolerance).sum()), count=len(values))
                if 'offset' in local:
                    report['arches'][name]['max_offset'] = float(local['offset'].abs().max())
            tb = tb.tb_next
    print(name, report['arches'][name], flush=True)
write_json(run / 'safeguarded_surface_diagnostic.json', report)
sys.exit(0 if all(row['passed'] for row in report['arches'].values()) else 1)
