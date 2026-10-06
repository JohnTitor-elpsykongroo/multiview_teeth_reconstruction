"""Compact live status for a cross-case campaign and its child fitting batch."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('campaign_root', type=Path)
args = parser.parse_args()
root = args.campaign_root.resolve(strict=True)
state = json.loads((root/'status.json').read_text())
result = {k: state.get(k) for k in ['status', 'current_case', 'phase']}
result['finished_cases'] = {k: v['status'] for k, v in state['cases'].items()}
if state.get('current_case'):
    case = root/'cases'/state['current_case']
    batches = []
    for name in ['shape_resume.log', 'shape.log']:
        path = case/name
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get('batch_root'):
                    batches.append(Path(value['batch_root']))
                    break
    if batches and (batches[0]/'status.json').exists():
        child = json.loads((batches[0]/'status.json').read_text())
        result.update({'current_job': child.get('current_job'), 'fit_phase': child.get('phase'),
                       'finished_jobs': {k: v['status'] for k, v in child['jobs'].items()}})
        if child.get('current_job'):
            attempts = sorted((batches[0]/'jobs'/child['current_job']).glob('fit_attempt_*'))
            if attempts and (attempts[-1]/'progress.jsonl').exists():
                for line in reversed((attempts[-1]/'progress.jsonl').read_text().splitlines()):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'active_mean_tooth_iou' in value:
                        result.update({'outer_iteration': value['outer_iteration'], 'active_iou': value['active_mean_tooth_iou']})
                        break
    else:
        attempts = sorted(case.glob('source_attempt_*'))
        if attempts and (attempts[-1]/'progress.log').exists():
            result['source_progress'] = (attempts[-1]/'progress.log').read_text().splitlines()[-1]
print(json.dumps(result))
