"""Compact live progress for a staged Shape-only batch."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('batch_root',type=Path)
args=parser.parse_args()
root=args.batch_root.resolve(strict=True)
state=json.loads((root/'status.json').read_text())
current=state.get('current_job')
result={'current_job':current,'phase':state.get('phase'),'stages':state['stages'],
        'finished_jobs':len(state['jobs']),'passed_jobs':sum(v['status']=='PASS' for v in state['jobs'].values())}
if current:
    attempts=sorted((root/'jobs'/current).glob('fit_attempt_*'))
    if attempts:
        progress=attempts[-1]/'progress.jsonl'
        if progress.exists():
            for line in reversed(progress.read_text().splitlines()):
                try:
                    event=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'active_mean_tooth_iou' in event:
                    result.update({'outer_iteration':event['outer_iteration'],'active_iou':event['active_mean_tooth_iou']})
                    break
else:
    result['status']=state.get('status')
print(json.dumps(result))
