"""Use identical observed view sets for each per-arch silhouette mean."""
import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
FIT = RUN / 'fit_pilot_008_resume_persistent'
BASELINE = RUN / 'fit_pilot_004'
OUT = RUN / ('review_' + FIT.name)
summary_path = OUT / 'summary.json'
summary = json.loads(summary_path.read_text('utf-8'))
report = json.loads((FIT / 'final_render/report.json').read_text('utf-8'))
initial_report = json.loads((BASELINE / 'initial_render/report.json').read_text('utf-8'))
labels = np.array([0, *range(11, 18), *range(21, 28), *range(31, 38), *range(41, 48)])
rows = []
for view, record in report['views'].items():
    with np.load(FIT / 'final_render' / record['file']) as data:
        final, truth, valid = labels[data['semantics'].argmax(-1)], data['labels'], data['valid']
    with np.load(BASELINE / 'initial_render' / initial_report['views'][view]['file']) as data:
        initial = labels[data['semantics'].argmax(-1)]
    for arch in ('both', 'upper', 'lower'):
        def foreground(value):
            if arch == 'upper':
                return (value >= 11) & (value <= 27) & valid
            if arch == 'lower':
                return (value >= 31) & (value <= 47) & valid
            return (value != 0) & (value != 255) & valid
        a, b, c = foreground(truth), foreground(initial), foreground(final)
        def iou(pred):
            return float((a & pred).sum() / (a | pred).sum()) if (a | pred).any() else None
        rows.append(dict(view=view, arch=arch, target_pixels=int(a.sum()),
            initial_predicted_pixels=int(b.sum()), final_predicted_pixels=int(c.sum()),
            initial_iou=iou(b), final_iou=iou(c)))
scores = {arch: {phase: float(np.mean([r[phase+'_iou'] for r in rows if r['arch'] == arch and r['target_pixels'] > 0]))
    for phase in ('initial', 'final')} for arch in ('both', 'upper', 'lower')}
assert scores['both'] == summary['silhouette_iou']['both'], 'Primary metric changed unexpectedly'
summary['silhouette_iou'] = scores
summary['silhouette_scope'] = 'Same observed target-positive views per arch; no-observation false positives retained in per_view_silhouette pixel counts'
summary['silhouette_view_counts'] = {arch: sum(r['arch'] == arch and r['target_pixels'] > 0 for r in rows) for arch in scores}
summary['per_view_silhouette'] = rows
temporary = OUT / 'summary.scope.tmp.json'
temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
temporary.replace(summary_path)
print(json.dumps(dict(scores=scores, view_counts=summary['silhouette_view_counts']), indent=2))
