"""Finalize the review wording and hash the user-facing deliverables."""
import hashlib
import html
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUN = Path((PROJECT / 'tmp/dmm_static_trial_current.txt').read_text('utf-8'))
state = json.loads((RUN / 'finalization_state.json').read_text('utf-8'))
assert state['status'] == 'COMPLETE'
OUT, EVAL = Path(state['review']), Path(state['geometry'])
summary = json.loads((OUT / 'summary.json').read_text('utf-8'))
conclusion = '结论：本轮尚未取得合格重建。分牙匹配有所改善，但整体轮廓IoU与三维表面RMS均略有退步；形状优化基本停滞。'
summary['conclusion_zh'] = conclusion
summary['accepted_pose_updates'] = 3
summary['accepted_shape_updates'] = 1
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
report = (OUT / 'RESULT_ZH.txt').read_text('utf-8')
if not report.startswith(conclusion):
    (OUT / 'RESULT_ZH.txt').write_text(conclusion+'\n\n'+report, encoding='utf-8')
page = (OUT / 'index.html').read_text('utf-8')
if html.escape(conclusion) not in page:
    page = page.replace('<h1>DMM dual-arch synthetic pilot</h1>', '<h1>DMM dual-arch synthetic pilot</h1><p><strong>'+html.escape(conclusion)+'</strong></p>')
    (OUT / 'index.html').write_text(page, encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

registry_path = RUN / 'source_snapshot/helper_registry.json'
registry = json.loads(registry_path.read_text('utf-8'))
for name in ('review_dmm_static_trial.py', 'evaluate_dmm_static_geometry.py',
             'refresh_dmm_static_silhouette_scope.py', 'complete_dmm_static_delivery.py'):
    path = PROJECT / 'scripts' / name
    shutil.copy2(path, RUN / 'source_snapshot/helpers' / name)
    registry[name] = dict(original_path=str(path), sha256=digest(path))
registry_path.write_text(json.dumps(registry, indent=2)+'\n', encoding='utf-8')
files = [p for folder in (OUT, EVAL) for p in folder.iterdir() if p.is_file()]
files += [RUN / 'attempt_outcomes.json', RUN / 'final_integrity_audit.json',
          RUN / 'fit_pilot_008_resume_persistent/result.json', RUN / 'fit_pilot_008_resume_persistent/checkpoint.json']
delivery = dict(case='L428SD7J', fit_status=summary['status'], review=str(OUT / 'index.html'),
    conclusion_zh=conclusion, quality_accepted=False,
    files={str(p.relative_to(RUN)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files})
(RUN / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(dict(status='DELIVERY_SAVED', files=len(files), review=delivery['review']), indent=2))
