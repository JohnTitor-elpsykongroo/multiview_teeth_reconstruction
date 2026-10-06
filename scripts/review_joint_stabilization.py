"""Summarize completed ablations without changing gates or selecting truth."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]).resolve()
protocol=json.loads((root/'protocol.json').read_text())
rows=[]
for job in sorted((root/'jobs').iterdir()):
    path=job/'fit_attempt_01/evaluation.json'
    if not path.exists():
        rows.append({'job':job.name,'status':'NOT_EVALUATED'}); continue
    e=json.loads(path.read_text()); p=e['pose_errors']['final']
    rows.append({'job':job.name,'status':e['status'],'rotation':p['rotation_degrees'],'translation':p['translation_dmm'],
        'clean_iou':e['final_active_mean_tooth_iou'],'heldout_iou':e['heldout']['final']['active_mean_tooth_iou'],
        'canonical_gain':e['surface_aggregate']['canonical']['mean_improvement_fraction'],'failed':e['failed_checks']})
(root/'review').mkdir(exist_ok=True)
(root/'review/metrics.json').write_text(json.dumps(rows,indent=2))
lines=['# Alternating pose/latent and boundary tolerance ablations','',
    'Packaged observations and image-only warm-start poses are used. No geometric truth is used by the fitter. Original acceptance gates are unchanged.',
    '', 'Budget: up to 24 outer iterations, 24 function evaluations per block, patience 8. This exceeds the earlier coupled control budget; improvements cannot be attributed to schedule alone.',
    '', '| Job | Status | Rotation deg | Translation DMM | Clean IoU | Heldout IoU | Canonical improvement |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    if 'rotation' in r:
        lines.append(f"| {r['job']} | {r['status']} | {r['rotation']:.4f} | {r['translation']:.6f} | {r['clean_iou']:.6f} | {r['heldout_iou']:.6f} | {r['canonical_gain']:.2%} |")
    else: lines.append(f"| {r['job']} | {r['status']} | | | | | |")
lines+=['','## Failed gates','']+[f"- {r['job']}: {', '.join(r.get('failed',['not evaluated'])) or 'none'}" for r in rows]
lines+=['','## Protocol','',json.dumps(protocol,indent=2),'',
    'Population runs change both the latent regularizer and output-selection score. The comparison does not isolate either change individually. Covariance is an empirical regularizer, not a calibrated anatomical probability.',
    '', 'A numerical gradient pass does not establish reconstruction accuracy. Canonical errors remain in the frozen DMM coordinate frame; no Procrustes truth alignment is applied. DMM distance units are not millimeters.']
(root/'review/report.md').write_text('\n'.join(lines),encoding='utf-8')
complete=[r for r in rows if 'rotation' in r]
if complete:
    fig,axes=plt.subplots(1,3,figsize=(14,5)); names=[r['job'] for r in complete]
    for ax,key,gate,title in zip(axes,['rotation','translation','canonical_gain'],[1,.01,.2],['Rotation error (degrees)','Translation error (DMM units)','Canonical mean improvement']):
        ax.bar(names,[r[key] for r in complete]); ax.axhline(gate,color='red',linestyle='--'); ax.set_title(title); ax.tick_params(axis='x',rotation=35)
    fig.tight_layout(); fig.savefig(root/'review/metrics.png',dpi=160); plt.close(fig)
print(json.dumps(rows,indent=2))
