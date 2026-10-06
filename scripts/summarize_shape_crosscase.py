"""Include every selected case, source failure, and seed in campaign evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('campaign_root', type=Path)
parser.add_argument('--surface-review', action='store_true', help='use uniformly recomputed finite surface metrics')
parser.add_argument('--output-folder', default='review')
args = parser.parse_args()
root = args.campaign_root.resolve(strict=True)
state = json.loads((root/'status.json').read_text())
plan = json.loads((root/'plan.json').read_text())
if Path(args.output_folder).name != args.output_folder:
    raise ValueError('output must be one folder name')
output = root/args.output_folder
output.mkdir(exist_ok=False)
records = []
lines = ['# 跨病例完整上颌 Shape-only', '',
         f"状态：`{state['status']}`。三个预选训练病例，每例 14 牙、280 维、0.25 sigma，两种初始化。相机及位姿已知。", '',
         '| 病例 | 分位 | seed | 状态 | 拟合 IoU 初始 → 最终 | 预留 IoU 初始 → 最终 | 平均表面误差降低 |',
         '| --- | ---: | ---: | --- | --- | --- | --- |']
for selected in plan['selection']['selected']:
    case_id = Path(selected['case']).stem
    case = state['cases'].get(case_id, {'status': 'NOT_COMPLETED'})
    if 'jobs' not in case:
        records.append(dict(selected, case=case_id, status=case['status']))
        lines.append(f"| {case_id} | {selected['selection_quantile']} | — | {case['status']} | — | — | — |")
        continue
    for job_id, job in case['jobs'].items():
        r = dict(job, case=case_id, selection_quantile=selected['selection_quantile'])
        if args.surface_review and job.get('fit_root'):
            review_path = Path(job['fit_root'])/'surface_review_v1/evaluation.json'
            review = json.loads(review_path.read_text())
            r.update({'original_status': job['status'], 'surface_review_evaluation': str(review_path),
                'status': 'PASS' if review['status']=='SHAPE_ONLY_SUBSET_PASS' else 'EVALUATION_FAIL',
                'surface_improvement': review['surface_mean_improvement_fraction'],
                'failed_checks': [k for k,v in review['checks'].items() if not v]})
        job = r
        records.append(r)
        if job['status'] not in ['PASS', 'EVALUATION_FAIL']:
            lines.append(f"| {case_id} | {selected['selection_quantile']} | {job_id} | {job['status']} | — | — | — |")
        else:
            lines.append(f"| {case_id} | {selected['selection_quantile']} | {job['seed']} | {job['status']} | "
                         f"{job['initial_iou']:.4f} → {job['final_iou']:.4f} | "
                         f"{job['heldout_initial_iou']:.4f} → {job['heldout_final_iou']:.4f} | {job['surface_improvement']:.1%} |")
lines += ['', '## 失败详情', '']
for r in records:
    if r['status'] != 'PASS':
        reason = ', '.join(r.get('failed_checks', [])) or r['status']
        if r.get('fit_root') and (Path(r['fit_root'])/'failure.json').exists():
            reason = json.loads((Path(r['fit_root'])/'failure.json').read_text()).get('error', reason)
        lines.append(f"- {r['case']}：{reason}")
lines += ['', '病例在拟合前按完整牙位、非镜像和 latent 距离分位固定，失败不替换。',
          '这是跨训练病例的同模型合成自洽验证；latent 距离分位不代表解剖多样性，seed 也不代表独立患者。',
          '统计的 PASS 指预设几何、投影、梯度及完整性门槛，不能解释为唯一 latent 恢复或未见病例/真实照片泛化。']
if args.surface_review:
    lines += ['', '本报告对所有已完成拟合统一使用 surface_review_v1 的有限距离复评。',
              '保留原样本、最近 32 个候选三角形及验收门槛；仅修正非有限距离，原始评价和拟合输出不覆盖。',
              '原始评价状态保留在 summary.json 的 original_status；原始病例状态见 cases_original。']
assessed_cases = {}
for selected in plan['selection']['selected']:
    name = Path(selected['case']).stem
    selected_records = [r for r in records if r['case']==name]
    assessed_cases[name] = 'PASS' if len(selected_records)==len(plan['definition']['seeds']) and all(r['status']=='PASS' for r in selected_records) else 'FAIL'
(output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(output/'summary.json').write_text(json.dumps({'status': state['status'], 'selection': plan['selection']['selected'],
    'surface_metric_review_used': args.surface_review, 'cases_assessed': assessed_cases,
    'cases_original': {k: v['status'] for k, v in state['cases'].items()}, 'records': records}, indent=2))
evaluated = [r for r in records if 'surface_improvement' in r]
if evaluated:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout='constrained')
    gates = plan['expansion_config']['acceptance']
    fields = [([(r['final_iou']-r['initial_iou'])/(1-r['initial_iou']) for r in evaluated],
               'Fitting-view IoU error reduction', gates['min_active_iou_error_reduction_fraction']),
              ([(r['heldout_final_iou']-r['heldout_initial_iou'])/(1-r['heldout_initial_iou']) for r in evaluated],
               'Held-out IoU error reduction', gates['min_heldout_iou_error_reduction_fraction']),
              ([r['surface_improvement'] for r in evaluated], '3D mean surface error reduction',
               gates['min_surface_mean_improvement_fraction'])]
    for ax, (values, title, threshold) in zip(axes, fields):
        ax.bar(np.arange(len(evaluated)), np.asarray(values)*100,
               color=['#29805b' if r['status']=='PASS' else '#b84c43' for r in evaluated])
        ax.axhline(threshold*100, color='#333333', linestyle='--', label=f'gate {threshold:.0%}')
        ax.set_xticks(np.arange(len(evaluated)), [f"{r['case'].replace('_upper','')}\nseed {r['seed']}" for r in evaluated],
                      rotation=65, ha='right', fontsize=9)
        ax.set_ylabel('reduction (%)'); ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Cross-case Shape-only: 14 teeth / 280 dimensions / 0.25 training sigma', fontsize=13)
    fig.savefig(output/'metrics.png', dpi=155)
    plt.close(fig)
print(json.dumps({'report': str(output/'report.md'), 'cases': {k: v['status'] for k,v in state['cases'].items()}}))
