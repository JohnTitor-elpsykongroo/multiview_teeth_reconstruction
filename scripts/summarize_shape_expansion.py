"""Export staged evidence, including every failed initialization and evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch_root',type=Path)
    parser.add_argument('--output-folder',default='review')
    args=parser.parse_args()
    root=args.batch_root.resolve(strict=True)
    state=json.loads((root/'status.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    if Path(args.output_folder).name!=args.output_folder:
        raise ValueError('output must be one folder name inside the batch')
    output=root/args.output_folder
    output.mkdir(exist_ok=False)
    records=[]
    for stage in plan['definition']['stages']:
        for seed in stage['seeds']:
            jid=f'{stage["id"]}_seed{seed}'
            job=state['jobs'].get(jid)
            if job is None:
                continue
            record={'stage':stage['id'],'teeth':len(stage['active_labels']),'sigma':stage['sigma'],'seed':seed,
                    'status':job['status'],'fit_root':job.get('fit_root')}
            if job['status'] in ['PASS','EVALUATION_FAIL']:
                evaluation=json.loads((Path(job['fit_root'])/'evaluation.json').read_text())
                record.update({key:evaluation[key] for key in ['initial_active_mean_tooth_iou','final_active_mean_tooth_iou',
                    'surface_mean_improvement_fraction','initial_surface_mean_dmm','final_surface_mean_dmm',
                    'active_iou_error_reduction_fraction','heldout_iou_error_reduction_fraction','gradient_max_relative_error']})
                record['heldout_initial']=evaluation['heldout']['initial']['active_mean_tooth_iou']
                record['heldout_final']=evaluation['heldout']['final']['active_mean_tooth_iou']
                record['failed_checks']=job['failed_checks']
                record['per_tooth_surface_improvement']={k:v['mean_improvement_fraction'] for k,v in evaluation['surface_per_tooth'].items()}
            else:
                failure_path=Path(job['fit_root'])/'failure.json' if job.get('fit_root') else None
                record['failure']=json.loads(failure_path.read_text()) if failure_path and failure_path.exists() else {}
            records.append(record)
    summary={'status':state.get('status'),'stages':state['stages'],'jobs':records,
             'scope':'one DMM training case, noiseless synthetic labels, known shared pose/cameras; all seeds retained',
             'acceptance':plan['definition']['acceptance'],'solver_overrides':plan['definition'].get('config_overrides',{})}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Shape-only 分阶段扩大实验','',f"运行状态：`{state.get('status')}`。病例、相机和上颌位姿固定，所有 seed 和失败均计入结果。",'',
           '| 阶段 | 牙数 | 扰动 sigma | seed | 状态 | 拟合 IoU 初始 → 最终 | 预留 IoU 初始 → 最终 | 表面误差降低 |',
           '| --- | ---: | ---: | ---: | --- | --- | --- | --- |']
    for r in records:
        fit=f"{r['initial_active_mean_tooth_iou']:.4f} → {r['final_active_mean_tooth_iou']:.4f}" if 'initial_active_mean_tooth_iou' in r else '—'
        held=f"{r['heldout_initial']:.4f} → {r['heldout_final']:.4f}" if 'heldout_initial' in r else '—'
        surface=f"{r['surface_mean_improvement_fraction']:.1%}" if 'surface_mean_improvement_fraction' in r else '—'
        lines.append(f"| {r['stage']} | {r['teeth']} | {r['sigma']} | {r['seed']} | {r['status']} | {fit} | {held} | {surface} |")
    lines+=['','阶段判定：','']+[f"- `{name}`：`{status}`" for name,status in state['stages'].items()]
    lines+=['','失败详情：','']
    for r in records:
        if r['status']!='PASS':
            lines.append(f"- `{r['stage']}/seed{r['seed']}`："+(', '.join(r.get('failed_checks',[])) or r.get('failure',{}).get('error','未完成')))
    lines+=['','该结果验证几何和投影恢复；不要求 latent 数值唯一等于真值。图像导数检查固定局部可见性和对应，尚未复现完整软语义 SemanticXY。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    evaluated=[r for r in records if 'surface_mean_improvement_fraction' in r]
    positions=np.arange(len(evaluated))
    labels=[f"{r['teeth']} teeth\n{r['sigma']} sigma / {r['seed']}" for r in evaluated]
    for ax,field,title,limit in zip(axes,['active_iou_error_reduction_fraction','heldout_iou_error_reduction_fraction',
                               'surface_mean_improvement_fraction'],['Fitting-view IoU error reduction','Held-out IoU error reduction',
                               '3D mean surface error reduction'],[.3,.2,.2]):
        ax.bar(positions,[r[field]*100 for r in evaluated],color=['#27875a' if r['status']=='PASS' else '#bf4c45' for r in evaluated])
        ax.axhline(limit*100,color='#333333',linestyle='--',label=f'gate {limit:.0%}')
        ax.set_xticks(positions,labels,rotation=75,ha='right',fontsize=8)
        ax.set_title(title,fontsize=11)
        ax.set_ylabel('reduction (%)')
        ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Shape-only staged expansion: all evaluated seeds; green PASS / red FAIL',fontsize=13)
    fig.savefig(output/'metrics.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'report':str(output/'report.md'),'summary':str(output/'summary.json'),'stages':state['stages']}))


if __name__=='__main__':
    main()
