"""Report every robustness job, clean-view geometry gates, and frozen provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_forward_check import PROJECT_ROOT,sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch_root',type=Path)
    args=parser.parse_args()
    root=args.batch_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('batch must be in project runs')
    output=root/'review'
    if output.exists():
        raise FileExistsError('review already exists')
    state=read(root/'status.json'); plan=read(root/'plan.json')
    if state['status']!='ROBUSTNESS_EXPERIMENTS_COMPLETED_REVIEW_REQUIRED':
        raise ValueError('batch not finished')
    control=Path(plan['control_run']); ce=read(control/'evaluation.json'); cp=read(control/'provenance.json')
    config=read(control/'resolved_config.json')
    hashes=read(root/'code_sha256.json')
    checks={'batch_sources_and_snapshots_unchanged':all(sha256(PROJECT_ROOT/'scripts'/n)==sha256(root/'code_snapshot'/n)==d for n,d in hashes.items()),
        'control_fitting_sources_unchanged':all(sha256(PROJECT_ROOT/'scripts'/n)==d for n,d in cp['code_sha256'].items()),
        'dmm_sources_unchanged':all(sha256(Path(n))==d for n,d in cp['dmm_source_sha256'].items()),
        'frozen_model_unchanged':sha256(Path(config['experiment'])/'ModelParameters'/f'dmm_{config["checkpoint"]}.pth')==cp['model_sha256']}
    records=[]; successes=[]
    for item in state['jobs']:
        directory=Path(item['directory']); fit=directory/'case/fit_input'
        contract=read(fit/'robustness_contract.json'); manifest=read(fit/'manifest.json')
        jc=read(directory/'config.json')
        checks[item['id']+'_input_contract']=sha256(fit/'manifest.json')==contract['prepared_manifest_sha256'] and all(contract['evidence']['unchanged_input_files'].values())
        checks[item['id']+'_input_file_hashes']=all(sha256(fit/n)==d for n,d in manifest['extra_input_sha256'].items()) and sha256(fit/manifest['camera_file'])==manifest['camera_sha256'] and all(sha256(fit/m['path'])==m['sha256'] for m in manifest['masks'])
        checks[item['id']+'_original_settings']=all(jc[k]==v for k,v in config.items() if k!='perturbation_seed')
        base={'id':item['id'],'axis':item['axis'],'phase':item['phase'],'run_root':str(directory/'fit_attempt_01'),
            'input_evidence':contract['evidence']}
        if item['phase']!='evaluated':
            base.update(status='EXECUTION_FAILED',failed_checks=[item['phase']])
            records.append(base); continue
        run=directory/'fit_attempt_01'; ev=read(run/'evaluation.json'); p=read(run/'provenance.json')
        checks[item['id']+'_codes_unchanged']=all(sha256(PROJECT_ROOT/'scripts'/n)==sha256(run/'code_snapshot'/n)==d for n,d in p['code_sha256'].items())
        checks[item['id']+'_contract_unchanged']=sha256(fit/'robustness_contract.json')==p['robustness_contract_sha256']
        checks[item['id']+'_limits_unchanged']=ev['acceptance_limits']==ce['acceptance_limits']
        checks[item['id']+'_no_geometry_truth_in_fitter']=not any(p[k] for k in ['active_truth_codes_read','ground_truth_pose_read','source_meshes_read','depth_maps_read'])
        base.update(status=ev['status'],failed_checks=ev['failed_checks'],checks=ev['checks'],pose_errors=ev['pose_errors'],
            clean_fit_iou=ev['final_active_mean_tooth_iou'],heldout_iou=ev['heldout']['final']['active_mean_tooth_iou'],
            observed_target_iou=ev['observed_target_iou'],canonical=ev['surface_aggregate']['canonical'],world=ev['surface_aggregate']['world'],
            gradient_max_relative_error=ev['gradient_max_relative_error'],
            change_from_control={'clean_fit_iou':ev['final_active_mean_tooth_iou']-ce['final_active_mean_tooth_iou'],
                'heldout_iou':ev['heldout']['final']['active_mean_tooth_iou']-ce['heldout']['final']['active_mean_tooth_iou'],
                'canonical_final_error_ratio':ev['surface_aggregate']['canonical']['final']/ce['surface_aggregate']['canonical']['final'],
                'rotation_degrees':ev['pose_errors']['final']['rotation_degrees']-ce['pose_errors']['final']['rotation_degrees'],
                'translation_dmm':ev['pose_errors']['final']['translation_dmm']-ce['pose_errors']['final']['translation_dmm']})
        records.append(base); successes.append((item['id'],run,ev))
    if not all(checks.values()):
        raise ValueError(f'integrity checks failed: {[k for k,v in checks.items() if not v]}')
    output.mkdir()
    pass_count=sum(r['status']=='JOINT_MINIMAL_PASS' for r in records)
    evaluated=len(successes)
    # A failed unperturbed control prevents an overall robustness-pass claim.
    overall='ROBUSTNESS_PILOT_PASS' if ce['status']=='JOINT_MINIMAL_PASS' and pass_count==len(records) else 'ROBUSTNESS_PILOT_NOT_ESTABLISHED'
    report={'status':overall,'control_status':ce['status'],'control_run':str(control),'perturbation_jobs':len(records),
        'evaluated_jobs':evaluated,'absolute_joint_pass_jobs':pass_count,'execution_failed_jobs':len(records)-evaluated,
        'records':records,'integrity_checks':checks,'scope':plan['scope'],
        'interpretation':'small controlled pilot, no statistical success-rate claim; control already fails pose gates',
        'prior_weight_note':'two-view job preserves original unnormalized residual sum, so fewer observations increase relative damping weight'}
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    panels=[('control',control,ce)]+successes
    original=Path(cp['fit_input']); fm=read(original/'manifest.json'); tr=original.parent/'truth'; tm=read(tr/'manifest.json')
    front=next(c['camera'] for c in fm['masks'] if c['camera']=='az_+0')
    target_front=np.asarray(Image.open(original/next(m['path'] for m in fm['masks'] if m['camera']==front)))
    target_heldout=np.asarray(Image.open(tr/tm['heldout_mask_file']))
    heldout_name=read(tr/tm['heldout_camera_file'])['name']
    fig,axes=plt.subplots(2,len(panels),figsize=(3*len(panels),7),squeeze=False)
    for row,(name,target) in enumerate([(front,target_front),(heldout_name,target_heldout)]):
        predictions=[]
        for index,(_,run,_) in enumerate(panels):
            folder=('final' if index==0 else 'clean_final') if row==0 else 'heldout_final'
            predictions.append(np.asarray(Image.open(run/'renders'/folder/f'{name}_predicted_labels.png')))
        union=target>0
        for prediction in predictions:
            union|=prediction>0
        yy,xx=np.nonzero(union); x0,x1=max(0,xx.min()-8),min(target.shape[1],xx.max()+9); y0,y1=max(0,yy.min()-8),min(target.shape[0],yy.max()+9)
        for column,((identifier,_,ev),prediction) in enumerate(zip(panels,predictions)):
            rgb=np.full((*target.shape,3),20,dtype=np.uint8)
            rgb[(target>0)&(prediction>0)]=(225,225,225)
            rgb[(target>0)&(prediction>0)&(target!=prediction)]=(250,190,48)
            rgb[(target>0)&(prediction==0)]=(242,70,83)
            rgb[(target==0)&(prediction>0)]=(60,146,245)
            scores=[np.count_nonzero((target==k)&(prediction==k))/np.count_nonzero((target==k)|(prediction==k)) for k in ev['active_labels']]
            ax=axes[row,column]; ax.imshow(rgb[y0:y1,x0:x1],interpolation='nearest'); ax.axis('off')
            ax.set_title(f'{identifier}\n{name}: IoU {np.mean(scores):.4f}',fontsize=10)
    fig.suptitle('Robustness: FINAL predictions vs CLEAN original observations\nWhite: same FDI | Yellow: wrong FDI | Red: target only | Blue: prediction only',fontsize=13)
    fig.subplots_adjust(top=.84,bottom=.03,left=.01,right=.99,wspace=.1,hspace=.4)
    fig.savefig(output/'clean_observation_review.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    identifiers=[p[0] for p in panels]; x=np.arange(len(panels))
    specs=[('Clean three-view IoU',[e['final_active_mean_tooth_iou'] for _,_,e in panels],.93),
        ('65 deg heldout IoU',[e['heldout']['final']['active_mean_tooth_iou'] for _,_,e in panels],None),
        ('Rotation error (deg)',[e['pose_errors']['final']['rotation_degrees'] for _,_,e in panels],1),
        ('Translation error (DMM)',[e['pose_errors']['final']['translation_dmm'] for _,_,e in panels],.01)]
    for ax,(title,values,threshold) in zip(axes.flat,specs):
        ax.bar(x,values,color=['#718096']+['#3182ce']*(len(x)-1)); ax.set_title(title); ax.set_xticks(x,identifiers,rotation=25,ha='right',fontsize=8)
        if threshold is not None:
            ax.axhline(threshold,color='#c53030',ls='--',label=f'gate {threshold}'); ax.legend(fontsize=8)
        if 'IoU' in title:
            ax.set_ylim(min(.85,min(values)*.95),1)
    for ax,frame in zip(axes.flat[4:],['canonical','world']):
        ax.bar(x-.18,[e['surface_aggregate'][frame]['initial'] for _,_,e in panels],width=.36,label='initial',color='#a0aec0')
        ax.bar(x+.18,[e['surface_aggregate'][frame]['final'] for _,_,e in panels],width=.36,label='final',color='#3182ce')
        ax.set_title(f'{frame} mean surface error (DMM)'); ax.set_xticks(x,identifiers,rotation=25,ha='right',fontsize=8); ax.legend(fontsize=8)
    fig.suptitle('Robustness pilot: original absolute gates; all failed cases retained',fontsize=14)
    fig.savefig(output/'metrics.png',dpi=160); plt.close(fig)
    rows=[]
    for r in records:
        if r['status']=='EXECUTION_FAILED':
            rows.append(f'| {r["id"]} | 执行失败 | — | — | — | — | — | {r["phase"]} |'); continue
        pe=r['pose_errors']['final']
        rows.append(f'| {r["id"]} | {r["status"].removeprefix("JOINT_MINIMAL_")} | {r["clean_fit_iou"]:.4f} | {r["heldout_iou"]:.4f} | {pe["rotation_degrees"]:.3f} | {pe["translation_dmm"]:.5f} | {r["canonical"]["mean_improvement_fraction"]:.1%} | {", ".join(r["failed_checks"]) or "无"} |')
    text=f'''# 第 4 项：两牙 Joint 最小稳健性实验

## 结论

状态 **{overall}**。5 个扰动子实验中，独立评价完成 {evaluated} 个，绝对 Joint 门槛通过 {pass_count} 个，执行失败 {len(records)-evaluated} 个。干净控制本身为 **{ce['status']}**，因此不能据此宣布 Joint 稳健性验证通过。

## 固定协议

- 控制：`{control.name}`，同训练病例、11/21 两牙、40 latent + 6DoF、pose 预优化后 Joint。
- 初始化：0.25 sigma，新增 seed202/303，与原 seed101 合计 3 个初始化；初始共享 pose 保持 [7,-5,9]° 旋转向量、[0.08,-0.06,0.05] 平移。
- 边界偏差：seed101，逐牙 3×3 核的 1 像素腐蚀/膨胀；膨胀冲突按距原牙区域最近分配，距离相同时采用较小 FDI，保留所有原内部标签。这是系统边界偏差，不是随机像素噪声或真实 SAM 错误模型。
- 视角减少：seed101，删除正面，拟合仅使用左右 30°；没有使用删除视角做初始化、轮廓对应、最佳选择或停止。
- 每次只改变一个因素。solver、warm-up、代码尺度、初始阻尼、局部/全局步长、预算和验收阈值保持一致，所有失败保留。没有增加统计 prior，也没有扩大牙数。
- 源文档的 M4 指 prior 校准与视角扩展；本报告按本次会话的“第 4 项稳健性”组织，不等于完成文档的 M4。

## 独立评价

准备输入已目视核查；1 像素扰动仅改变边界，视角减少明确标记未提供的正面观测。

![输入扰动](../input_review.png)

评价在拟合结束后才读取 truth：全部 **3 个干净原视角**、65° 预留视角、canonical/world 表面及 GT pose。即使少一视角，正面也只在事后计入干净 3 视角评价；受扰动掩码的拟合 IoU 另存 `observed_target_iou`，与干净 IoU 分开。

绝对门槛保持：旋转 ≤1°、平移 ≤0.01 DMM、干净三视角 IoU ≥0.93 且提升 ≥0.05、65° IoU 提升 ≥0.05、canonical 表面均值减少 ≥20%、world 减少 ≥80%，逐牙 canonical 均值/P95 不恶化超过 5%/10%，梯度与网格完整性通过。表面距离采用双向各 6000 面积采样点、最近 32 候选三角形；不是完整 Hausdorff，DMM 单位不直接等于 mm。

控制最终：干净三视角 IoU {ce['final_active_mean_tooth_iou']:.4f}，65° IoU {ce['heldout']['final']['active_mean_tooth_iou']:.4f}，旋转 {ce['pose_errors']['final']['rotation_degrees']:.3f}°，平移 {ce['pose_errors']['final']['translation_dmm']:.5f}。

| 条件 | 验收 | 干净 3 视角 IoU | 65° IoU | 旋转 ° | 平移 DMM | canonical 误差减少 | 失败项 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

![独立指标](metrics.png)

![最终预测对干净观测](clean_observation_review.png)

初始化 seed202 的 3D 核查：左列 canonical、右列 world；各列初始/最终共用初始 P95 色标，列间色标不同。图示为单向每牙 1500 点，仅供空间核查，验收仍使用独立双向指标。

![seed202 3D 核查](../jobs/init_seed202/fit_attempt_01/geometry_review.png)

腐蚀 1 像素的 3D 核查，采用同样的采样和列内共享色标规则：

![腐蚀边界的 3D 核查](../jobs/mask_erode1/fit_attempt_01/geometry_review.png)

## 限制及证据

只有一个训练病例、两颗牙、小扰动及已知相机；3 个 seed 不支持统计成功率或真实照片鲁棒性结论。没有测试未知相机、嘴唇/牙龈遮挡、缺牙或错 FDI，未完成 prior 校准。两视角实验沿用原未归一化残差求和，因此观测量减少也会提高阻尼项的相对权重；它测试当前整个求解流程的敏感性，不能将变化完全归因于纯视角几何信息。

全部源码/输入/门槛/冻结模型完整性复核通过。所有 job、失败项、扰动像素数、模型与代码哈希和相对控制的指标变化见 [summary.json](summary.json)、[计划](../plan.json)、[进度](../status.json)。子实验各自保存 `case/fit_input/robustness_contract.json`、`fit_attempt_01/evaluation.json`、完整 pose/latent/网格/渲染及进程日志。
'''
    (output/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':overall,'report':str(output/'report.md'),'evaluated':evaluated,'absolute_joint_pass':pass_count}))


if __name__=='__main__':
    main()
