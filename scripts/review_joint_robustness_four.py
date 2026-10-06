"""Audit all four-tooth robustness records; distinguish completion from accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_forward_check import PROJECT_ROOT, sha256
from run_joint_robustness_four import read, verify, fingerprint_tree


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('batch_root', type=Path); args = parser.parse_args()
    root = args.batch_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('Outside project runs')
    state = read(root/'status.json'); plan = read(root/'plan.json')
    if state['status'] != 'ROBUSTNESS_EXPERIMENTS_COMPLETED_REVIEW_REQUIRED':
        raise ValueError('Campaign not finished')
    output = root/'review'
    if output.exists():
        raise FileExistsError('Never overwrite historical review')
    verify(root)
    checks = {'frozen_engine_sources_inputs_model': True,
              'plan_matches_declared_config': plan == read(PROJECT_ROOT/'configs/joint_robustness_four.json'),
              'clean_replay_matched': read(root/'clean_replay_comparison.json')['status'] == 'REPLAY_MATCH',
              'controller_matches_snapshot': sha256(PROJECT_ROOT/'scripts/run_joint_robustness_four.py') == sha256(root/'controller_snapshot.py')}
    control = Path(plan['control_run']); config = read(control/'resolved_config.json')
    ce = read(control/'evaluation.json')
    records = []; panels = []
    for job in state['jobs']:
        base = {'id': job['id'], 'axis': job['axis'], 'phase': job['phase']}
        if 'directory' not in job:
            records.append(dict(base, status='EXECUTION_FAILED', error=job.get('error'))); continue
        directory = Path(job['directory']); fit = directory/'case/fit_input'; run = directory/'fit_attempt_01'
        contract = read(directory/'input_contract.json'); fm = read(fit/'manifest.json'); jc = read(directory/'config.json')
        prefix = job['id'] + ':'
        settings_exceptions = {'fit_input', 'warm_start_run', 'robustness'}
        if job['axis']['kind'] == 'initialization':
            settings_exceptions.add('perturbation_seed')
        elif job['axis']['kind'] == 'pose_initialization':
            settings_exceptions.update(['initial_rotation_degrees_xyz', 'initial_translation_dmm'])
        source_fit = Path(read(control/'provenance.json')['fit_input'])
        checks[prefix+'manifest_unchanged'] = sha256(fit/'manifest.json') == contract.get('final_manifest_sha256', contract['prepared_manifest_sha256'])
        checks[prefix+'truth_copy_unchanged'] = fingerprint_tree(directory/'case/truth') == contract['truth_copy_unchanged']
        checks[prefix+'single_factor_payload'] = all(contract['evidence']['unchanged_payload_files'].values()) and all(
            sha256(fit/name) == sha256(source_fit/name) for name in contract['evidence']['unchanged_payload_files'])
        checks[prefix+'input_hashes'] = all(sha256(fit/n) == d for n,d in fm['extra_input_sha256'].items()) and sha256(fit/fm['camera_file']) == fm['camera_sha256'] and all(sha256(fit/m['path']) == m['sha256'] for m in fm['masks'])
        checks[prefix+'solver_settings'] = all(jc[k] == v for k,v in config.items() if k not in settings_exceptions)
        base['input_contract'] = contract
        progress = run/'progress.jsonl'
        logs = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
        base['local_surface_root_retry_count'] = sum('STABLE_LOCAL_STEP_RETRY' in r.get('message', '') for r in logs)
        if job['phase'] != 'evaluated':
            failure = read(run/'failure.json') if (run/'failure.json').exists() else None
            records.append(dict(base, status='EXECUTION_FAILED', error=job.get('error', job.get('returncode')), solver_failure=failure)); continue
        ev = read(run/'evaluation.json'); p = read(run/'provenance.json')
        fr = read(run/'fit_report.json')
        base['completed_outer_iterations'] = len(fr['iterations'])
        base['best_iteration'] = fr['best_iteration']
        base['fit_elapsed_seconds'] = fr['elapsed_seconds']
        checks[prefix+'frozen_solver'] = all(sha256(root/'engine/scripts'/n) == sha256(run/'code_snapshot'/n) == d for n,d in p['code_sha256'].items())
        checks[prefix+'original_gates'] = ev['acceptance_limits'] == ce['acceptance_limits']
        checks[prefix+'truth_not_read_by_solver'] = not any(p[k] for k in ['active_truth_codes_read', 'ground_truth_pose_read', 'source_meshes_read', 'depth_maps_read'])
        if job['id'] != 'clean_replay':
            warm = directory/'warmup'; wp = read(warm/'fit_attempt_01/provenance.json')
            wr = read(warm/'fit_attempt_01/warmup_report.json')
            wg = read(warm/'fit_attempt_01/warmup_gradient_check.json')
            base['warmup_diagnostics'] = {'gradient_max_relative_error': wg['max_relative_error'],
                                         'iterations': len(wr['iterations']), 'best_iteration': wr['best_iteration'],
                                         'final_observed_iou': wr['active_mean_tooth_iou']}
            checks[prefix+'warmup_shape_fixed_and_gradient'] = wr['fixed_perturbed_meshes_unchanged'] and not wr['active_codes_optimized'] and wg['status'] == 'POSE_WARMUP_GRADIENT_PASS'
            checks[prefix+'fresh_image_only_warmup'] = not any(wp[k] for k in ['active_truth_codes_read', 'ground_truth_pose_read', 'source_meshes_read', 'depth_maps_read']) and wp['fit_manifest_sha256'] == fm['warm_start_manifest_sha256'] == sha256(warm/'fit_input/manifest.json')
            wc = read(warm/'config.json'); original_wc = read(Path(config['warm_start_run'])/'resolved_config.json')
            checks[prefix+'warmup_settings'] = all(wc[k] == v for k,v in original_wc.items() if k not in settings_exceptions)
        base.update(status=ev['status'], active_labels=ev['active_labels'], failed_checks=ev['failed_checks'], checks=ev['checks'],
                    clean_fit_iou=ev['final_active_mean_tooth_iou'], heldout_iou=ev['heldout']['final']['active_mean_tooth_iou'],
                    observed_target_iou=ev['observed_target_iou'], pose_errors=ev['pose_errors'],
                    canonical=ev['surface_aggregate']['canonical'], world=ev['surface_aggregate']['world'],
                    surface_per_tooth=ev['surface_per_tooth'], geometry=ev['geometry'], gradient_max_relative_error=ev['gradient_max_relative_error'],
                    newly_failed_checks=sorted(set(ev['failed_checks'])-set(ce['failed_checks'])),
                    recovered_checks=sorted(set(ce['failed_checks'])-set(ev['failed_checks'])),
                    change_from_clean_control={
                        'clean_iou': ev['final_active_mean_tooth_iou']-ce['final_active_mean_tooth_iou'],
                        'heldout_iou': ev['heldout']['final']['active_mean_tooth_iou']-ce['heldout']['final']['active_mean_tooth_iou'],
                        'canonical_final_ratio': ev['surface_aggregate']['canonical']['final']/ce['surface_aggregate']['canonical']['final'],
                        'rotation_degrees': ev['pose_errors']['final']['rotation_degrees']-ce['pose_errors']['final']['rotation_degrees'],
                        'translation_dmm': ev['pose_errors']['final']['translation_dmm']-ce['pose_errors']['final']['translation_dmm']})
        records.append(base); panels.append((job['id'], directory, ev))
    if not all(checks.values()):
        raise ValueError(f'Integrity failed: {[k for k,v in checks.items() if not v]}')
    output.mkdir()
    shutil.copyfile(Path(__file__), output/'review_script_snapshot.py')
    perturbations = [r for r in records if r['id'] != 'clean_replay']
    evaluated = sum(r['status'] != 'EXECUTION_FAILED' for r in perturbations)
    passes = sum(r['status'] == 'JOINT_MINIMAL_PASS' for r in perturbations)
    status = 'ROBUSTNESS_PASS' if ce['status'] == 'JOINT_MINIMAL_PASS' and passes == len(perturbations) else 'ROBUSTNESS_NOT_ESTABLISHED'
    report = {'status': status, 'diagnostic_execution': 'COMPLETED' if evaluated == len(perturbations) else 'PARTIAL',
              'control_status': ce['status'], 'perturbation_count': len(perturbations), 'evaluated_count': evaluated,
              'absolute_joint_pass_count': passes, 'execution_failed_count': len(perturbations)-evaluated,
              'records': records, 'integrity_checks': checks, 'scope': plan['scope'], 'deferred': plan['deferred'],
              'view_drop_weight_note': plan['view_drop_weight_note'], 'clean_replay': read(root/'clean_replay_comparison.json'),
              'legacy_scope_note': 'Frozen evaluator prose says two teeth; numerical active_labels, dimensions, arrays and this report use four teeth.'}
    (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    original = Path(read(control/'provenance.json')['fit_input']); truth = original.parent/'truth'
    clean = read(truth/'clean_fit_manifest.json'); tm = read(truth/'manifest.json')
    targets = [(m['camera'], np.asarray(Image.open(truth/m['path'])), 'clean_final') for m in clean['masks']]
    targets.append((read(truth/tm['heldout_camera_file'])['name'], np.asarray(Image.open(truth/tm['heldout_mask_file'])), 'heldout_final'))
    fig, axes = plt.subplots(len(targets), len(panels), figsize=(3.1*len(panels), 10), squeeze=False)
    for row, (name, target, folder) in enumerate(targets):
        predictions = [np.asarray(Image.open(directory/'fit_attempt_01/renders'/folder/f'{name}_predicted_labels.png')) for _,directory,_ in panels]
        union = target > 0
        for prediction in predictions:
            union |= prediction > 0
        yy, xx = np.nonzero(union); x0,x1 = max(0,xx.min()-8), xx.max()+9; y0,y1 = max(0,yy.min()-8), yy.max()+9
        for column, ((identifier, _, ev), prediction) in enumerate(zip(panels, predictions)):
            rgb = np.full((*target.shape,3),20,dtype=np.uint8)
            rgb[(target > 0) & (prediction > 0)] = (225,225,225)
            rgb[(target > 0) & (prediction > 0) & (target != prediction)] = (250,190,48)
            rgb[(target > 0) & (prediction == 0)] = (242,70,83)
            rgb[(target == 0) & (prediction > 0)] = (60,146,245)
            scores = [np.count_nonzero((target == k) & (prediction == k))/np.count_nonzero((target == k) | (prediction == k)) for k in ev['active_labels']]
            ax = axes[row,column]; ax.imshow(rgb[y0:y1,x0:x1], interpolation='nearest'); ax.axis('off')
            ax.set_title(f'{identifier}\n{name}: {np.mean(scores):.4f}', fontsize=9)
    fig.suptitle('Four-tooth FINAL predictions vs CLEAN observations (postfit only)\nWhite: same FDI | Yellow: wrong FDI | Red: target only | Blue: prediction only', fontsize=13)
    fig.subplots_adjust(top=.9, bottom=.02, left=.01, right=.99, hspace=.3, wspace=.05)
    fig.savefig(output/'clean_observation_review.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    identifiers = [p[0] for p in panels]; x = np.arange(len(panels))
    specs = [('Clean 3-view IoU',[e['final_active_mean_tooth_iou'] for _,_,e in panels],.93),
             ('65 deg holdout IoU',[e['heldout']['final']['active_mean_tooth_iou'] for _,_,e in panels],None),
             ('Rotation error (deg)',[e['pose_errors']['final']['rotation_degrees'] for _,_,e in panels],1),
             ('Translation error (DMM)',[e['pose_errors']['final']['translation_dmm'] for _,_,e in panels],.01)]
    for ax,(title,values,threshold) in zip(axes.flat,specs):
        ax.bar(x,values,color=['#718096']+['#3182ce']*(len(x)-1)); ax.set_title(title); ax.set_xticks(x,identifiers,rotation=30,ha='right',fontsize=8)
        if threshold is not None:
            ax.axhline(threshold,color='#c53030',ls='--',label=f'gate {threshold}'); ax.legend(fontsize=8)
        if 'IoU' in title:
            ax.set_ylim(min(.85,min(values)*.95),1)
    for ax,frame in zip(axes.flat[4:],['canonical','world']):
        ax.bar(x-.18,[e['surface_aggregate'][frame]['initial'] for _,_,e in panels],width=.36,label='initial',color='#a0aec0')
        ax.bar(x+.18,[e['surface_aggregate'][frame]['final'] for _,_,e in panels],width=.36,label='final',color='#3182ce')
        ax.set_title(f'{frame} mean surface error (DMM)'); ax.set_xticks(x,identifiers,rotation=30,ha='right',fontsize=8); ax.legend(fontsize=8)
    fig.suptitle('Original Joint gates retained; diagnostic experiment completion is separate from accuracy')
    fig.savefig(output/'metrics.png',dpi=160); plt.close(fig)
    rows = []
    for r in records:
        if r['status'] == 'EXECUTION_FAILED':
            rows.append(f'| {r["id"]} | 执行失败 | — | — | — | — | — | {r["phase"]} |'); continue
        pe = r['pose_errors']['final']
        rows.append(f'| {r["id"]} | {r["observed_target_iou"]["final"]:.4f} | {r["clean_fit_iou"]:.4f} | {r["heldout_iou"]:.4f} | {pe["rotation_degrees"]:.3f} | {pe["translation_dmm"]:.5f} | {r["canonical"]["mean_improvement_fraction"]:.1%} | {", ".join(r["failed_checks"]) or "无"} |')
    text = f'''# 第 4 项：四牙 Joint 小规模单因素检查

## 状态

诊断执行 **{report['diagnostic_execution']}**，6 项扰动评价完成 {evaluated} 项，绝对 Joint 门槛通过 {passes} 项，执行失败 {len(perturbations)-evaluated} 项。稳健性结论 **{status}**。干净控制本身为 **{ce['status']}**；本次保留全部失败，不为通过门槛调参。

## 冻结协议

- 基线：`{control}`。病例 01328DDN，FDI 11/12/21/22，80 latent + 6DoF；已知 0/±30° 相机、65° 留出相机，仅渲染四颗活动牙。
- 原始 population prior（排除源病例及其镜像行、5% 协方差收缩、单位权重）、交替 shape→pose、零像素边界容差、24 轮/每块 24 nfev、局部步长/停止/验收阈值保持不变。每牙原训练尺度、sampling box、模型均保持原值。
- 历史归档复制到本批次 `engine/scripts`；子运行都在 `engine/runs`。干净重放先运行，IoU/pose/canonical/code 的预设比较阈值写在计划，全部匹配后才启动扰动。重放检查证明复现，不代表精度通过。
- 两个 shape seed：202/303，均为 GT code + 原训练尺度的 0.25σ 独立高斯扰动；原 seed101 为控制。GT code 仅用于合成输入准备，fitter 不读取 GT。
- pose：原旋转向量 [7,-5,9]° 和平移 [.08,-.06,.05] 同时取反；两者模长不变、shape seed101 不变。角度是旋转向量分量，不是 Euler。
- pose/code 边界继续以各自初值为中心，绝对可行区间随初值平移；本检查包含当前求解流程的这一行为，未改为无约束或共同绝对边界。
- 掩码：逐牙 3×3 Chebyshev 核、1 像素腐蚀/膨胀；膨胀保留所有原内部标签，新增像素按距原牙最近分配，平距按较小 FDI。
- 删除正面：仅 ±30° 提供给 warmup 与 Joint。保留原未归一化图像项，因此视角减少也改变 prior 相对权重；这是现有整体流程敏感性检查。
- 每个扰动重新运行同一图像 pose warmup；干净重放复用原图像 warmup。warmup、Joint 的形状/掩码/相机/pose 数值输入对应一致。GT pose、GT mesh、留出相机与干净参考掩码均不进入拟合/最佳选择/停止。
- 最佳结果仍按观测轮廓 soft-L1 + population prior 选择。最大两进程并发，每进程 CPU threads=4；预算一致但提前停止会导致实际迭代数不同。

## 输入视觉核查

![输入扰动](../input_review.png)

腐蚀/膨胀仅改变边界；丢失正面明确标记未提供。像素计数、单因素文件差异见各 job 的 `input_contract.json`。

图像为 640×480。左斜侧 FDI22 的原可见区域仅 180 像素，腐蚀后 99 像素（减少 45%）；右斜侧 FDI12 从 425 减至 246 像素（减少约 42.1%）。因此“1 像素”对小面积、受自遮挡的牙区域相对较强；它是预设的系统边界偏差，不代表随机分割噪声分布。所有 FDI 仍有非空观测，未重新选图或病例。

## 事后独立评价

评价使用全部三幅干净原图、65° 留出图、GT pose、canonical/world 表面。`observed` 为实际拟合输入 IoU；`clean` 为干净三视角 IoU，二者分开报告。删除正面的条件仍在拟合结束后独立评价正面。

验收沿用原值：旋转 ≤1°、平移 ≤0.01 DMM、clean IoU ≥0.93 且提升 ≥0.05、留出 IoU 提升 ≥0.05、canonical/world 均值分别降低 ≥20%/80%、逐牙 canonical 均值/P95 不恶化超过 5%/10%、梯度最大相对误差 ≤0.005、每牙闭合且单组件。表面距离是双向各 6000 面积采样点对最近 32 候选三角形，非完整 Hausdorff；DMM 单位未标定为 mm。

| 条件 | observed IoU | clean IoU | 65° IoU | 旋转 ° | 平移 DMM | canonical 降低 | 失败项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

![独立指标](metrics.png)

![四视角最终预测与干净观测](clean_observation_review.png)

## 可追溯性与范围

源码、控制与 warmup 历史输出、DMM 源码、模型、输入、truth 复制件、门槛完整性全部通过；详细逐牙结果、相对控制增量、新增/恢复失败项见 [summary.json](summary.json)。[计划](../plan.json)、[状态](../status.json)、[重放比较](../clean_replay_comparison.json)、每项独立 warmup/fit/eval 日志均保留。

冻结 evaluator 旧版文字仍有“两牙”字样；其 `active_labels`、80 latent/86 维输入与逐牙数值实际为四牙，本报告按动态标签解释。

这是一例训练病例、四牙、小扰动、已知相机检查，不能说明整牙弓/真实照片/未知相机/跨病例泛化。遮挡检查需先实现显式有效区域契约；本轮未做局部遮挡、错 FDI、缺牙。对话第 4 项“稳健性”与源计划文档 M4“prior 校准和视角扩展”不同。
'''
    (output/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':status,'evaluated':evaluated,'pass':passes,'report':str(output/'report.md')}))


if __name__ == '__main__':
    main()
