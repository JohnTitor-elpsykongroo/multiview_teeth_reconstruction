"""Post-fit comparison of centroid-only Joint and fixed-shape contour warm-up."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from joint_common import pose_matrix
from run_forward_check import PROJECT_ROOT,sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()):
        raise ValueError('run must be within project runs')
    if any((root/name).exists() for name in ['review.md','warmup_comparison.json']):
        raise FileExistsError('review already exists')
    config=read(root/'resolved_config.json'); provenance=read(root/'provenance.json')
    baseline=Path(config['baseline_run'])
    old=read(baseline/'evaluation.json'); ev=read(root/'evaluation.json')
    old_config=read(baseline/'resolved_config.json'); old_prov=read(baseline/'provenance.json')
    fit=Path(provenance['fit_input']); manifest=read(fit/'manifest.json')
    warm=read(root/'warmup_report.json'); report=read(root/'fit_report.json')
    initial=dict(np.load(root/'initial_codes.npz')); warm_codes=dict(np.load(root/'warmup_codes.npz'))
    model=Path(config['experiment'])/'ModelParameters'/f'dmm_{config["checkpoint"]}.pth'
    checks={
        'same_baseline_input_manifest':sha256(fit/'manifest.json')==provenance['fit_manifest_sha256']==old_prov['fit_manifest_sha256'],
        'same_all_original_joint_settings':all(config[key]==value for key,value in old_config.items()),
        'same_acceptance':ev['acceptance_limits']==old['acceptance_limits']==config['acceptance'],
        'baseline_fitting_sources_unchanged':all(sha256(PROJECT_ROOT/'scripts'/n)==d for n,d in old_prov['code_sha256'].items()),
        'new_fitting_sources_and_snapshots_unchanged':all(sha256(PROJECT_ROOT/'scripts'/n)==sha256(root/'code_snapshot'/n)==d for n,d in provenance['code_sha256'].items()),
        'frozen_dmm_sources_unchanged':all(sha256(Path(n))==d for n,d in provenance['dmm_source_sha256'].items()),
        'frozen_model_unchanged':sha256(model)==old_prov['model_sha256']==provenance['model_sha256'],
        'warmup_codes_equal_initial':set(initial)==set(warm_codes) and all(np.array_equal(initial[k],warm_codes[k]) for k in initial),
        'warmup_meshes_unchanged':warm['fixed_perturbed_meshes_unchanged'],
        'same_initial_canonical_surface_error':np.isclose(ev['surface_aggregate']['canonical']['initial'],old['surface_aggregate']['canonical']['initial'],rtol=0,atol=1e-12),
        'same_initial_fit_iou':ev['initial_active_mean_tooth_iou']==old['initial_active_mean_tooth_iou'],
    }
    checks={k:bool(v) for k,v in checks.items()}
    if not all(checks.values()):
        raise ValueError(f'comparison integrity checks failed: {checks}')
    truth=read(fit.parent/'truth/manifest.json')
    gt=np.array(read(fit.parent/'truth'/truth['pose_file'])['arch_to_world'])
    def pose_error(pose):
        matrix=pose_matrix(np.array(pose))
        return {'rotation_degrees':float(np.rad2deg(Rotation.from_matrix(matrix[:3,:3]@gt[:3,:3].T).magnitude())),
            'translation_dmm':float(np.linalg.norm(matrix[:3,3]-gt[:3,3]))}
    records=[]
    for stage,record in [('initial',report['initial']),('centroid',report['centroid']),('warmup_best',warm),('joint_final',report['final'])]:
        records.append({'stage':stage,'fit_iou':record['active_mean_tooth_iou'],**pose_error(record['pose'])})
    warm_trace=[{'outer_iteration':r['outer_iteration'],'fit_iou':r['active_mean_tooth_iou'],**pose_error(r['pose'])} for r in warm['iterations']]
    warm_gradient=read(root/'warmup_gradient_check.json')
    retries=sum(len(r['rejected_trial_solves']) for r in report['iterations'])
    result={'status':ev['status'],'baseline_status':old['status'],'baseline_run':str(baseline),'integrity_checks':checks,
        'stage_pose_errors_post_fit_only':records,'warmup_pose_trace_post_fit_only':warm_trace,
        'warmup_stop_reason':warm['stop_reason'],'warmup_iterations':len(warm['iterations']),
        'warmup_gradient_max_relative_error':warm_gradient['max_relative_error'],
        'joint_local_trial_rejections':retries,
        'evaluation_source_sha256':{n:sha256(PROJECT_ROOT/'scripts'/n) for n in ['evaluate_joint.py','evaluate_shape_only.py','surface_metric_utils.py','review_joint_warmup.py']}}
    (root/'warmup_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    can,world=ev['surface_aggregate']['canonical'],ev['surface_aggregate']['world']
    can_desc=f'降低 {can["mean_improvement_fraction"]:.1%}' if can['mean_improvement_fraction']>=0 else f'增大 {-can["mean_improvement_fraction"]:.1%}'
    stage_rows='\n'.join(f'| {r["stage"]} | {r["fit_iou"]:.6f} | {r["rotation_degrees"]:.4f}° | {r["translation_dmm"]:.6f} |' for r in records)
    next_step=('下一步可增加重复初始化，核查这次通过是否稳定，再扩大活动牙数。' if ev['status']=='JOINT_MINIMAL_PASS' else
        '仅增加轮廓 pose 预优化仍未满足 Joint 验收。下一步应在新对照中调查训练码统计 prior、全局刚体模式的约束或分块更新策略，继续保持本次输入和验收门槛；不能依据高 IoU 放宽几何门槛。')
    conclusion=('预优化后的 Joint 同时满足位姿及形状门槛，本次两牙单初始化实验通过。' if ev['status']=='JOINT_MINIMAL_PASS' else
        '预优化后的 Joint 仍有独立指标未达标；失败输出保留。')
    text=f'''# 固定扰动 shape 的轮廓 pose 预优化 → Joint

## 结果

**{ev['status']}**。{conclusion}

使用原两牙 case、seed101、0.25 sigma、三台已知相机、同一初始位姿；活动牙 11/21，共 40 latent + 6DoF。原输入包及全部原 Joint 设置一致，唯一新增求解阶段是固定扰动 shape 的轮廓 pose 预优化。相机及掩码固定，无其他牙齿或牙龈。

## 阶段比较

| 阶段 | 三视角逐牙平均 IoU | 旋转误差 | 平移误差（DMM 单位） |
| --- | ---: | ---: | ---: |
{stage_rows}

阶段位姿误差在拟合完成后才通过独立真值计算，不进入停止或选择策略。质心初始对齐后，固定扰动初始码和网格，仅求解 pose 的 6 维；每轮重新渲染并更新双向逐牙可见轮廓。预优化最多 60 轮，至少 10 轮，连续 12 轮未超过最佳 IoU 1e-5 即停止；实际 {len(warm['iterations'])} 轮，最佳第 {warm['best_iteration']} 轮，停止原因 `{warm['stop_reason']}`。不设置基于真值的停止条件，也不要求固定扰动 shape 的 IoU 达到真值 shape 水平。

保存最佳拟合视角 IoU 的预优化位姿，然后释放全部 46 维；初始码阻尼、尺度、边界、局部步长、outer/inner 预算、重试和 Joint 最佳选择策略均与旧基线相同。Joint 共 {len(report['iterations'])} 轮，最佳第 {report['best_iteration']} 轮，{retries} 次局部求根失败经减小步长恢复。

## 与旧 Joint 对照

| 指标 | 旧 Joint 最终 | 预优化后 Joint 最终 | 原门槛 |
| --- | ---: | ---: | --- |
| 拟合 IoU | {old['final_active_mean_tooth_iou']:.6f} | {ev['final_active_mean_tooth_iou']:.6f} | ≥ 0.93，且相对原始输入提升 ≥ 0.05 |
| 65° 预留 IoU | {old['heldout']['final']['active_mean_tooth_iou']:.6f} | {ev['heldout']['final']['active_mean_tooth_iou']:.6f} | 相对原始输入提升 ≥ 0.05 |
| 旋转误差 | {old['pose_errors']['final']['rotation_degrees']:.4f}° | {ev['pose_errors']['final']['rotation_degrees']:.4f}° | ≤ 1° |
| 平移误差 | {old['pose_errors']['final']['translation_dmm']:.6f} | {ev['pose_errors']['final']['translation_dmm']:.6f} | ≤ 0.01 DMM 单位 |
| canonical 表面平均误差 | {old['surface_aggregate']['canonical']['final']:.8f} | {can['final']:.8f} | 相对同一初始 shape 减少 ≥ 20% |
| world 表面平均误差 | {old['surface_aggregate']['world']['final']:.8f} | {world['final']:.8f} | 相对同一原始输入减少 ≥ 80% |

本次 canonical 表面平均误差 {can_desc}；world 平均误差降低 {world['mean_improvement_fraction']:.1%}。失败检查：{', '.join(ev['failed_checks']) or '无'}。逐牙均值/P95 非恶化门槛、网格封闭连通、梯度和 pose/codes 同时更新检查沿用原配置。

固定 mesh 的 6 维 pose 残差 Jacobian 有限差分最大相对误差 {warm_gradient['max_relative_error']:.3e}；释放后完整 Joint 为 {ev['gradient_max_relative_error']:.3e}，门槛均为 0.005。两次检查固定局部可见性及对应；硬栅格 IoU 不求导。

## 可视化

图像：白色相同 FDI、黄色错误 FDI、红色目标独有、蓝色预测独有；65° 为预留视角。

![初始与最终语义](semantic_review.png)

左列为 canonical 形状，右列为施加拟合位姿后的 world 几何。每列初始/最终共享初始 P95 色标，列间色标不同。图示为单向每牙 1500 点；验收使用双向各 6000 面积采样点及最近 32 候选三角形距离，非完整 Hausdorff。DMM 单位不直接等同于毫米。

![3D 误差](geometry_review.png)

预优化阶段三视角叠图（仅前景重叠，详细 FDI 以语义图为准）：

![预优化叠图](renders/warmup/overlay_sheet.png)

## 完整性与解释

复核全部通过：原输入 manifest、全部原 Joint 设置及验收门槛、原拟合代码、新拟合归档代码、冻结 DMM 源码与权重、预优化 codes 等于初始 codes、固定网格未变、初始指标一致。旧基线未覆盖。预优化与 Joint 拟合器只读取 fit_input 与冻结 DMM 权重，不读取 GT codes/pose/meshes、深度或预留观测。

{next_step}

本次仍为单训练病例、两牙、单扰动种子、已知相机、无噪声的合成实验，未验证真实照片或唯一 latent 恢复。结果入口：[evaluation.json](evaluation.json)、[warmup_report.json](warmup_report.json)、[warmup_comparison.json](warmup_comparison.json)、[fit_report.json](fit_report.json)、[旧基线报告](../{baseline.name}/review.md)。
'''
    (root/'review.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':ev['status'],'report':str(root/'review.md'),'stage_pose_errors':records,'integrity_checks_pass':all(checks.values())}))


if __name__=='__main__':
    main()
