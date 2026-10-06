"""Summarize a failed minimal Joint and a passed oracle-pose diagnostic control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_forward_check import PROJECT_ROOT, sha256
from joint_common import pose_matrix


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('joint_run',type=Path)
    parser.add_argument('--shape-control',type=Path,required=True)
    args=parser.parse_args()
    root=args.joint_run.resolve(strict=True)
    control=args.shape_control.resolve(strict=True)
    if any(not p.is_relative_to((PROJECT_ROOT/'runs').resolve()) for p in [root,control]):
        raise ValueError('all results must be in project runs')
    if any((root/n).exists() for n in ['review.md','summary_verification.json']):
        raise FileExistsError('summary already exists')
    ev,ce=read(root/'evaluation.json'),read(control/'evaluation.json')
    if ev['status']!='JOINT_MINIMAL_FAIL' or ce['status']!='SHAPE_ONLY_SUBSET_PASS':
        raise ValueError('this failure diagnosis requires failed Joint and passed matched Shape-only')
    provenance,cp=read(root/'provenance.json'),read(control/'provenance.json')
    config=read(root/'resolved_config.json')
    report=read(root/'fit_report.json')
    fit,cf=Path(provenance['fit_input']),Path(cp['fit_input'])
    fm,cm=read(fit/'manifest.json'),read(cf/'manifest.json')
    verification={
        'joint_fit_code_unchanged':all(sha256(PROJECT_ROOT/'scripts'/n)==d and sha256(root/'code_snapshot'/n)==d for n,d in provenance['code_sha256'].items()),
        'frozen_dmm_sources_unchanged':all(sha256(Path(n))==d for n,d in provenance['dmm_source_sha256'].items()),
        'frozen_model_unchanged':sha256(Path(config['experiment'])/'ModelParameters'/f'dmm_{config["checkpoint"]}.pth')==provenance['model_sha256'],
        'joint_input_manifest_unchanged':sha256(fit/'manifest.json')==provenance['fit_manifest_sha256'],
        'control_input_manifest_unchanged':sha256(cf/'manifest.json')==cp['fit_manifest_sha256'],
        'control_active_labels_match':fm['active_labels']==cm['active_labels']==ce['active_labels']==ev['active_labels'],
        'control_initial_codes_identical':sha256(fit/fm['initial_codes'])==sha256(cf/cm['initial_codes']),
        'control_cameras_identical':sha256(fit/fm['camera_file'])==sha256(cf/cm['camera_file']),
        'control_sampling_boxes_identical':sha256(fit/fm['sampling_boxes'])==sha256(cf/cm['sampling_boxes']),
        'control_training_scales_identical':sha256(fit/fm['training_scales'])==sha256(cf/cm['training_scales']),
        'control_masks_identical':fm['masks']==cm['masks'] and all(sha256(fit/m['path'])==sha256(cf/m['path'])==m['sha256'] for m in fm['masks']),
        'truth_codes_and_meshes_excluded_from_joint_fitter':not any(provenance[k] for k in ['active_truth_codes_read','ground_truth_pose_read','source_meshes_read','depth_maps_read']),
    }
    jt,ct=read(fit.parent/'truth/manifest.json'),read(cf.parent/'truth/manifest.json')
    verification['control_heldout_identical']=all(sha256(fit.parent/'truth'/jt[k])==sha256(cf.parent/'truth'/ct[k]) for k in ['heldout_camera_file','heldout_mask_file'])
    if not all(verification.values()):
        raise ValueError(f'integrity or matched-control check failed: {verification}')
    gt=np.array(read(fit.parent/'truth'/jt['pose_file'])['arch_to_world'])
    pose_trace=[]
    for name,record in [('initial',report['initial']),('centroid',report['centroid'])]+[(f'outer_{r["outer_iteration"]}',r) for r in report['iterations']]:
        matrix=pose_matrix(np.array(record['pose']))
        pose_trace.append({'stage':name,'fit_iou':record['active_mean_tooth_iou'],
            'rotation_error_degrees':float(np.rad2deg(Rotation.from_matrix(matrix[:3,:3]@gt[:3,:3].T).magnitude())),
            'translation_error_dmm':float(np.linalg.norm(matrix[:3,3]-gt[:3,3]))})
    diagnosis=read(root/'observability_v2.json')
    retry_count=sum(len(r['rejected_trial_solves']) for r in report['iterations'])
    summary={'checks':verification,'pose_trace_post_fit_only':pose_trace,'local_trial_rejections':retry_count,
        'joint_status':ev['status'],'shape_control_status':ce['status'],'shape_control':str(control),
        'joint_source_files_checked':len(provenance['code_sha256']),'dmm_source_files_checked':len(provenance['dmm_source_sha256'])}
    (root/'summary_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    can,world=ev['surface_aggregate']['canonical'],ev['surface_aggregate']['world']
    pe=ev['pose_errors']
    rows=[
        f'| 拟合逐牙平均 IoU | {ev["initial_active_mean_tooth_iou"]:.6f} | {ev["final_active_mean_tooth_iou"]:.6f} | ≥ 0.93，且提升 ≥ 0.05 | 通过 |',
        f'| 65° 预留视角 IoU | {ev["heldout"]["initial"]["active_mean_tooth_iou"]:.6f} | {ev["heldout"]["final"]["active_mean_tooth_iou"]:.6f} | 提升 ≥ 0.05 | 通过 |',
        f'| 旋转误差 | {pe["initial"]["rotation_degrees"]:.4f}° | {pe["final"]["rotation_degrees"]:.4f}° | ≤ 1° | 失败 |',
        f'| 平移误差 | {pe["initial"]["translation_dmm"]:.6f} | {pe["final"]["translation_dmm"]:.6f} | ≤ 0.01 | 失败 |',
        f'| canonical 平均表面误差 | {can["initial"]:.8f} | {can["final"]:.8f} | 减少 ≥ 20% | 失败，增大 {-can["mean_improvement_fraction"]:.1%} |',
        f'| world 平均表面误差 | {world["initial"]:.8f} | {world["final"]:.8f} | 减少 ≥ 80% | 通过，减少 {world["mean_improvement_fraction"]:.1%} |',
    ]
    text=f'''# 最小 Joint 实验核查

## 结论

实验已完成，独立验收状态为 **{ev['status']}**。46 维联合链路能改善图像和世界坐标中的几何，但未同时恢复正确的共享位姿与 canonical 形状。原始门槛及失败输出保留。

## 范围与求解

- DMM epoch295，同一训练病例 `01328DDN_upper.npz`；11、21 两颗中切牙，40 个 latent + 一个共享 6DoF 位姿。
- 场景只含这两颗牙，重新生成正面及左右 30° 掩码；相机已知。无牙龈、嘴唇或其他牙齿。与先前完整 14 牙观测的 Shape-only 场景不同。
- latent 扰动：seed101、每维训练标准差的 0.25 倍。位姿旋转向量分量为 [7,-5,9]°，平移 [0.08,-0.06,0.05] DMM 单位；不是 Euler 角。
- 图像质心粗对齐使用扰动后的初始网格，然后同时求解全部 46 维。采用双向逐牙可见轮廓对应、隐式 SDF 表面微分和 SO(3) 位姿导数；外层刷新网格、遮挡与对应关系。
- 初始码阻尼 1.0 和标准差缩放是数值设置，未校准为统计 shape prior。当前目标仍不是论文完整的软 SemanticXY 损失。
- 最终选择第 {report['best_iteration']} 轮的最佳拟合视角 IoU；共 {len(report['iterations'])} 轮，{retry_count} 次局部求根失败通过减半步长恢复，完整日志保留。预留视角和真值不参与选择。

## 独立指标

| 指标 | 初始 | 最终 | 预设门槛 | 结果 |
| --- | ---: | ---: | --- | --- |
{chr(10).join(rows)}

两牙的 canonical 均值和 P95 非恶化检查也失败；最终网格均封闭且单连通。表面误差单位为 DMM 坐标单位，不能直接视为 mm。验收采用双向各 6000 面积采样点及最近 32 个候选三角形距离，不是完整 Hausdorff 距离。

Joint 残差 Jacobian 的有限差分最大相对误差为 {ev['gradient_max_relative_error']:.3e}，通过 0.005 门槛。检查包含 6 个位姿轴、每牙 latent 方向及混合方向，两个步长；固定局部可见性与对应关系。该值使用向量残差 Jacobian 的相对范数，不能与旧 Shape-only 的标量 loss 导数相对误差直接排名；没有对硬 IoU 求导。

## 匹配 Shape-only 对照

对照运行：[{control.name}](../{control.name}/evaluation.json)。使用完全相同的初始码、训练尺度、采样域、相机、两牙掩码、预留视角；固定生成真值位姿。这是明确的 oracle-pose 诊断。数值设置匹配，验收沿用既有 Shape-only 扩展门槛。

- 状态：**{ce['status']}**。
- 拟合 IoU：{ce['initial_active_mean_tooth_iou']:.6f} → {ce['final_active_mean_tooth_iou']:.6f}。
- 65° 预留 IoU：{ce['heldout']['initial']['active_mean_tooth_iou']:.6f} → {ce['heldout']['final']['active_mean_tooth_iou']:.6f}。
- canonical 平均表面误差：{ce['initial_surface_mean_dmm']:.8f} → {ce['final_surface_mean_dmm']:.8f}，减少 {ce['surface_mean_improvement_fraction']:.1%}；逐牙均值及 P95 均改善。

对照通过，支持优先调查联合求解中的位姿与形状补偿、粗初始化偏差及局部极小。不能仅凭一次失败判定问题严格不可辨识。局部图像 Jacobian 条件数约 {diagnosis['joint_image_condition_number']:.0f}，pose 与 latent 子空间主夹角 {min(diagnosis['pose_latent_principal_angles_degrees']):.1f}°–{max(diagnosis['pose_latent_principal_angles_degrees']):.1f}°，表现为部分重叠，未发现近零主夹角；该分析固定当前对应和可见性，只是局部诊断。见 [observability_v2.json](observability_v2.json)。

## 可视化

Joint 图像：白色为相同 FDI，黄为错误 FDI，红为目标独有，蓝为预测独有。

![Joint 图像](semantic_review_v2.png)

左列 canonical 与右列 world 分开比较；每列初始与最终共用初始 P95 色标，列间色标不同。图示为单向每牙 1500 点，仅供空间误差核查，验收仍用上述双向指标。

![Joint 3D 误差](geometry_review_v2.png)

![固定真值位姿的匹配对照](../{control.name}/semantic_review.png)

## 完整性与后续

当前复核全部通过：{len(provenance['code_sha256'])} 个 Joint 依赖/归档代码、{len(provenance['dmm_source_sha256'])} 个 DMM 源码、模型、输入 manifest 与匹配对照输入。Joint 拟合器未读取 GT latent、GT 位姿、源网格、深度及预留视角；生成器和事后评估读取 truth。见 [summary_verification.json](summary_verification.json)、[evaluation.json](evaluation.json)、[fit_report.json](fit_report.json)。

下一步建议在同一两牙输入上增加“固定扰动 shape 的轮廓 pose warm-up → 释放全部参数”的新对照，继续使用相同验收门槛；随后评估训练码统计 prior 或全局刚体模式的约束。先验证初始化影响，再决定是否需要改变参数化。原失败基线与此前 Shape-only 输出保留。
'''
    (root/'review.md').write_text(text,encoding='utf-8')
    print(json.dumps({'joint_status':ev['status'],'control_status':ce['status'],'review':str(root/'review.md'),
        'verification_pass':all(verification.values()),'centroid_pose_error':pose_trace[1]}))


if __name__=='__main__':
    main()
