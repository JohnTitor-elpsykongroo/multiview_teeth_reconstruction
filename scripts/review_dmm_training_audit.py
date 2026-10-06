"""Add mirror-gauge diagnostics, scientific plots and a retraining audit report."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_dmm_training import read,stats,rigid_fit
from run_forward_check import PROJECT_ROOT,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit_root',type=Path)
    args=parser.parse_args();root=args.audit_root.resolve(strict=True)
    if not root.is_relative_to((PROJECT_ROOT/'runs').resolve()) or any((root/n).exists() for n in ['review.md','training_audit_review.png','mirror_gauge.json']):
        raise ValueError('invalid run or report already exists')
    status=read(root/'audit_status.json');ds=read(root/'dataset_summary.json');latent=read(root/'latent_regularization.json')
    fields=read(root/'rigid_field_probe.json');alignment=read(root/'alignment_cases.json');npz=read(root/'npz_sample_audit.json')
    digests=read(root/'input_sha256.json')
    if not all(status['checks'].values()) or not all(sha256(Path(p))==h for p,h in digests.items()):
        raise ValueError('audit integrity failed')
    dataset=Path(next(p for p in digests if p.endswith('avg_centroids.txt'))).parent
    train=sorted(read(dataset/'splits/train_split.json'));labels=sorted(int(k) for k in ds['avg_centroid_per_tooth_augmented_mean_difference'])
    original_points={k:[] for k in labels};mirror_points={k:[] for k in labels};rows=[]
    avg=np.loadtxt(dataset/'avg_centroids.txt')
    for name in train:
        with (dataset/'SdfSamples'/Path(name).with_suffix('.pkl')).open('rb') as f:centers=pickle.load(f)
        points=np.stack(list(centers.values())).astype(float);ids=list(centers)
        R,t,e=rigid_fit(points,avg[ids])
        mirrored='__mirror' in name
        rows.append({'case':name,'mirrored':mirrored,'points':len(points),'extra_fit_rotation_degrees':float(np.rad2deg(Rotation.from_matrix(R).magnitude())),
            'extra_fit_translation_norm':float(np.linalg.norm(t)),'residual_mean':float(e.mean())})
        for label,point in centers.items():(mirror_points if mirrored else original_points)[label].append(point)
    orig_mean=np.stack([np.mean(original_points[k],0) for k in labels]);mirror_mean=np.stack([np.mean(mirror_points[k],0) for k in labels])
    aug_mean=np.stack([np.mean(original_points[k]+mirror_points[k],0) for k in labels])
    data={'note':'extra rigid fit to mean describes reference/augmentation anatomy; not pose ground truth or alignment error',
        'original_mean':orig_mean.tolist(),'mirrored_mean':mirror_mean.tolist(),'augmented_mean':aug_mean.tolist(),'labels':labels,
        'mean_mirrored_minus_original':(mirror_mean-orig_mean).mean(0).tolist(),
        'extra_rigid_fit_to_original_average':{kind:{key:stats([r[key] for r in rows if r['mirrored']==flag]) for key in ['extra_fit_rotation_degrees','extra_fit_translation_norm','residual_mean']}
            for kind,flag in [('original',False),('mirror',True)]},'per_case':rows}
    (root/'mirror_gauge.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    train_align=[r for r in alignment if r['split']=='train']
    field_summary={key:stats([f[source][subkey] if subkey else f[source] for f in fields]) for key,source,subkey in [
        ('single_rigid_residual_median','residual_to_best_single_rigid','median'),
        ('translation_field_std_norm','pointwise_translation_std_norm',None),
        ('latent_perturbation_rigid_rotation_degrees','latent_perturbation_deformation_change','best_rigid_rotation_degrees'),
        ('latent_perturbation_rigid_translation_norm','latent_perturbation_deformation_change','best_rigid_translation_norm')]}
    (root/'field_summary.json').write_text(json.dumps(field_summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for p,label,color,marker in [(orig_mean,'Original mean / model center target','#126bb5','o'),(mirror_mean,'Mirrored mean','#ed8032','x'),(aug_mean,'Actual augmented mean','#228b55','s')]:
        axes[0,0].scatter(p[:,0],p[:,1],s=35,label=label,c=color,marker=marker)
    for k,p in zip(labels,orig_mean):axes[0,0].annotate(str(k),p[:2],xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0,0].set(xlabel='x (DMM units)',ylabel='y (DMM units)',title='Center target and mirrored training distribution')
    axes[0,0].axis('equal');axes[0,0].legend(fontsize=8)
    axes[0,1].hist([r['scale'] for r in train_align],bins=32,color='#126bb5',alpha=.85)
    axes[0,1].set(xlabel='Per-case similarity scale (DMM / source unit)',ylabel='Original training cases',title='526 independently scaled original cases')
    for kind,flag,color in [('Original',False,'#126bb5'),('Mirror',True,'#ed8032')]:
        axes[1,0].hist([r['extra_fit_translation_norm'] for r in rows if r['mirrored']==flag],bins=32,alpha=.6,label=kind,color=color)
    axes[1,0].set(xlabel='Extra rigid translation to model center target (DMM)',ylabel='Cases',title='Reference / augmentation descriptor; not GT pose error')
    axes[1,0].legend()
    axes[1,1].scatter([f['latent_perturbation_deformation_change']['best_rigid_rotation_degrees'] for f in fields],
        [f['latent_perturbation_deformation_change']['best_rigid_translation_norm'] for f in fields],s=18,alpha=.65,c='#6846a5')
    axes[1,1].set(xlabel='Best rigid rotation of internal field change (deg)',ylabel='Best rigid translation of internal field change (DMM)',
        title='0.25 std latent perturbation; 126 case-tooth probes')
    fig.suptitle('Frozen DMM training audit: alignment, scale, mirror gauge and deformation field',fontsize=13)
    fig.savefig(root/'training_audit_review.png',dpi=160);plt.close(fig)
    s=ds['train_alignment_statistics'];mirror_fit=data['extra_rigid_fit_to_original_average']['mirror']
    lp=latent['per_label'];logstats=latent['last_16_logged_batches_statistics']
    sources=Path('D:/WorkSpace/Dental/DMM');prep=Path('D:/WorkSpace/Dental/teeth3DS_to_DMM')
    report=f'''# DMM 重训前核查：对齐、尺度、变换分支与 latent 正则

## 结论

按后续要修改模型并重训的方向整理本次依据。当前未发现冻结训练数据的坐标错配、非法旋转或镜像 FDI 错误。比“数据坏了”更明确的上游问题是：**镜像增强与模板质心的参考系偏置、逐点 SE(3) 变换场与 shape/pose 混合、latent 正则及损失归一化缺少清晰契约**。这些足以成为重训设计的检查项，但本审计不能证明它们单独造成了某次 Joint 的位姿误差。

本次为只读核查，不执行重训，不改变冻结数据、模型或实验 checkpoint。结果保存在独立运行目录。

## 证据范围

- 所有 1052 个训练条目（526 原例 + 526 镜像）与 49 个测试条目的 PKL 质心/哈希，575 个原始病例变换及 aligned_centroids/冻结哈希。
- 23 个 NPZ 的数组和内容哈希抽检：包含尺度、残差、越界率的预选极值/中位数病例、3 个测试病例及对应镜像；不按模型拟合成绩选样。
- epoch295 的所有训练 latent；10 个预选原训练病例的 126 个牙位，各 128 个表面坐标探针。所有内部场查询和固定 0.25σ 扰动均为诊断，不进行拟合。
- 当前训练/模型/数据预处理源码、配置、完整 train.log、best 权重及 latent。开始和结束 SHA-256 一致，详细列表见 [input_sha256.json](input_sha256.json)。当前训练脚本未随历史 WSL 训练归档，不能以当前文件哈希证明训练时脚本完全相同；参数、模型结构和 best 指标另由日志/checkpoint 核对。
- 未重新遍历原始 OBJ 全部顶点来独立重做每例对齐；全量检查是已冻结的变换、质心与它们的对应关系。未重哈希全部大 NPZ，只对选出的 23 个做内容哈希与数组检查。患者关联未独立从源患者清单重建，病例文件名 train/test 无交集。

## 1. 数据对齐：未发现实现链路错配

当前 profile 为 `strict_fixed`，使用固定参考病例 `6XSU5W4B_upper` 的牙齿面积加权质心，通过 FDI 对应做 **相似变换 x′ = sRx + t**。不是已经运行的迭代 GPA，也不是逐颗牙独立摆正。[求解源码]({(prep/'src/teeth3ds_dmm/alignment/similarity.py').as_posix()})。

- 526 个原训练病例：512 PASS、14 WARN；全部旋转 det≈+1，最大 det 偏差 {s['det_error']['max']:.2e}，最大正交误差 {s['orthogonality_error_recomputed']['max']:.2e}。
- PKL 与 aligned_centroids 一致到 float32 精度；重算对参考质心 RMSE 与保存值一致。RMSE 中位数 {s['rmse_reference']['median']:.5f}、P95 {s['rmse_reference']['p95']:.5f}、最大 {s['rmse_reference']['max']:.5f} DMM 单位。
- 保存的 arch_orientation 指标换算角度，中位数 {s['arch_orientation_angle_degrees']['median']:.3f}°、P95 {s['arch_orientation_angle_degrees']['p95']:.3f}°、最大 {s['arch_orientation_angle_degrees']['max']:.3f}°。这是冠向/参考方向描述，不是整个位姿真值误差。
- 镜像 PKL 全量一致；抽检 NPZ 的 x 反射、法线 x 反射、11↔21 等牙位交换均一致。未发现标签反射错误。

这里的质心残差混合了病例解剖差异、牙列变化和拟合误差；不能据此把正常病例判为“错对齐”。原始病例对训练平均质心做额外刚体拟合，旋转中位数仅 {s['extra_rigid_fit_to_mean_rotation_degrees']['median']:.3f}°，不足以支持“全局姿态大量未消除”的判断。

## 2. 尺度：独立归一化，物理牙列尺寸不能直接恢复

每例独立估计 s，训练范围 **{s['scale']['min']:.5f}–{s['scale']['max']:.5f}**，中位数 {s['scale']['median']:.5f} DMM/source-unit。参考固定尺度为 0.02329766，但病例实际尺度不固定；相对参考范围 {s['scale_relative_reference']['min']:.3f}–{s['scale_relative_reference']['max']:.3f}。

这会归一化牙列整体大小，而保留牙列比例、局部形态和布局差异。它是预处理设计选择，不是发现了单位混用。原单位未在此次审计中独立确认为 mm，因此所有重建数值仍应报告 DMM 单位。需要物理尺寸时，必须明确源坐标单位，保存/使用每例 s、R、t，并在新模型或照片拟合中明确尺度参数和尺度证据；不能用一个统一常数把所有 DMM 距离换为 mm。

抽检表面并非全部位于 [-1,1]³；原变换的少量越界并未自动截断。统计和案例列表见 [npz_sample_audit.json](npz_sample_audit.json)。新采样域和推理域应覆盖真实牙齿范围，不可假设 cube 内包含全部表面。

## 3. 镜像与平均质心：需要统一模板参考系

训练含 526 镜像，镜像操作为 x→−x 后交换左右 FDI，不再对齐到原参考病例。[镜像源码]({(prep/'src/teeth3ds_dmm/dmm_export/mirror.py').as_posix()})。

`avg_centroids.txt` 则只由 526 原训练病例的 aligned_centroids 计算。[导出源码]({(prep/'src/teeth3ds_dmm/dmm_export/stages.py').as_posix()})。与原例均值最大分量差 {ds['avg_centroid_max_abs_difference_original_mean']:.2e}，符合导出契约；与实际 1052 条目增强均值最大分量差 **{ds['avg_centroid_max_abs_difference_augmented_mean']:.5f}**，每牙向量差 **0.028–0.033 DMM**。

镜像均值相对原例均值的平均偏移向量为 {np.array(data['mean_mirrored_minus_original']).round(6).tolist()}。镜像病例若额外刚体拟合到原模板质心，平移范数中位数 {mirror_fit['extra_fit_translation_norm']['median']:.5f}，原例相应中位数 {s['extra_rigid_fit_to_mean_translation_norm']['median']:.5f} DMM。这是参考与增强偏置的描述，不能解读为真值 pose 错误。

**重训前必须选择并固定一致策略**：使用对称的 canonical/template 参考与增强后的训练均值，或镜像后重新执行同一 canonical 对齐，并重新计算训练模板质心。不能只替换旧 avg 文件；应生成新 dataset profile 和 provenance。无需删除镜像来消除这种偏置。

## 4. 所谓刚体分支实际上是逐点 SE(3) 场

牙齿网络输出 8 维：[ωx,ωy,ωz,vx,vy,vz,Δs,blend-logit]，HyperNetwork 由同一个 20 维 z 生成网络权重，网络又接受空间坐标 x。故 R、T 是 R(x,z)、T(x,z)，并非每颗牙仅一个 R(z)、T(z)。[DMM]({(sources/'networks/dmm_net.py').as_posix()})、[DeformNet]({(sources/'networks/deform_net.py').as_posix()})。

126 个训练病例/牙位探针中，内部映射相对最佳单一刚体的残差中位数再取跨探针中位数为 {field_summary['single_rigid_residual_median']['median']:.5f} DMM，最大为 {field_summary['single_rigid_residual_median']['max']:.5f}。这确认其非刚性能力，不等于模型实现错误。

固定坐标处对 z 加 0.25σ 的预定方向扰动，内部映射变化的最佳刚体部分：旋转中位数 **{field_summary['latent_perturbation_rigid_rotation_degrees']['median']:.3f}°**、最大 {field_summary['latent_perturbation_rigid_rotation_degrees']['max']:.3f}°；平移中位数 **{field_summary['latent_perturbation_rigid_translation_norm']['median']:.5f}**、最大 {field_summary['latent_perturbation_rigid_translation_norm']['max']:.5f} DMM。该量位于内部映射坐标系，不是最终提取牙齿网格的 pose 误差，也不能与 Joint 的门槛直接等价比较。

约束方面：center loss 仅在每牙一个质心点上监督映射到平均质心，不能唯一确定旋转或整个坐标场；`grad_deform` 对位移 Jacobian 做范数惩罚，不能消除恒定平移的自由度；SDF correction 也由同一个 z 控制。[损失实现]({(sources/'networks/loss.py').as_posix()})。

**改模型的优先方向**：把每牙显式刚体、内在形状和共享上颌 pose 分开；若保留非刚性场，要明确它的用途和幅度/平滑约束。还必须定义局部牙齿变换的共同刚体模式怎样归入共享 pose；仅增加一个独立刚体 head 并不会自动保证解耦。

数值缺陷：`screw_axis_to_rt` 直接除以 ||ω||，没有零角度分支；输入全零 6 维参数实际产生非有限输出。本次已训练场探针全部有限，不能据此声称历史训练出现 NaN。新模型需要稳定的 SE(3) exp-map 小角度实现；普通 clamp 分母不足以代替正确的极限展开。[数学实现]({(sources/'utils/math.py').as_posix()})。

## 5. latent 正则：系数大，但并非已校准 prior

训练硬编码 `1e6 * mean(z²)`，每标签求均值后对 batch 出现标签平均；初始化标准差 0.002。没有独立 shape/rigid latent 正则配置或由数据校准的白化 prior。原合成拟合中的“相对初始码阻尼 1.0”与这个训练正则不是同一项。

- epoch295 全条目/全标签估算的正则均值为 **{latent['estimated_full_row_full_label_regularizer']:.4f}**。
- 最后 16 个日志采样 batch：latent 项中位数 **{logstats['latent']['median']:.4f}**，total 中位数 **{logstats['total']['median']:.4f}**。这不是完整 batch 分解，不能用它声称正则必然过强或不足。
- FDI11 每维 std 范围 {lp['11']['dimension_std']['min']:.2e}–{lp['11']['dimension_std']['max']:.2e}，cov 条件数 {lp['11']['covariance_condition_number']:.1f}；FDI21 cov 条件数 {lp['21']['covariance_condition_number']:.1f}。每维 std 缩放仍不消除维间相关，不能称为完整白化，也未证明高斯分布。
- FDI17 只有 {lp['17']['present_training_rows']}/1052 条目存在；缺牙条目的 latent 不能和有牙条目无区别地建立解剖 shape prior。当前 batch 标签取并集，regularizer 对该标签的全部 batch embeddings 计算，包括该牙在对应病例中缺失的行。
- HyperNetwork 首层为仿射映射，z 与首层输入权重存在相反缩放的等效关系。默认 Adam 无额外权重衰减，单独压小 z 不保证限制 decoder 敏感度或建立可比较的 latent 尺度。

**重训建议**：损失权重放入版本化配置；按牙位存在性建立统计量/正则策略；记录 shape、rigid 分支的独立正则及 decoder 敏感度。可比较零中心 L2 和 present-only 协方差白化，但需设协方差收缩/特征值下限，并在验证集上选择，不预先认定某个 λ 值最好。

## 6. 训练/损失接口的附带发现

- center loss 使用 L1 `sum`，逐病例/牙位累加，只除 batch 出现标签数，没有除 batch size/有效牙位计数。改变 ScenesPerBatch 会改变其相对权重；缺牙率也影响有效监督数量。新训练应明确按有效观测归一化。
- component 表面 loss 的 mask 外置零后对整个张量取 mean，存在牙位频率/缺失对有效权重的影响；8192 点中 4096 表面、2048 pos、2048 neg，表面为 pooled 随机采样，非逐牙固定配额。重训时可使用按有效标签/点数归一化并显式控制牙位采样。
- NPZ pos/neg 的 sdf 列均为 0、标签均为 −1，与 `offsurface_compat` 和当前 loss 的 off-surface 排斥/Eikonal 训练一致；这里不应当作带正负真值距离的 SDF 监督。若改成带符号 SDF 回归，需要新数据和新 loss 契约。[DataLoader]({(sources/'data/data_with_labels.py').as_posix()})。
- specs 的 `ClampingDistance=0.1` 未在当前训练/loader/loss 路径中使用；`GradientClipNorm` 即使配置也只是读取/日志，当前未调用裁剪。调这些键不会自动改变训练行为。
- best 为 epoch295 的最低均值训练总 loss 6.02149，无独立验证循环；不能作为最佳泛化 checkpoint。原日志学习率阶梯和 specs 一致，最终训练段 LR 6.25e−6，未看到有证据支持的 LR 配置读取错误。[训练脚本]({(sources/'train_dmm.py').as_posix()})。

## 7. 为后续改模型与重训准备的顺序

1. **定义参数契约**：共享 arch pose、每牙局部刚体、shape code、非刚性场/ΔSDF、presence、物理尺度分别负责什么，以及共同刚体模式的约束。
2. **新建数据版本**：统一镜像与模板参考、决定保留物理尺寸还是每例相似归一化；补齐单位及逆变换记录。保持旧冻结 profile 可复现，不就地改写。
3. **改训练接口**：稳定 SE(3)、权重可配置、center/分量 loss 按有效样本归一化、存在性相关的 latent prior；记录各分项 loss 和敏感度。
4. **设独立验证**：以病例/患者为单位划分，镜像随原例同组；验证未见病例的 code fitting 和表面/布局误差，不只选训练 loss。无需在验证病例上预先保存优化过的训练 embedding。
5. **分阶段训练消融**：在同输入和固定评价流程下比较结构/参考系/prior 修改。逐步验证 forward、pose-only、shape-only、Joint；保留失败，不把多个改动一次打包后将改善归因给单项。

现在可确定要重训，但本次没有据此启动旧结构重训；尚需把将要采用的新模型契约具体化。上述实测值是旧模型/旧数据的基线，不是新模型参数的最优值。

## 图与数据

![训练核查](training_audit_review.png)

详细文件：[数据汇总](dataset_summary.json)、[逐例对齐](alignment_cases.json)、[镜像参考系分析](mirror_gauge.json)、[latent 与日志](latent_regularization.json)、[逐牙变换场探针](rigid_field_probe.json)、[完整性状态](audit_status.json)。
'''
    (root/'review.md').write_text(report,encoding='utf-8')
    (root/'review_source_sha256.json').write_text(json.dumps({str(PROJECT_ROOT/'scripts'/n):sha256(PROJECT_ROOT/'scripts'/n) for n in ['audit_dmm_training.py','review_dmm_training_audit.py']},indent=2),encoding='utf-8')
    print(json.dumps({'report':str(root/'review.md'),'mirror_extra_fit_median_t':mirror_fit['extra_fit_translation_norm']['median'],'field_summary':field_summary}))


if __name__=='__main__':main()
