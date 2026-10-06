# SemanticXY 匹配约定与实现

2026-10-04。算法版本 `partial_soft_semanticxy_v1`，数据约定仍为 `dual_arch_static_semantic/1.0.0`。本步骤完成多视图牙齿重建的匹配模块，**不启动正式 DMM 训练**。正式训练安排在所需源码修改和扩展完成之后。

实现直接位于 [DMM 的 semanticxy.py](../third_party/DMM/dmm/semanticxy.py)，默认参数完整保存在 [semanticxy_static_v1.json](../configs/semanticxy_static_v1.json)。本文件明确语义、XY、采样、质量和未匹配处理，并约束渲染器与拟合器的接入方式。

## 1 论文定义与本项目选择

ICCV 2025 论文第 4.2.1–4.2.2 节、式 4–7 使用 DMM 混合权重生成软语义，渲染由几何投影得到的 XY，通过 OT 建立对应后计算 Warp 的 L1 残差。论文第 4.2.4 节还专门处理完全遮挡导致梯度路径中断的问题。[本地论文](D:/文献/参数牙/Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_ICCV_2025_paper.pdf)

长距离对应和 XY 导数的基本思路来自论文引用的 [DROT 原论文](https://cg.cs.tsinghua.edu.cn/people/~kun/2022DROT/paper_DROT.pdf)。以下固定 28 通道、条件语义、归一化、采样配额、部分传输未匹配槽、对称 Warp、损失 reduction 和参数数值是**本项目的工程选择**，不是 ICCV 论文已给出或验证的最优配置。当前实现也不是复现其脸部遮挡恢复策略。

## 2 五项约定摘要

| 项目 | 本版本固定定义 |
| --- | --- |
| 语义权重 | OT 特征使用固定 28 维条件牙齿软语义；语义距离为 L1/2，权重 4；XY L1 距离权重 1 |
| XY 归一化 | 像素中心为整数；减去 `((W-1)/2,(H-1)/2)`，两个轴同除以 `sqrt(W²+H²)` |
| 采样 | 按 dominant FDI 分层，每侧每层至多 16 个代表，最多 448 个；保留真实软语义；大层按前景质量进行确定性系统重采样 |
| 质量归一化 | 每像素前景质量为牙齿概率和，统一除以该视图有效像素数 V；两侧不分别强制总质量为 1；点数变化不改变面积总量 |
| 未匹配区域 | 部分传输允许进入未匹配槽，单边单位质量代价 0.75；保留完整有效区的逐像素语义损失；ignore 不参与；空预测必须报告可见性恢复需求 |

其中“质量”指 OT 的 mass，即所代表的前景面积，不是图像质量评分。

## 3 输入和语义特征

预测输入 `RenderedSemanticXY`：

- `semantics`：H×W×29 浮点概率。通道 0 为背景，随后为上颌 11–17、21–27，下颌 31–37、41–47。有效像素非负、有限且和为 1。
- `xy_pixels`：H×W×2，未乘 alpha 的几何投影 XY 属性图，单位为拟合图像像素。不得把 premultiplied XY 直接交给匹配器。
- `xy_source="projected_geometry"`：记录几何来源。拟合模式同时要求 `xy_pixels.requires_grad=True`。

目标使用现有 `Observation` 的 FDI mask、valid 和 view_id。255 是 ignore，0 是确认的非牙齿区域。presence 及相机已经由 scene loader 校验；匹配器不根据当前可见区域更改牙位存在性。

令 P 的 28 个牙齿概率为 p，定义前景质量与条件软语义：

```text
alpha(x) = sum_k p_k(x)
s_k(x)   = p_k(x) / alpha(x),   alpha(x) > 0
```

目标牙齿像素 alpha=1，s 为固定 28 通道中的 one-hot；目标背景 alpha=0。预测的 gum/background 不独立进入 OT；它们仍参与场景深度遮挡和逐像素语义监督。预测牙龈表面若带有少量牙齿概率，会按其真实 alpha 贡献少量前景质量，不用阈值把它抹掉。

按 dominant FDI 分层只用于分配采样配额，**不会把匹配特征硬化为 argmax 标签**，也不限制传输只能发生在同一 FDI。不同牙位间的匹配由软语义代价抑制。该版本不根据可见牙数重新编号或重新缩放语义距离。

单对样本的代价：

```text
C(i,j) = 4 * 0.5 * sum_k |s_pred(i,k) - s_target(j,k)|
       + 1 * (|u_pred(i)-u_target(j)| + |v_pred(i)-v_target(j)|)
```

两种不同 one-hot 的语义代价为 4。放弃源、目标各一单位质量的总代价为 1.5，因此不会为了强制完成匹配而把高置信度错误牙位拉过去。同一 one-hot 的语义代价为 0；在图像范围内，归一化 XY 的最大 L1 距离不超过 sqrt(2)，允许建立大位移对应。这是默认权重的尺度解释，不是硬匹配保证；熵正则会产生软分配。

软语义很不确定时，质量可能主要进入未匹配槽。返回 `LOW_MATCH_MASS` 提醒调用方检查语义和初始化，不能把很小的 Warp 项单独当作拟合成功。

## 4 XY 与梯度约定

```text
D = sqrt(W*W + H*H)
u = (x_pixel - (W-1)/2) / D
v = (y_pixel - (H-1)/2) / D
```

目标 XY 来自目标像素中心。预测 XY 必须来自 mesh/表面点的屏幕投影及渲染属性插值，并保留相应几何导数；背景及 ignore 区域不提供 OT 位置监督。两轴用同一分母，保持图像长宽比。不会按牙齿包围盒归一化，也不会单独中心化每颗牙。

等比例缩放时，像素位置满足 `x_new=s*(x_old+0.5)-0.5`，上述归一化保持一致。裁剪或非等比例缩放仍须按相机约定同步更新 K；本模块不修改相机。

`evaluate` 模式可显式接受 `fixed_grid_diagnostic` 做无几何梯度的诊断。拟合模式拒绝常量网格。但来源字符串和 requires_grad 只能做接口检查，不能自动证明它连接到 DMM latent。[渲染与表面梯度桥](differentiable_rendering_v1.md) 已通过 CPU 和 GPU 原生 DMM 的 `latent → SDF → 表面 → 渲染 XY → loss` 局部有限差分，以及 GPU 大三角形轮廓 AA 导数检查。细网格整图重光栅化的 AA 有限差分仍存在离散跳变，具体边界见 [GPU 验证说明](staged_fitting_gpu_v1.md)。XY 分支在每次反传中冻结材质点插值权重，避免其与投影导数抵消。

## 5 采样与面积质量

每视图每侧先取 valid 且 alpha>0 的像素；背景不占用 OT 配额。按条件语义的最大分量分为至多 28 层，每层默认配额 n=16：

1. 像素数不超过配额：保留全部像素，每点质量为 `alpha_i / V`。
2. 像素数超过配额：按该层 alpha 累积分布进行系统重采样，取 n 个代表；每个代表质量为 `sum_layer(alpha)/(n*V)`。

系统重采样的固定偏移由 seed、view_id、prediction/target 和通道共同决定。按行序排列像素，不依赖全局随机状态或迭代编号。高质量像素可能被重复选中，这是携带有限代表质量的重复，不是重复增加面积。

两侧采样后分别满足：

```text
A = sum_i a_i = sum_valid(alpha_pred) / V
B = sum_j b_j = count_valid_target_teeth / V
```

固定完整有效区 V 同时用于两侧；不按匹配成功点数、采样点数、牙位数或各自前景面积重新归一化。因此 A≠B 的面积差被保留。分层采样避免一个小牙位因大牙位占满采样预算而完全丢失，但各牙位最终仍按真实面积贡献权重，不做逐牙等质量强制平衡。

预测只保证已出现的 dominant 层得到采样。对存在但不可见、或从未成为 dominant 的牙位，不伪造对应点。每次前向重新采样和匹配；本次反向冻结离散层划分及采样索引。抽样是空间分布近似，增加配额可能改变近似对应，但不改变所代表的总面积。

## 6 部分软传输与求解

真实样本的传输矩阵为 pi，行和不超过 a、列和不超过 b。两侧分别有一个未匹配槽，真实点到未匹配槽的代价都是 kappa=0.75，槽到槽代价为 0。

为避免“全部未匹配”时槽到槽质量趋近零造成病态，显式加入备用质量 R=A+B：

```text
augmented row masses    = [a, B+R]
augmented column masses = [b, A+R]
total augmented mass   = A+B+R = 2*(A+B)
```

槽到槽的质量等于 `R + real_matched_mass`；R 不进入图像损失，也不计入匹配率。真实点间传输仍保留原来的 a、b 容量。求解器临时对两边同时除以同一 augmented 总质量，解出后乘回，不抹掉 A/B 比例。

使用熵正则的平衡 OT 扩展问题，最终 epsilon=0.05。log-domain Sinkhorn 先按 0.4、0.2、0.1、0.05 温度逐步预热。近乎分离的匹配块若造成慢收敛，使用固定一个列势的阻尼 Newton 修正列半对偶；仍检查原始行列边际，不放宽误差门槛。

默认每阶段最多 100 次 Sinkhorn 更新、最后最多 50 次 Newton，总迭代数上限 2000；最终归一化 augmented 边际 L1 误差不超过 1e-6。单次代价矩阵最多 2,000,000 项；Newton 最多 512 个真实目标变量，默认最多 448 个。超过预算、非有限值或未收敛会明确报错，不能静默跳过视图或将损失置零。

这是含未匹配槽的**平衡扩展问题实现部分传输**，不是 KL 边际松弛的 unbalanced Sinkhorn。未匹配比例有明确面积容量含义。

## 7 Warp、未匹配和语义损失

OT 只建立对应，反向传播不穿过 Sinkhorn、Newton、采样索引或 transport plan。对本次冻结 pi，令 r、c 为真实匹配块的行、列质量。使用 pi 构造双向条件重心：

```text
target_bar_i = sum_j pi_ij * target_feature_j / r_i
source_bar_j = sum_i pi_ij * source_feature_i / c_j

L_warp = 0.5 * [sum_i r_i * d(source_i,target_bar_i)
             + sum_j c_j * d(source_bar_j,target_j)]
```

d 与匹配代价使用相同的语义、XY 权重和 L1 定义。零匹配行/列贡献为零，不使用背景坐标填补重心。两侧平均避免双向重复加倍；没有再除以已匹配质量，否则只匹配少量容易区域也可能产生误导性的小损失。

源特征保留到软语义和几何 XY 的梯度，目标特征固定。这是对应更新与参数更新交替进行的对称 Warp 目标，相比论文式 7 的书写新增反向项。有限差分检查针对**冻结对应的目标**，不能把重新求解 pi 的标量数值差分与冻结 pi 的梯度混为一谈。

未匹配项与全有效区语义项：

```text
L_unmatched = kappa * [sum_i relu(a_i-r_i) + sum_j relu(b_j-c_j)]
L_dense     = 0.5 * sum_valid sum_29_channels |P_pred-P_target| / V
L_view      = L_warp + L_unmatched + L_dense
L_multiview = mean_over_all_views(L_view)
```

未匹配项在当前匹配点等于进槽质量代价，允许求解器误差范围内的差异。反向时 r/c 固定，源质量 a 保留对预测前景概率的梯度；目标质量固定。空间位置的作用来自 Warp，缺失/多余面积还由 dense 项约束。三项权重默认均为 1，完整值写入 resolved_config；先验相对权重属于后续完整优化目标，不在本模块隐式设定。

dense 使用背景加 28 个牙齿通道，不再除以 29。它对 valid 背景上的多余牙齿、目标牙齿处的漏渲染以及错误牙位都保留误差，不受 OT 抽样遗漏影响。ignore 不参与采样、面积分母或 dense。

所有观测等权平均，包括有效的空目标视图；不写死视角数量。缺少某个渲染视图、重复 view_id、完全无 valid 像素均报错。完全无 valid 像素不同于有 valid 像素但无牙齿。

## 8 空区域和异常状态

| 情形 | 行为与诊断 |
| --- | --- |
| 预测无前景、目标有牙齿 | Warp=0，目标未匹配项和 dense 保留；返回 EMPTY_PREDICTION_REQUIRES_VISIBILITY_RECOVERY |
| 预测有牙齿、目标为有效背景 | 源质量进入未匹配项，dense 惩罚多余预测；返回 EMPTY_TARGET_NEGATIVE_EVIDENCE |
| 两侧都是有效背景 | 三项均为 0，视图仍在多视图平均中 |
| 只有部分区域能对应 | 保留软匹配块，同时报告双方未匹配质量和比例 |
| 匹配量小于较小一侧质量的 1% | 返回 LOW_MATCH_MASS，不使用其单独判断成功 |
| ignore 区域 | 不匹配、不惩罚、不贡献梯度 |
| 数值不收敛或非法输入 | 明确失败，不跳过、不返回伪零损失 |

空预测时语义损失在概率输入层仍可非零，但如果渲染器没有可见几何，未必有到几何/latent 的有效梯度。`match_scene` 汇总 `needs_visibility_recovery`，后续求解器必须处理这些视图才能判定成功。本模块不生成虚假 XY，不取消双颌/牙龈遮挡，也不声明已解决完全遮挡的可见性恢复。

soft OT 和重心 Warp 具有熵偏差及重心抵消的可能性；它不是三维误差，也不保证完美对齐时严格零损失。最终需联合 dense 语义、覆盖率、三维几何评价和先验，后期细化策略另行验证。

## 9 接入方式和验证范围

将 `third_party/DMM` 加入 Python 路径后：

```python
from dmm.semanticxy import SemanticXYConfig, SemanticXYMatcher, RenderedSemanticXY

config = SemanticXYConfig.load("configs/semanticxy_static_v1.json")
matcher = SemanticXYMatcher(config)

# 由原生 scene.render_views() 提供，每个 scene.observations 中的 view 都必须存在。
rendered_by_view = {
    view_id: RenderedSemanticXY(semantic_probability_image, projected_xy_image)
    # 其余视图按相同结构加入
}
result = matcher.match_scene(rendered_by_view, scene.observations, mode="fit")
image_loss = result["loss"]
image_loss.backward()
# 求解器必须检查 result["needs_visibility_recovery"] 并保存 resolved_config 和每视图 diagnostics。
```

`match_view` 的结果还提供 `source/target` 采样点、pixel_indices、真实面积 mass、冻结的 `plan` 和三项损失，便于检查对应图。所有状态描述匹配行为，不等于重建质量验收。

项目根目录执行新增测试：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_semanticxy.py -v
```

测试仅使用人工张量与投影属性优化，不训练 DMM，不读取旧 sanity check。覆盖 28 牙位默认采样预算、软语义、长距离梯度方向、未匹配、面积守恒、点复制不改变质量、像素中心缩放一致性、冻结对应有限差分、熵正则最优性条件、空区域、ignore、动态视角和失败行为。验证记录见 [semanticxy_validation.json](semanticxy_validation.json)。

此文的 20 项验证只覆盖匹配模块。后续已接入双颌软语义 / XY 渲染与隐式表面梯度桥，见 [渲染文档](differentiable_rendering_v1.md)；CUDA 局部导数验证、分阶段优化器与可见性恢复见 [最新实现与验证](staged_fitting_gpu_v1.md)。正式训练未启动。
