# 第 4 项：四牙 Joint 小规模单因素检查

## 结论

本轮 **诊断执行完成**：一个干净重放、六项扰动均完成拟合和独立评价，执行失败 0 项。六项扰动按原完整 Joint 门槛 **0/6 通过**，状态 `ROBUSTNESS_NOT_ESTABLISHED`。干净基线本身仍只在旋转和平移门槛失败，所有结果按原阈值保留。

干净重放的三视角 IoU、65° IoU、pose 误差、canonical 均值、归一化最终 code 的最大差均为 **0**。最新四牙求解器复现成功；它与旧两牙稳健性批次分别记录。

运行：[joint_robustness_four_20261004T061410Z_de71c9c1](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/)。详细 [协议及结果](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/report.md)、[完整数值及逐牙指标](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/summary.json)。

## 固定条件与范围

单个已训练病例 01328DDN；FDI 11/12/21/22，80 latent + 6DoF；640×480、已知 0/±30° 三个相机，65° 留出。模型为 `20260824/best`，epoch295。仅渲染四颗牙，没有牙龈、唇、下颌或冻结背景牙。

历史四牙基线的求解器归档复制到本批次 `engine/scripts`，运行目录为 `engine/runs/jobs`。保留交替 shape→pose、排除源病例及镜像行的 population prior、5% 协方差收缩、原 prior 权重、零边界容差、24 轮/每块 24 nfev、原局部重试、停止规则和全部验收阈值。六项扰动均重新进行同一图像 pose warmup，干净重放复用原 warmup。

四类因素、六项条件：shape 初值 seed202/303（同 0.25σ，原 seed101 为控制）；共享 pose 初值同时反向（旋转向量和平移模长不变）；1 像素腐蚀/膨胀；删除正面、仅保留左右 30°。每次只改变一个因素。GT code 仅用于生成预设合成初值，GT pose/mesh、干净参考与留出视图只用于事后评价。

初值实验保持原相对边界设置，绝对可行区间随初值平移。删除视角沿用原未归一化图像项，prior 相对权重也随观测量改变。这些是当前整体求解流程的敏感性检查。

## 全部结果

`clean IoU` 对全部三幅原始干净观测评价；每个条件实际拟合的 `observed IoU` 单独记录。所有距离为 DMM 单位，未标定成 mm。

| 条件 | clean IoU | 65° IoU | 旋转误差 ° | 平移误差 DMM | canonical 均值降低 | 相对基线新增失败项 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 干净重放 | 0.982673 | 0.968043 | 2.05597 | 0.021757 | 44.32% | — |
| shape seed202 | 0.982957 | 0.968836 | 2.32346 | 0.020087 | 38.10% | 逐牙均值/P95 |
| shape seed303 | 0.986321 | 0.975497 | 1.27489 | 0.010849 | 44.16% | 逐牙均值/P95 |
| 反向 pose | 0.983585 | 0.967709 | 2.32101 | 0.024281 | 42.51% | 无 |
| 腐蚀 1 像素 | 0.902087 | 0.943511 | 1.47122 | 0.016754 | 31.74% | clean IoU |
| 膨胀 1 像素 | 0.916460 | 0.948244 | 2.68576 | 0.027642 | 33.03% | clean IoU |
| 删除正面 | 0.983859 | 0.960860 | 1.63225 | 0.018556 | 41.27% | 逐牙均值 |

全部条件仍失败于旋转 ≤1°、平移 ≤0.01 DMM。全部条件均通过解析梯度核对、闭合单组件网格、canonical 总体降低 ≥20%、world 总体降低 ≥80% 等其余公共门槛；逐牙与 clean IoU 的额外失败如表。

## 检查揭示了什么

### 1. 高 IoU 与总体表面改善会掩盖个别牙退化

- seed202 的 FDI12 均值误差增加 **13.30%**、P95 增加 **22.47%**；最终 canonical 总体误差为控制的 **1.521 倍**。
- seed303 的 FDI12/22 均值分别增加 **17.70%/31.70%**，P95 分别增加 **19.62%/23.84%**。其总体 canonical 比控制稍低、pose 也更准确，但两颗侧切牙仍退化。
- 因此本例的初始化敏感性表现需要按牙检查；三次初始化不足以估计统计成功率，也不证明 latent 唯一恢复。

### 2. 系统边界偏差是本轮最明显的图像一致性敏感项

相对控制，腐蚀/膨胀的 clean IoU 分别下降 **8.06/6.62 个百分点**，65° IoU 下降 **2.45/1.98 个百分点**。最终 canonical 总体误差分别为控制的 **1.226/1.203 倍**；两项仍比各自初始形状改善，没有新增逐牙退化门槛失败。

膨胀的 observed IoU 为 **0.95743**，clean IoU 为 **0.91646**，体现对错误边界的拟合与干净恢复质量之间的差异。腐蚀的 observed IoU 为 **0.87821**，clean 为 **0.90209**；不能将这两项都描述为高度拟合了错误目标。

“1 像素”的相对强度取决于可见面积：左斜侧 FDI22 从 180 像素降至 99（减少 45%），右斜侧 FDI12 从 425 降至 246（减少 42.1%）。所有牙仍有非空观测，保留原病例、图像和扰动定义。该实验是系统边界偏差，未模拟真实分割误差分布。

### 3. 反向 pose 与减少视角的结论需分别解释

反向 pose 的 clean/65° IoU 及 canonical 总体误差与控制接近，未新增失败项；原位姿精度问题仍在。只检查一个反向初值，不能推断任意 pose 初始化都能恢复。

删除正面的 clean IoU 仍高，但 65° IoU 比控制下降 **0.72 个百分点**，最终 canonical 均值增大 **5.49%**，FDI12 均值相对自己的初始值增加 **8.61%**，触发逐牙门槛。此条件位姿误差比控制更小，不能将原位姿失败简单归因于删除正面，也不能据此认定两视角足够。图像量与 prior 相对权重同时发生了原流程所规定的变化。

### 4. 数值检查与恢复验收分开

七次运行的梯度最大相对误差均约 `7.37e-8–1.85e-7`，全部最终牙网格闭合、单组件。控制/seed202/seed303/反向 pose/腐蚀/膨胀/删正面的局部求根重试分别为 **0/3/2/0/2/3/0**，全部在原重试规则内完成。

实际完成轮数依次 **21/19/18/24/9/12/14**，保留的最佳轮次依次 **13/11/10/18/1/4/6**。原观测轮廓损失加 prior 的选择准则决定最终结果；腐蚀保留第 1 轮，随后 8 轮未改善该分数。没有利用干净图、GT 或留出指标重新挑选轮次。

## 可视化与证据

![全部条件的独立指标](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/metrics.png)

![全部四视角预测对干净观测](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/clean_observation_review.png)

白：同 FDI；黄：不同 FDI；红：参考独有；蓝：预测独有。腐蚀与膨胀出现的系统边界残差与数值一致。

所有七组 3D 图均已核查并保存在各自 `fit_attempt_01/geometry_review.png`，同目录另有 `semantic_review.png`（拟合观测与留出）。3D 图为每牙单向 1500 面积采样点；初始/最终在每列共享初始 P95 色标，列间及不同条件之间的色标不同，验收采用独立双向 6000 点指标。

代表性的 [seed303 3D](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/engine/runs/jobs/init_seed303/fit_attempt_01/geometry_review.png)、[腐蚀 3D](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/engine/runs/jobs/mask_erode1/fit_attempt_01/geometry_review.png)、[删除正面 3D](../runs/joint_robustness_four_20261004T061410Z_de71c9c1/engine/runs/jobs/drop_front/fit_attempt_01/geometry_review.png)。

最终 **78/78 完整性项通过**，包括冻结引擎、原控制及 warmup 历史输出、DMM 源码、模型、单因素输入、truth 复制件、门槛和独立 warmup 对应。旧 evaluator 的“两牙”文字保留，数值数组及本报告均按四牙解释。

## 本轮完成边界与后续

本轮完成了会话第 4 项的小规模单因素诊断。它是一例训练病例、四牙、已知相机的合成自洽检查，不支持整牙弓、未知相机、真实照片或未见病例的稳健性结论；与源文档 M4 prior 校准/视角扩展分开。

当前证据支持继续定位边界观测可靠性与逐牙形状可辨识性。局部遮挡需要先将显式有效区域传入轮廓对应、损失和最佳选择，避免把不可见牙区域当作背景约束，再单独运行遮挡诊断。本轮未扩展遮挡或调参。

## 复现

新运行会使用新目录；准备和输入视觉核查后再执行。运行中的子进程有 PID/分阶段日志；恢复时若记录子进程仍在运行，控制器拒绝重复启动。失败条件保留，不自动覆盖或重新拟合。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\run_joint_robustness_four.py --prepare-only
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_robustness_four_inputs.py runs\<new-batch>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_joint_robustness_four.py --resume runs\<new-batch>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\review_joint_robustness_four.py runs\<new-batch>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_robustness_four_geometry.py runs\<new-batch> init_seed303
```

计划：[joint_robustness_four.json](../configs/joint_robustness_four.json)。原 DMM、checkpoint、历史运行均保持冻结，新增控制器和复核工具与历史求解器分开保存。
