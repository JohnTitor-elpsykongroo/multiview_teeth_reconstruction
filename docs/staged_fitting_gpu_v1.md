# GPU 验证、分阶段优化与不可见区域恢复

## 1 当前交付

本步直接扩展项目中的 DMM 源码与命令入口，未启动正式 DMM 训练。

- [fitting.py](../third_party/DMM/dmm/fitting.py)：pose → shape → 交替 joint、回溯、检查点、状态与图像门槛。
- [recovery.py](../third_party/DMM/dmm/recovery.py)：逐 FDI 缺失检测、已知相机射线初始化、组件投影与遮挡深度辅助约束。
- [gpu_validation.py](../third_party/DMM/dmm/gpu_validation.py)：拟合启动前实际运行 GPU 抗锯齿梯度检查，失败则拒绝拟合。
- [完整拟合配置](../configs/fit_static_v1.json)、[CUDA 渲染配置](../configs/render_nvdiffrast_v1.json)：所有默认值显式保存。

目前验证范围是确定性**未训练 DMM 八面体夹具**的源码、计算链路与实际图像拟合。没有读取历史 sanity check 或外部已训练 DMM，也不代表真实牙模、照片或几何恢复验收。

## 2 GPU 环境与本次修复

已在 RTX 5050 Laptop GPU / SM 12.0 上编译并运行 nvdiffrast：

| 项目 | 实际使用 |
| --- | --- |
| Python / PyTorch | 现有 `teeth-yolo` 环境，Python 3.11 / PyTorch 2.14.0+cu130 |
| CUDA 编译器 | 项目 `.runtime/cuda-13.2.1`，NVCC 13.2.78 |
| C++ 编译器 | MSVC 14.51.36231 |
| 数学库头文件 | 读取现有 `3dteethland/Library/include`，不修改该环境 |
| Python 增补依赖、扩展二进制 | 项目 `.runtime/gpu-python` |
| nvdiffrast 来源 | 官方源码 `253ac4fcea7de5f396371124af597e6cc957bfae` 加下述兼容补丁 |

编译器 13.2 与 PyTorch 的 CUDA 13.0 属于同一主版本，但存在次版本差异，构建日志保留了 PyTorch 的提示。是否可用以本机实际前后向与数值检查为依据，不推广到其他环境。

### 已修复的两个问题

1. **设备选择**：原 DataParallel 在单 GPU 主机上会提前将子网络移到 cuda:0，导致显式 CPU 状态与 GPU 网络混用。原生独立颌路径改为保留 `.module` 权重名称的单设备容器，由 `.to(device)` 决定设备；旧 `arch=None` 路径保留原行为。
2. **SM 12.x 梯度累加**：官方 `common.h` 的共享内存合并原子累加没有显式 warp 同步。本机直接运行时，大三角形面积导数随通道数变化且严重偏离有限差分。对 SM ≥ 1200 启用源码已有的逐项 `atomicAdd` 分支后，同一测试导数稳定到约 5.71845，有限差分约 5.71442。补丁只改依赖中的这个编译条件，可能牺牲部分性能；没有放宽数值门槛。

环境准备没有修改系统 CUDA 或现有 Conda 包。新增编译组件来自 [NVIDIA 官方 13.2.1 组件清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.1.json)，下载后逐个检查 SHA256。

[编译来源、补丁和二进制哈希](../runs/gpu_render_20261004T151656324373Z/backend_provenance.json) 已归档。构建入口是 `scripts/prepare_cuda_redist.py`、`scripts/build_gpu_backend.ps1`；运行环境入口是 `scripts/gpu_env.ps1`。构建脚本还处理了本机 Python 启动器重设 PATH、无控制台 OEM 解码失败的问题，修改仅在构建进程内生效。

### GPU 验证边界

GPU 已通过前向方向、像素中心、共同深度缓冲、牙龈遮挡、抗锯齿反传、DMM latent/位姿到图像的局部有限差分，以及完整 SemanticXY 反传。

| 检查 | 自动微分 | 有限差分 | 绝对误差 |
| --- | ---: | ---: | ---: |
| DMM q11 → 内部材质点 XY | -3.088887 | -3.088951 | 6.41e-5 |
| 上颌 x 平移 → 内部材质点 XY | 0.874329 | 0.874519 | 1.91e-4 |
| 大三角形顶点位移 → AA 面积 | 5.718449 | 5.714417 | 0.00403 |

**细密 marching-cubes 网格的全图 AA 标量，在微小参数扰动下重新光栅化的严格有限差分仍未通过。** 此失败没有计入通过项，也没有被上述局部测试替代。轮廓候选、像素覆盖等离散选择会变化；官方同样说明小三角形会导致不完整抗锯齿和较噪的梯度，并建议提高渲染分辨率。[nvdiffrast 文档](https://nvlabs.github.io/nvdiffrast/)

因此原生拟合配置使用 **4×4 超采样**，对语义、预乘 XY 与 coverage 联合平均，再解预乘；深度与面编号选每个输出像素中最近的子样本。`max_pixels` 限制的是超采样后的像素数。后端构造器仍允许显式 `supersample=1` 用于基础检查。

优化器使用回溯和实际目标下降判断；shape 更新还必须通过重新提取网格后的全渲染检查。这里不承诺细网格轮廓目标处处光滑，不能将 CPU double 局部有限差分精度推广到 GPU 整图目标。

[GPU 检查记录](../runs/gpu_render_20261004T151656324373Z/validation.json)、[未通过检查的保留日志](../runs/gpu_render_20261004T151656324373Z/diagnostics/aa-probe-sm120.log)。

## 3 分阶段优化

后续已增加默认关闭的双向防穿插代理及相对位姿输出，使用方式见 [双颌可选扩展](dual_arch_optional_extension_v1.md)。本文件下方的历史数值与运行目录仍对应未启用该扩展的原版本。

相机、毫米尺度、presence、牙龈固定码与两套 decoder 始终固定。存在但从未在任何有效观测中出现的牙位，其 q 在拟合期间冻结；报告列出这些牙位，不声称从背景恢复了它们。

| 阶段 | 可更新参数 | 默认预算 |
| --- | --- | ---: |
| pose | 上下颌各一个跨视角共享 SE(3) 增量 | 40 |
| shape | 观测过的存在牙位 q | 40 |
| joint | pose 与 shape 按迭代交替更新 | 80 |

这些是最大预算，不是收敛保证。joint 不增加逐牙刚体变量，也不混合两颌隐式场。

目标为：

\[
L=L_{SemanticXY}+0.001L_{prior}+0.2L_{recovery}.
\]

SemanticXY 使用所有观测等权平均，包含有效空背景视图；语义权重、partial OT、未匹配和 XY 归一化沿用已有协议。prior 是既有白化 q 的均方项，不将白化解释成高斯分布证明。

每步刷新完整表面、光栅结果、抽样和 OT 计划；一次反传内固定离散对应。按旋转、平移和 q 分组构造有界下降方向，默认尺度分别为 0.01 rad、1 mm、0.1，最多进行 8 次减半回溯。

候选更新必须使重新匹配后的总目标下降；shape 候选还要重新 marching cubes 验证，防止只改善旧拓扑上的目标。越界、根求解失败、OT 失败等会拒绝该候选并记录原因；全部拒绝则恢复原参数。没有用优化器动量掩盖拒绝步骤。

梯度范数小或已接受的改善量持续很小，才计入稳定次数；有显著梯度但线搜索失败是停滞，不记作收敛。默认还要求所有存在目标/预测区域的逐牙、逐视图 IoU 最小值 ≥ 0.85，且没有待恢复区域，才返回 `IMAGE_FIT_CONVERGED`。这一状态仍仅表示图像拟合，不表示 3D 唯一解或临床验收。

## 4 不可见区域恢复

### 检测按牙位进行

对观测中每个可见 FDI，检查预测的软语义面积与主导牙号像素数。如果软面积不足目标的 5%，或没有至少 1 个主导像素，则列入恢复清单。这样不会因为同一张图中另一颗牙仍可见，就漏掉完全被遮挡的牙。

目标背景、ignore、显式不存在的牙，不产生恢复目标。预测为空时原 SemanticXY 损失继续保留。

### 整颌完全丢失时的初始化

当某颌全部观测区域均丢失，用目标牙号掩码质心的已知相机射线，与当前 DMM 组件表面质心建立平移最小二乘问题。

- 只提议整体平移，保持旋转、尺度、latent 与另一颌固定。
- 射线秩不足、条件数过大、位移超过 200 mm 时明确拒绝。
- 掩码质心受遮挡影响，此估计仅供粗初始化，不等同于真实牙齿中心或深度测量。
- 候选必须改善包含全部物理遮挡的完整渲染目标，才能接受。

### 局部恢复辅助约束

使用原生 `decoder.component()` 的零面提供当前牙位的三维辅助点。该组件表面**只用于辅助对应**；正式渲染仍使用完整混合牙弓，牙龈与另一颌继续参与遮挡。

对目标牙位的有效像素采样，寻找投影接近的组件点，并加入两项约束：

1. 对角线归一化的二维位置距离；
2. 已知相机前向深度约束，以及相对当前目标像素遮挡面深度的单边约束，默认间距 0.5 mm、归一化尺度 50 mm。

后者只使用当前渲染深度，不读取真实深度。源点最多 128、目标像素最多 64，确定性抽样；在一次反传中固定最近对应。辅助投影可以为出画面或被遮挡的组件提供梯度，但不能保证所有错误初始化都可恢复。

每次更新后重新检查真实渲染可见性；满足可见性条件后撤去相应辅助项。组件无有效零面或其梯度退化时不伪造辅助点，保留错误诊断。

## 5 使用方式与恢复运行

```powershell
. scripts/gpu_env.ps1
& $gpuPython -B third_party/DMM/dmm_cli.py fit-scene <scene.json> `
  --device cuda --fit-config configs/fit_static_v1.json `
  --surface-config configs/surface_static_v1.json `
  --render-config configs/render_nvdiffrast_v1.json `
  --matching-config configs/semanticxy_static_v1.json `
  --output runs/<new-fit-directory>
```

要求使用与当前源码指纹一致的 v1 模型包。正式训练仍未启动，因此此处没有拿历史上颌模型伪装成新双颌模型包。

输出包含 `settings.json`、逐步 `progress.jsonl`、原子替换的 `checkpoint.json`、初始化和最终预览/NPZ、`result.json`。每次运行使用新目录。

中断后增加 `--resume <old-run>/checkpoint.json`，同时指定新的输出目录。恢复会核对源码、输入观测、实际模型/统计量、全部配置及 GPU 二进制哈希，只接受 `RUNNING` 检查点。阶段位置、稳定计数和参数从最后一次完整检查点恢复；不能通过修改输入或阈值继续同一实验。

结果状态区分：

- `IMAGE_FIT_CONVERGED`：满足数值稳定、图像门槛、可见性要求。
- `VISIBILITY_RECOVERY_UNRESOLVED`：仍有观测牙位未恢复。
- `IMAGE_GATE_NOT_MET`：未满足图像门槛。
- `BUDGET_EXHAUSTED_OR_STALLED`：图像可能较好，但没有满足收敛条件。
- `FIT_FAILED`：异常或非法状态，记录到 `failure.json`。

CPU reference 仅可通过显式 `diagnostic_reference=true` 运行诊断，结果状态带 `REFERENCE_DIAGNOSTIC_` 前缀，不能自动替代 CUDA 正式拟合路径。

## 6 本次验证结果

真实 CUDA 环境运行 **43 项测试，43 通过、0 跳过**：渲染 14、SemanticXY 20、拟合/恢复 9。覆盖参数分阶段冻结、全视图处理、实际遮挡深度恢复梯度、背景不触发恢复、秩不足拒绝、GPU 拟合、缺失恢复、关闭恢复的失败对照，以及中断续跑与连续执行一致性。

固定短预算夹具在运行前登记，全部保留：

| 夹具 | 初始最差 IoU | 最终最差 IoU | 未恢复区域 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 整体平移 2 mm，三阶段 | 0.7651 | 0.9527 | 0 | 预算耗尽或停滞 |
| 整体平移 100 mm，出画面 | 0 | 0.9338 | 0 | 预算耗尽或停滞 |
| 下颌被上颌完全遮挡 | 0 | 0.9669 | 0 | 预算耗尽或停滞 |
| 同样出画面，关闭恢复 | 0 | 0 | 4 | 可见性恢复未解决 |

这些运行验证了接口和恢复机制，没有追加迭代使其达到收敛标签。最差 IoU 是所有有效牙位/视图组合的最小值，不是平均分；高 IoU 也不替代 3D 形状与位姿验收。

[固定案例与配置](../runs/staged_fit_20261004T151423673499Z/cases.json)、[完整汇总](../runs/staged_fit_20261004T151423673499Z/validation.json)、[测试日志](../runs/staged_fit_20261004T151423673499Z/tests.log)、[遮挡初始图](../runs/staged_fit_20261004T151423673499Z/occluded_lower/initial_render/preview.png)、[恢复后图](../runs/staged_fit_20261004T151423673499Z/occluded_lower/final_render/preview.png)。已检查可视化中的牙号、深度、XY 方向及下颌重新出现。

复现：

```powershell
. scripts/gpu_env.ps1
& $gpuPython -B scripts/check_gpu_render.py
& $gpuPython -B scripts/check_staged_fitting.py
```

正式训练仍应等其余已约定的源码、数据及模型约束核查完成后再安排。本步没有改变这一顺序。
