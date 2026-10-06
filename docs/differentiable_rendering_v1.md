# 双颌可微语义 / XY 渲染与隐式表面梯度桥

## 1 本步交付与状态

代码直接加入 `third_party/DMM/dmm`，通过原生 `DualArchScene` 与 `dmm_cli.py` 接入。
上下颌仍各有独立 DMM、latent 和静态整体位姿；相机已知。没有启动正式训练，也没有读取或重新验收历史 sanity check。

| 部分 | 实现与验证状态 |
| --- | --- |
| 每颌混合隐式场 → 可微表面 | 已实现；原生 DMM 确定性小网络有限差分通过 |
| 50 mm/model-unit、两颌位姿、共同深度缓冲 | CPU 参考路径通过 |
| 29 通道软语义、透视插值、material XY | CPU 前向、软语义梯度、几何梯度通过 |
| SemanticXY → latent / 整颌 pose 反传 | 冻结局部对应的检查通过 |
| nvdiffrast 抗锯齿与轮廓梯度 | GPU 前后向与局部数值检查已通过；细网格整图 AA 离散导数有限制 |
| 多视图分阶段拟合器、不可见区域恢复 | 已接入，见后续独立验证记录 |

本文第 6 节保留最初 CPU 步骤的验证记录。当时 `dmm` 环境缺少 GPU 构建依赖；后续已在项目内编译并验证 CUDA 后端，接入分阶段拟合与恢复。最新环境、SM 12.x 修复、验证范围及数值限制见 [GPU 与分阶段拟合说明](staged_fitting_gpu_v1.md)。

## 2 表面：提取拓扑，再连接隐式梯度

实现：[surface.py](../third_party/DMM/dmm/surface.py)。配置：[surface_static_v1.json](../configs/surface_static_v1.json)。

1. 对每颌的完整混合场 `Phi(x,q)` 单独采样，检查采样盒六面为正、域内有零面。
2. marching cubes 提供不参与反传的拓扑、锚点和三角形。固定采样域来自模型包，不按当前图像自动缩放。
3. 在锚点处求混合场梯度并单位化，建立固定法向 `n` 的局部坐标。
4. 沿 `x = anchor + t*n` 解 `Phi(x,q)=0`；检测残差、法向导数、偏移信赖域及域边界。
5. 在根处用隐式函数定理接入一阶梯度：

\[
\frac{\partial x}{\partial q}
=-n\frac{\partial\Phi/\partial q}{\nabla_x\Phi\cdot n}.
\]

DMM 混合场不必是梯度模长恰为 1 的距离场，因此分母不能省略。再从连接后的顶点查询软语义，同时保留顶点移动及混合权重直接依赖 latent 的导数。转换为毫米后应用对应颌位姿。

这是局部法向参数化的一阶梯度桥，不是对 marching cubes 拓扑变化求导，也不声称复现论文的全部 NIE 实现。不支持用它推断二阶几何导数。

默认每轴分辨率 48、根残差阈值 `1e-6` model units、最大法向偏移 `0.1` model units。后者为局部数值信赖域，不是临床误差容限。无零面、边界截断、退化梯度、不收敛和越界均报错，不改用任意网格继续优化。

正常每次新的外层迭代应重新提取 charts；一次局部反传固定 charts。只有明确进行局部线搜索或梯度检查时才复用 charts；信赖域失败必须缩小更新或重建。缺失牙位不进入场混合，牙龈固定码保留。

## 3 渲染约定

实现：[rendering.py](../third_party/DMM/dmm/rendering.py)。

- 两颌完整网格拼接后统一做深度测试；不将两颌 SDF 混合，也不按牙号分别渲染后相加。
- 牙龈仍是几何遮挡体，在语义上贡献背景通道。
- 通道固定为 `[背景, 11..17, 21..27, 31..37, 41..47]`。使用顶点混合概率，保持软语义。
- 世界与相机位置单位为 mm；相机 `x` 向右、`y` 向下、`z` 向前；整数像素中心；背景深度为 `+inf`。
- `RenderOutput` 包含 `semantics, xy_pixels, depth_mm, coverage, face_index, metadata`，可直接传给 SemanticXYMatcher。
- 所有输入观测都渲染；不固定视角数量、不丢弃有效空视图。

### CPU 参考后端

[render_reference_v1.json](../configs/render_reference_v1.json) 显式选择 `torch_reference`。

分块投影、透视正确插值、硬最近深度选择。它支持内部软语义和 material XY 的几何梯度；**不支持轮廓覆盖率与遮挡边界导数**，不能作为最终轮廓拟合后端的等价替代。

对穿越 near/far 平面的三角形明确拒绝；完整在范围外的三角形剔除。默认 near 为 0.1 mm，far 为 10000 mm。有限差分可以传入冻结的 `RasterContext`；必须是相同相机、图像尺寸与网格拓扑，只用于局部检查。

### CUDA 后端

[render_nvdiffrast_v1.json](../configs/render_nvdiffrast_v1.json) 显式选择 `nvdiffrast`，要求 CUDA float32。实现使用 rasterize → interpolate → antialias，联合抗锯齿语义、预乘 XY 及覆盖率，再对 XY 解预乘。语义的负舍入误差截为零并恢复单位和。

相机投影转成 OpenGL clip 坐标时包含整数中心的 `+0.5`，在输出端将原生底部向上的图像翻转为顶部向下。深度与 face ID 保留硬命中定义；抗锯齿边缘可能有非零 coverage 而硬深度为 `+inf`，不能用硬深度的有限性替代 coverage。

后端缺失时明确报错，不自动降级。GPU 前向方向、XY、覆盖率反传及大三角形 AA 有限差分已通过，细密网格的重新光栅化目标不保证严格有限差分一致，详见后续验证。当前 CUDA 配置采用 4×4 超采样；输出深度与面编号取最近子样本。API 与坐标约定依据 [nvdiffrast 官方文档](https://nvlabs.github.io/nvdiffrast/)。

## 4 为什么 XY 需要冻结材质点插值

固定像素处，若普通插值权重与几何投影同时参与反传，插值出的 XY 在前向始终是该像素坐标，几何导数可能相互抵消。

本实现每次前向先计算透视材质权重；在 XY 分支固定权重，用它插值**仍可微的相机空间顶点**，再通过 K 投影。前向得到对应像素坐标，反向跟踪该材质点的屏幕移动。普通语义插值仍保留其几何导数，GPU 抗锯齿步骤也保留几何导数。

因此这里的 XY 梯度是一次局部对应更新下的几何导数。有限差分必须冻结法向 chart、XY 材质权重、面选择、采样索引与 OT 计划；每次重新光栅化或重新求 OT 后得到的标量，不是同一个检查目标。

XY 归一化、语义权重、面积质量、partial OT、未匹配区域沿用 [SemanticXY 协议](semanticxy_matching_v1.md)。整个预测不见时保留损失和 `needs_visibility_recovery`，本步没有实现初始化搜索或恢复策略。

## 5 原生 API 与命令

```python
from dmm.surface import SurfaceConfig
from dmm.rendering import RenderConfig
from dmm.semanticxy import SemanticXYMatcher

rendered, mesh, charts = scene.render_views(
    SurfaceConfig.load("configs/surface_static_v1.json"),
    RenderConfig.load("configs/render_nvdiffrast_v1.json"),
)
result = SemanticXYMatcher().match_scene(rendered, scene.observations)
result["loss"].backward()  # decoder 固定；梯度到 q 和各颌整体 pose
```

在具有相符源码指纹的 v1 模型包和场景输入就绪后：

```powershell
& $dmmPython -B third_party/DMM/dmm_cli.py render-scene <scene.json> `
  --device cuda --surface-config configs/surface_static_v1.json `
  --render-config configs/render_nvdiffrast_v1.json `
  --matching-config configs/semanticxy_static_v1.json --backward-check `
  --output runs/<new-render-directory>
```

`render-scene` 不包含优化器。导出每视图 NPZ、双颌网格 NPZ、预览图、完整配置、来源指纹、输出哈希、匹配诊断与梯度范数。输出目录必须是新的；状态为 `RENDER_DIAGNOSTIC_NOT_RECONSTRUCTION`。CPU 诊断须显式改用 `--device cpu` 和 reference 配置。

旧模型包绑定其导出时的源码指纹；新增源码不自动兼容旧指纹。本步没有修改旧模型包的元数据绕过校验。正式训练仍在完整源码扩展完成之后。

## 6 本次验证证据

复现（无训练）：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -B scripts/check_render_bridge.py
```

使用原生 DMM 的 reference MLP、DeformNet、HyperNetwork、混合权重及 ArchState 构造确定性**未训练八面体场**。它只是检验公式与接口，不是牙齿模型质量样本。没有读取外部 DMM 权重。

本轮共 34 项：**33 通过、1 跳过、0 失败**（渲染/梯度 13 通过，GPU 1 跳过；SemanticXY 回归 20 通过）。包括两颌共享遮挡、牙龈遮挡、透视深度、像素中心、场景 API、命令导出、软语义 latent 导数和空预测失败语义。

冻结局部对应的原生 DMM 图像 Warp 损失导数：

| 参数 | 自动微分 | 中心有限差分 | 绝对误差 |
| --- | ---: | ---: | ---: |
| 上颌 FDI11 q[0] | 0.001525603365169 | 0.001525603365183 | 1.36e-14 |
| 上颌世界 x 平移 / mm | -0.000478452250083 | -0.000478452250076 | 7.57e-15 |

此精度来自 double 的小型局部解析夹具，不是对 float32 真实牙模、拓扑变化或 GPU 边界导数的精度保证。

[完整验证记录](../runs/render_bridge_20261004T125832331919Z/validation.json)、[运行报告与来源指纹](../runs/render_bridge_20261004T125832331919Z/report.json)、[测试日志](../runs/render_bridge_20261004T125832331919Z/tests.log)、[已检查的可视化](../runs/render_bridge_20261004T125832331919Z/preview.png)。预览可见两个视图的两颌代理表面、正确前后遮挡和连续 XY 颜色方向；它不展示解剖牙齿。

后续已完成 CUDA 运行与局部导数验证，并接入分阶段 pose/latent 优化、chart/对应刷新、不可见区域恢复和收敛报告，见 [独立记录](staged_fitting_gpu_v1.md)。上述最初 CPU 记录不改写。正式训练仍未启动。
