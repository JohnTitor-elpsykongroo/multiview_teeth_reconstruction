# 双颌数据加载与场景组装接口

2026-10-04。实现位于项目内 `third_party/DMM`，直接修改 DMM 源码及训练入口。上颌、下颌分别训练，场景加载时才组合两套模型。参数含义遵循 [双颌约定 v1](dual_arch_parameter_data_contract_v1.md)。

当前完成数据加载、单颌训练、模型包导出、固定相机场景加载、两颌位姿与 latent 参数、混合表面导出及跨颌深度合成接口。后续已接入可微渲染、隐式表面梯度桥及分阶段图像拟合；正式真实数据训练尚未启动。独立 Marching cubes 导出仍不能用于图像到 latent 的反向传播。工作站部署见 [WSL2 / RTX 5090 训练手册](WSL_RTX5090_environment_and_training.md)。

## 1 源码入口

可选的双颌训练调度、跨颌患者划分检查、中断续跑和联合拟合防穿插配置见 [可选扩展说明](dual_arch_optional_extension_v1.md)。独立单颌入口无需启用此扩展。

| 位置 | 职责 |
| --- | --- |
| `third_party/DMM/train_dmm.py --config ...` | 新 manifest 训练入口；每次只训练指定的一颌 |
| `third_party/DMM/data/arch_dataset.py` | ArchDataset、逐组件采样、患者划分检查 |
| `third_party/DMM/networks/dmm_net.py` | 原 DMM 类新增 arch 校验、component、query；沿用原网络及权重键 |
| `third_party/DMM/utils/math.py` | 原 screw_axis_to_rt 修复零旋转，新增 transform_screw |
| `third_party/DMM/training/arch_training.py` | 存在性驱动的损失、独立 embedding、训练存档和 prior 导出 |
| `third_party/DMM/dmm/bundle.py` | 严格加载模型权重、协方差与来源绑定 |
| `third_party/DMM/dmm/scene.py` | load_scene、DualArchScene、ArchState、compose_depth_layers |
| `third_party/DMM/dmm_cli.py` | 配置生成、数据检查、模型导出、场景检查命令 |

旧 `train_dmm.py -e ...` 仍使用原始数据协议，未获得本次 presence 与 manifest 训练逻辑。新训练通过 `--config` 明确选择。本次未改外部 `D:\WorkSpace\Dental\DMM`、已训练模型或历史 sanity check。

## 2 独立单颌训练数据

每颌一个 manifest，不要求配对病例或同时提供另一颌。所有文件引用使用相对**所在 JSON** 的路径与 SHA-256：

```json
{"path": "samples/case001.npz", "sha256": "完整的64位SHA256"}
```

顶层 manifest 为以下结构；这里的 `cases` 必须填入真实病例行，不能留空：

```json
{
  "contract_id": "dual_arch_static_semantic",
  "version": "1.0.0",
  "artifact_kind": "arch_training_manifest",
  "representation_profile": "coupled_component_dmm",
  "arch": "upper",
  "cases": []
}
```

每行字段沿用约定第 8 节。具体要求如下：

- `case_id` 在该颌清单中唯一，按字典序排列；`embedding_row` 从 0 连续编号，包括 val/test 占位行。
- `patient_id`、`arch`、`split` 必填；split 为 train/val/test。训练只更新 train 病例，两颌分别维护 embedding 表。
- `source_geometry`、`source_annotation`、`samples`、`centers` 都是上述文件引用，并核验 hash。
- `source_unit` 接受 mm/cm/m/um，`source_unit_to_mm` 分别为 1/10/1000/0.001；`unit_evidence` 必须记录来源依据。加载器不会替调用方推断单位或执行对齐。
- `T_arch_mm_from_source_mm` 是刚体矩阵；samples 和 centers 必须已经完成单位转换、刚体 canonical 对齐和除以 50。`model_unit_mm` 必须为 50。
- `canonical_reference_id` 同一清单内一致。`sampling_domain_model` 是 `[min_xyz,max_xyz]`，同一清单使用一个共同采样域。
- `presence` 包含该颌完整 14 个字符串 FDI 键，值只能为 Boolean。上颌 11–17/21–27；下颌 31–37/41–47。不可由随机采样点或图像推断。
- `surface_definition` 为非空表面范围说明，例如扫描牙冠与牙龈、封口处理；不得假定扫描中存在牙根。
- 原始病例的 `augmentation_parent_id`、`augmentation_transform` 均为 null。
- 首版增强支持 `kind=mirror_x`，`matrix_model=diag(-1,1,1,1)`，`label_map` 明确交换左右 FDI，如上颌 11↔21。父病例必须在本清单内、为原始病例且同患者同 split。样本坐标、法线、标签应由预处理器完成相应变换；加载器核验声明和存在性，不自动生成镜像。

NPZ 不使用 pickle，精确包含以下四个数组：

| 数组 | 形状与含义 |
| --- | --- |
| surface_points | N×3，canonical 模型坐标 |
| surface_normals | N×3，单位法线，与表面点逐行一致 |
| surface_labels | N 个整数，0 表示本颌牙龈，其他为本颌 FDI |
| offsurface_points | M×3，同一模型坐标中的非表面采样点 |

表面池必须覆盖牙龈及每个明确存在的牙位，不得含已声明缺失的牙位。未知 SDF 距离列不参与监督。centers JSON 为 `{"11":[x,y,z], ...}`，仅包含存在牙位，坐标同样为模型坐标；合法的零质心不会被跳过。

加载时逐个文件验证，训练按病例读取，不把整个数据集保存在内存中。每病例每个存在组件采样相同配额，池不足时有放回采样；缺牙不进入损失。随机采样由 seed、epoch 和病例索引决定，DataLoader 批量时可使用 `collate_cases` 保留变长组件列表。

### Canonical 参考文件

使用同一公共 header，`artifact_kind=canonical_reference`，并提供：

- `arch`、`reference_id`、`model_unit_mm:50`、`axes:"LPS"`。
- `centers_model`：本颌全部 14 个 FDI 中心。左右对应中心需关于 X=0 对称，14 个中心平均值为零。
- `training_case_ids`：用于建立参考的原始 train 病例 ID 列表，不得含增强、验证或测试病例。
- `direction_anchor_evidence`：解剖方向锚点与建立过程的来源说明。

格式和 hash 校验不能证明单位依据或解剖方向正确；这些必须在制作数据时核实。本次未将任何旧的逐例缩放数据自动转换成毫米规范。

## 3 命令行使用

以下在项目根目录执行。`data_v1` 为用户后续制作的数据路径示例，目前未生成真实训练数据。

```powershell
$dmmPython = 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe'

& $dmmPython -B third_party/DMM/dmm_cli.py make-config --arch upper --manifest data_v1/upper/manifest.json --canonical-reference data_v1/upper/canonical.json --specs third_party/DMM/examples/dual_arch/upper_specs.json --output configs/dual_arch/upper_train.json

& $dmmPython -B third_party/DMM/dmm_cli.py make-config --arch lower --manifest data_v1/lower/manifest.json --canonical-reference data_v1/lower/canonical.json --specs third_party/DMM/examples/dual_arch/lower_specs.json --output configs/dual_arch/lower_train.json

& $dmmPython -B third_party/DMM/dmm_cli.py validate-training --config configs/dual_arch/upper_train.json --other-config configs/dual_arch/lower_train.json

& $dmmPython -B third_party/DMM/train_dmm.py --config configs/dual_arch/upper_train.json --output runs/upper_v1_run001 --device cpu
& $dmmPython -B third_party/DMM/train_dmm.py --config configs/dual_arch/lower_train.json --output runs/lower_v1_run001 --device cpu
```

单独训练任一颌只需要自己的 config；`--other-config` 仅在两份清单均可用时检查跨颌患者 split，一份清单的校验不能证明另一份没有泄漏。每次输出目录必须不存在。GPU 环境可显式传 `--device cuda:0`；本文件最初的训练接口测试使用 CPU，后续渲染与拟合 GPU 验证见独立文档；目标工作站上的完整网络训练反传仍应按 WSL 手册实测。

`make-config` 自动填充文件引用 hash。网络示例继承官方结构，目前牙龈与牙齿默认均 10 维，可以分别修改 specs 的 latent_dim；修改后需重新生成引用 hash。维数由 specs/model bundle 驱动，代码未写死 10 或 20。

默认配置为 100 epochs、每组件 256 个表面点、2048 个 offsurface 点、Adam 学习率 1e-4。它们是起始配置，未在真实数据上调优。首版按病例依次优化，覆盖全部 train 病例，不提供多卡或验证集模型选择。损失保留 DMM 的表面、法线、Eikonal、非零排斥、修正、形变、模板法线、中心、分割和 latent 正则项；新路径按有效点/组件取均值，因此不保证与官方旧训练数值等价。

保存 `progress.jsonl`、配置和 provenance、每 10 轮 checkpoint、最低**训练损失**的 `best_train.pth`、末轮 `final.pth`。这不是最佳验证集模型。checkpoint 包含本颌网络、embedding、canonical buffer、优化器、随机状态、病例行号及来源信息。恢复到新的目录：

```powershell
# 把同一 config 的 epochs 调大后执行；数据、网络及其他训练参数须保持一致。
& $dmmPython -B third_party/DMM/train_dmm.py --config configs/dual_arch/lower_train.json --resume runs/lower_v1_run001/final.pth --output runs/lower_v1_run002 --device cpu
```

## 4 从独立训练结果导出模型

```powershell
& $dmmPython -B third_party/DMM/dmm_cli.py export-bundle --checkpoint runs/upper_v1_run001/final.pth --output scene_demo/model_bundle/upper --model-id upper_v1_run001
& $dmmPython -B third_party/DMM/dmm_cli.py export-bundle --checkpoint runs/lower_v1_run001/final.pth --output scene_demo/model_bundle/lower --model-id lower_v1_run001
```

每份导出包括 `model.json`、严格匹配原 DMM 网络键的 `weights.pth`、`latent_statistics.npz`。权重文件无训练病例 embedding 表。

统计文件对内部组件 0 及每个牙位保存 `mu_<id>`、`cov_<id>`、`L_<id>`、标量整数 `count_<id>`。只统计原始 train 病例中存在的组件，镜像不加入；按约定进行 0.05 对角收缩与特征值下限处理。各组件至少 d+1 例，方差不足则拒绝导出，不能用缺牙占位或随机码补齐。

模型 JSON 绑定权重、统计、训练清单、split、canonical ID、尺度、采样域、训练 epoch 与源码指纹。导出和加载要求使用记录的源码版本；更改相关源码后需明确处理版本兼容，不能静默用旧 bundle。原始上颌实验 checkpoint 不自动迁移为本版本。

## 5 场景加载与参数接口

目录和相机字段遵循约定第 5、9 节。JSON 的公共 header 使用相同 contract/version/profile，并分别设置：

| 文件 | artifact_kind | 额外结构 |
| --- | --- | --- |
| fit_input/manifest.json | fit_manifest | scene_id、length_unit、presence、camera_file、views、model_bundle、initialization |
| fit_input/cameras.json | cameras | views 为完整相机记录列表 |
| initialization/parameters.json | scene_initialization | method、origin、arches |
| model_bundle/upper或lower/model.json | arch_model_bundle | 由 export-bundle 生成 |

manifest 中 `presence`、`model_bundle` 均按 upper/lower 两键组织；`views` 是按 cameras 顺序的 view_id 字符串列表；camera_file、initialization 和两份 model_bundle 都是相对引用加 hash。masks 使用 FDI/0/255，valid 图只能为 0/1。语义张量有 29 通道：background 在前，随后固定 28 个牙位。

initialization 的 `method` 接受 `training_mean` 或 `independent_estimate`，`origin` 记录来源。`arches.upper/lower` 分别有 `T_world_from_arch`、`q`、`model_sha256`、`statistics_sha256`。q 仅包含存在牙位；`training_mean` 下全为零；gum 从统计均值固定读取。

```powershell
& $dmmPython -B third_party/DMM/dmm_cli.py validate-scene scene_demo/fit_input/manifest.json

# 可选导出双颌混合表面，仅供检查/评估；未经训练的网络可能没有合法等值面。
& $dmmPython -B third_party/DMM/dmm_cli.py validate-scene scene_demo/fit_input/manifest.json --mesh-output scene_demo/assembled.npz --resolution 64
```

Python 中先将 `third_party/DMM` 设为工作目录或加入 `sys.path`：

```python
from dmm.scene import load_scene, compose_depth_layers

scene = load_scene(".../scene_demo/fit_input/manifest.json", device="cpu")
parameters = [p for p in scene.parameters() if p.requires_grad]
upper_pose = scene.arches["upper"].T_world_from_arch
lower_pose = scene.arches["lower"].T_world_from_arch
relative_pose = scene.T_upper_from_lower
upper_codes = scene.arches["upper"].codes()
fields = scene.query_world(world_points_mm)  # 两个独立 SDF 与软语义结果
prior_loss = scene.latent_prior()
```

可优化参数恰为两套 6D pose_delta 和存在牙的白化 q。pose_delta 是左乘到初始位姿的 SE(3) 增量，前三项弧度、后三项 mm；decoder 冻结，gum 无自由参数。`query_world` 返回各颌的 model-unit `sdf`、`sdf_mm`、组件权重及语义，不跨颌混合 SDF。

`extract_mesh` 每颌分别提取混合等值面，在顶点上重新查询软语义，再施加 50 mm 尺度与整颌位姿。输出 vertices_world_mm、faces、semantics、face_arch（0 上颌、1 下颌）。无等值面、非有限值或非正边界会失败，避免静默截断。

`compose_depth_layers` 接受 A×H×W 的正相机 Z 深度（未命中为 +inf）和 A×H×W×29 概率，选择每像素最近表面，包含牙龈的遮挡。相同深度按输入顺序选第一层；无命中输出背景。该接口用硬深度选择，可传播被选中概率的梯度，未实现遮挡边界的可微处理。

场景加载器只接受 fit_input、model_bundle、initialization 允许目录中的哈希引用，不扫描根目录、不读取 truth。不同相机共用相同掩码可接受，完全重复观测拒绝；有效空视图保留为负证据。每颌至少两个不同相机观察到牙齿，未观察到的存在牙位在 summary 中列入 unobserved。

## 6 验证范围与后续工作

本次在 Python 3.12.14 / PyTorch 2.13.0+cpu 下通过 14/14 项测试，无失败、无跳过；来源指纹和范围记录见 [验证记录](dual_arch_interface_validation.json)。

在项目根目录运行：

```powershell
& $dmmPython -B -m unittest discover -s tests -v
```

测试以临时的小网络和人工点池运行，包含：上、下颌独立训练与导出；缺牙不更新 embedding；采样池与完整 presence 冲突拒绝；模型查询对齐原生 inference；latent 有限差分与零旋转一、二阶梯度；续训与连续训练逐张量一致；场景尺度/相对位姿/固定语义通道；双颌混合表面组装；牙龈遮挡；动态视角、相机/hash/mask 校验；truth 路径隔离和训练 CLI。

这些是接口与计算链路测试，不是训练模型质量、真实咬合关系或照片重建效果的验收。后续已接入 [SemanticXY 匹配](semanticxy_matching_v1.md)、[原生渲染与梯度接口](differentiable_rendering_v1.md)，以及 [CUDA 验证、fit-scene 分阶段优化和可见性恢复](staged_fitting_gpu_v1.md)。正式上下颌训练安排在其余所需源码、数据及模型约束核查完成后。旧 synthetic sanity check 保持参考状态。
