# DMM 多视图重建：前向核对、Pose-only、Shape-only 与 Joint

**Git 训练协作入口：[分轮训练、目标机克隆与反馈计划](docs/TRAINING_ITERATION_PLAN.md)。** 项目根目录现作为独立仓库，DMM 和 nvdiffrast 修改源码直接纳入版本管理。数据、权重和环境另行迁移。当前只执行 R0 预检与 R1 短程 pilot，反馈后再修改、更新和继续训练。仓库内 [主设计文档](docs/multiview_teeth_reconstruction_semanticxy.md) 随代码同步。

远程仓库：[JohnTitor-elpsykongroo/multiview_teeth_reconstruction](https://github.com/JohnTitor-elpsykongroo/multiview_teeth_reconstruction)。新机器部署使用 `training-r0-v2` 标签（相对 v1 仅补齐部署文档，训练源码和配置相同）。

RTX 5090 / WSL2 训练交接：[交接入口与修复证据](docs/TRAINING_HANDOFF_20261006.md)。新配置在 `configs/training_handoff_v1/`，使用非零参考初始化与稳定法线分母；脚本提供包校验、目标 GPU 预检、smoke、pilot、显式长训入口。正式几何质量尚待 pilot/训练验证。

**当前主线（2026-10-06）：直接使用Blender现成逐牙/组织标签，暂不开发或训练识别分割。** [oracle观测打包与阶段三接入](docs/stage2_oracle_workflow.md)已在8场景80张图上运行；后续识别模型采用本地SegmentAnyTooth。

2026-10-06 源码整合：[正式训练前的实现与验证说明](docs/training_readiness_v1.md)。新增显式训练配方、冻结 decoder 的验证码拟合与 best_val、训练子集统计约束、decoder 兼容绑定、双像素约定及均值模型候选初始化。配置位于 `configs/training_readiness/`；正式训练与真实重建质量验收尚未完成。

照片流程新增：[阶段二 RGB → 逐牙观测初步实现](docs/stage2_image_understanding_v1.md)；[阶段二 → 阶段三观测契约](docs/tooth_observations_v1.md)；[Blender pilot验收与RGB遮挡分支改进](docs/stage2_blender_acceptance_20261005.md)。两组pilot已用于输入验收和数值联调，25项测试通过；尚无正式分割训练或识别精度结论。

工作站部署与训练：[ThinkStation / RTX 5090 / 128GB 的 WSL2 环境配置与训练手册](docs/WSL_RTX5090_environment_and_training.md)，包含源码迁移、数据校验、GPU 预检、独立双颌训练、续跑、模型导出和可选联合拟合。

## 当前双颌开发约定

当前目标为**已知相机、静态上下颌、逐牙语义掩码驱动的合成重建**。参数与数据含义以 [双颌参数与数据约定 v1](docs/dual_arch_parameter_data_contract_v1.md) 和 [机器可读固定选项](configs/contracts/dual_arch_static_semantic_v1.json) 为准。已在 `third_party/DMM` 源码中实现单颌数据加载、独立训练入口、模型包导出、双颌场景组装与分阶段图像拟合；使用方式与数据格式见 [双颌接口使用说明](docs/dual_arch_interfaces.md)。真实数据训练尚未启动，历史运行器仍使用原协议。

V1 使用两套单颌 DMM、两个跨视角共享的整体刚体位姿、固定毫米尺度与显式牙位存在性；现有耦合 latent 不额外叠加逐牙位姿。形状与排列解耦另立后续模型版本。已由用户检查认可的 synthetic sanity check 作为参考保留。

SemanticXY 匹配、双颌软语义 / XY 渲染和隐式表面梯度桥已接入原生源码：[匹配协议](docs/semanticxy_matching_v1.md)、[渲染约定](docs/differentiable_rendering_v1.md)。CUDA 前后向与局部数值检查已完成，并修复 SM 12.x 依赖梯度累加问题；已新增 `fit-scene` 分阶段优化、不可见区域恢复和检查点续跑。使用方式、43 项通过的回归测试、细网格 AA 导数限制与全部短预算结果见 [GPU 与分阶段拟合说明](docs/staged_fitting_gpu_v1.md)。正式 DMM 训练仍未启动，安排在所需源码修改与扩展完成后。

可选扩展已提供 `make-dual-config` / `validate-dual-training` / `train-dual`，用于统一校验与调度两套独立训练；联合拟合可显式启用防穿插代理，默认关闭，不强制上下牙接触。详见 [上下颌训练与联合拟合可选扩展](docs/dual_arch_optional_extension_v1.md)。

以下为历史上颌实验的运行说明和原始记录，其路径、单位、协议与状态不自动升级为双颌 v1。

第 0 步读取冻结的 DMM 权重、同 epoch 的训练 latent、训练 split 和既有零码网格，在本项目 `runs/` 下新建一次性运行目录；不会写入 DMM 源码、训练数据或既有实验目录。Pose-only 步骤读取前向运行生成的逐牙掩码、已知相机和固定牙齿网格，联合优化上颌 14 颗牙的一个刚体位姿。

## 运行

在本目录执行：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_forward_check.py --config configs\forward_check.json
```

依赖：PyTorch、NumPy、SciPy、scikit-image、Pillow、plyfile、trimesh。运行器使用 CPU。进度实时写到 stderr 和 `runs/<run-id>/progress.log`，stdout 最后一行是 JSON 结果。每次运行都创建新目录；若同一秒内的同配置运行 ID 碰撞，会拒绝覆盖。

## 核对范围

- 严格加载 `20260824` 的 `best` 模型和同 epoch latent，检查 15 个组件的标签及维度。
- 根据训练 split 与 `SdfSamples` 的排序恢复病例行号；选定病例必须确实有全部 14 颗上颌牙。
- 用零码与病例训练码分别前向生成牙龈和 14 颗牙的网格。每个组件使用同一局部采样域，保存边界检查、表面 SDF 残差、连通性及封闭性。
- 根据前牙和后牙中心构造正面、左右 30° 的已知相机；保存 `K`、世界到相机的 `R,t` 和投影坐标约定。
- 通过深度缓冲输出逐牙 ID 图、语义彩图、深度图、牙齿单独渲染图及汇总图。
- 保存配置、输入 SHA-256、checkpoint 元数据、病例行号、相机检查及逐项数值报告。

`report.json` 的 `FORWARD_RENDERED_VISUAL_REVIEW_REQUIRED` 只表示前向计算完成。图像与网格须另行检查，不能据此推断照片拟合、shape 梯度或临床可用性。训练码生成的病例仅用于模型自洽性核对，不是未见病例的泛化验证。

对一次完成的运行增加放大图与咬合面检查视图：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\render_review_views.py runs\<run-id>
```

首次已核对运行的状态和限制见 [review.md](runs/forward_20260929T092217Z_b45f45ab/review.md)。牙龈小岛已单独标记，不能把牙龈总览掩码直接作为监督。

若要将结果作为下一阶段的输入，先运行：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\package_forward_observations.py runs\<run-id>
```

它创建 `fit_input/manifest.json`（只含逐牙掩码与已知相机）以及独立的 `truth/`。Pose-only 因为明确固定了 shape，还读取同次生成的 14 个牙齿网格作为 `z_GT` 对应的固定几何；其优化目标不读取 latent、深度或真值位姿。

## Pose-only：三视角固定 shape 恢复上颌位姿

`configs/pose_only.json` 指向已完成的前向运行，并设置扰动初始位姿。`initial_rotation_degrees_xyz` 是旋转向量三个分量的角度值，`initial_translation_dmm` 是 DMM 坐标单位。三台已知相机为正面及左右 30°。运行：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_pose_only.py --config configs\pose_only.json
```

拟合先用逐牙质心求粗位姿，再在每个视角、每颗牙的可见轮廓间建立双向最近邻对应，优化同一个 6DoF 位姿。每轮重新渲染逐牙掩码并计算 IoU。shape 网格、FDI 标签和相机全程固定。新运行目录保存 `provenance.json`、`progress.jsonl`、`fit_report.json`、`pose.json`，以及 initial、centroid、final 三阶段的逐视角叠图。`fit_report.json` 的完成状态仍需独立真值评估。

拟合结束后，另行运行：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\evaluate_pose_only.py runs\<pose-only-run-id>
```

评估器此时才读取前向运行的 `truth/manifest.json`，同时复核相机、掩码及固定网格的 SHA-256，生成 `evaluation.json`。首次实验结果为 [evaluation.json](runs/pose_only_20260929T100436Z_413ea93f/evaluation.json)：`POSE_ONLY_PASS`，旋转误差 0.0627°、平移误差 0.00101 DMM 单位，逐牙平均 IoU 从 0.1134 升到 0.9663。三视角 [初始叠图](runs/pose_only_20260929T100436Z_413ea93f/renders/initial/overlay_sheet.png) 与 [最终叠图](runs/pose_only_20260929T100436Z_413ea93f/renders/final/overlay_sheet.png) 可人工核查。叠图白色表示目标和预测均有牙齿，红色表示漏检，蓝色表示多检。

该结果只验证同一 DMM 生成、无噪声、逐牙 ID 已知、相机已知、shape 已知的合成问题；尚未验证从照片估计相机、提取掩码、优化 shape 或处理遮挡与域差异。平移误差单位是 DMM 坐标单位，不能直接解读为毫米。

### 固定 shape 与相机的多组随机位姿检查

`configs/pose_random.json` 定义四档初始扰动，每档 4 组。旋转角度为 5–10°、10–20°、25–40°、60–90°，平移范数分别为 0.02–0.05、0.05–0.12、0.15–0.25、0.35–0.50 DMM 单位。旋转轴和平移方向独立均匀分布在单位球面，角度及平移范数在各区间均匀采样；固定 seed20261004，并在拟合前保存全部 16 个初始位姿。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_pose_random.py --plan configs\pose_random.json
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_pose_random.py --resume runs\<pose-random-batch-id>
```

批次归档原 Pose-only 拟合器、渲染器和独立评价器，使用归档代码执行，保持原来的质心粗配准及 5 轮轮廓求解。最多 4 个 CPU 子进程同时拟合；`status.json` 保存每个试验的阶段、子进程 PID、结果目录和失败原因，每个 job 有完整日志。恢复时检查冻结输入和归档代码哈希，跳过已完成试验，保留中断尝试；控制器或子进程仍运行时拒绝重复启动。

验收保留旋转误差 ≤1°、平移误差 ≤0.02、最终逐牙平均 IoU ≥0.85，并要求 IoU 不下降。旧单次检查的 IoU 提升 ≥0.5 单独记录为诊断，因为小扰动的初始 IoU 可能已较高。拟合结束后才执行真值评价；所有成功和失败进入 `review/summary.json`、`review/report.md`，另保存指标散点图及各档最低最终 IoU 的叠图。重新汇总：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\summarize_pose_random.py runs\<pose-random-batch-id>
```

已完成批次 `pose_random_20261004T043020Z_89047cb6`：四档各 4/4，共 **16/16 通过**。实际初始旋转 5.12–78.60°、平移范数 0.0209–0.4511 DMM 单位；最终逐牙平均 IoU 0.966293–0.966351，最大旋转误差 0.062840°，最大平移误差 0.0010079 DMM 单位。16 组也全部满足旧的 IoU 提升 ≥0.5 条件，最小提升 0.523590。19 个固定输入文件哈希均一致。完整证据见 [逐组报告](runs/pose_random_20261004T043020Z_89047cb6/review/report.md)、[图像核查与结论](runs/pose_random_20261004T043020Z_89047cb6/review/assessment.md) 和 [完整叠图](runs/pose_random_20261004T043020Z_89047cb6/review/full_comparison.png)。完整画面与全部最终语义图可重新生成：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_pose_random_review.py runs\<pose-random-batch-id>
```

## Shape-only：固定相机与位姿，恢复少量牙齿的 latent

按文档 M2 的最小范围，先开放 11、21 两颗中切牙的全部 40 个 latent 维度。相机、上颌位姿、其他 12 颗牙保持固定；牙龈不参与。这里其他牙齿使用生成真值码作为已知遮挡几何，这是明确记录的合成诊断条件。

准备器负责读取生成真值，给活动牙代码添加固定随机种子的扰动，并将拟合入口与真值分开。当前扰动为每维训练码标准差的 0.25 倍；这些尺度只是优化变量的数值缩放，尚未建立统计 prior。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\prepare_shape_only.py --config configs\shape_only.json
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_shape_only.py --config configs\shape_only.json --fit-input runs\<shape-case-id>\fit_input
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\evaluate_shape_only.py runs\<shape-only-run-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_shape_review.py runs\<shape-only-run-id>
```

拟合器只读取打包的掩码、已知相机、固定真值位姿、初始码、训练尺度、采样盒和冻结 DMM 权重；活动牙的 `z_GT`、源病例网格、深度和预留视角均在独立评估阶段使用。运行目录记录输入和代码 SHA-256、迭代日志、初始/最终 codes、网格及渲染。

当前图像目标是同 FDI 的双向可见轮廓对应。每次局部优化中，对模型轮廓点沿固定法线求 DMM 零等值面位置，再投影至图像；通过隐式微分和 PyTorch 的 SDF 对 latent 导数，把轮廓残差回传到代码。这采用 [DeepMesh](https://arxiv.org/abs/2106.11795) 的隐式表面微分思路。外层更新网格、遮挡和对应关系。它是用于最小核对的连续轮廓目标；尚未实现论文完整的软语义 SemanticXY 渲染损失，也没有对硬栅格 IoU 求导。

有限差分检查在每颗活动牙上测试 4 个代码方向及 2 个步长，对比完整的“代码 → 隐式表面 → 图像轮廓 loss”导数；检查时固定当前的可见性及轮廓对应。另用重新解码和深度缓冲渲染的逐牙 IoU 检查实际改善。使用局部/全局代码步长约束及轻微初始码阻尼，未将它们解释为经过校准的统计 prior。

首次完成运行为 `shape_only_20260930T014548Z_49b983d8`，使用 `shape_case_20260930T014539Z_49b983d8/fit_input`。独立评价为 [SHAPE_ONLY_SUBSET_PASS](runs/shape_only_20260930T014548Z_49b983d8/evaluation.json)：图像到 latent 梯度最大相对误差 1.73e-6，活动牙拟合视角平均 IoU 0.9589 → 0.9907，预留 65° 咬合面视角 IoU 0.9489 → 0.9794，双向采样表面平均误差 0.002350 → 0.001267 DMM 单位（降低 46.1%）。该表面指标按每个方向 6000 点采样，并在最近 32 个候选三角形中计算距离，不是完整 Hausdorff 距离。两颗最终牙齿网格均封闭且单连通。

人工核查见 [review.md](runs/shape_only_20260930T014548Z_49b983d8/review.md) 与 [初始/最终放大叠图](runs/shape_only_20260930T014548Z_49b983d8/review_comparison.png)。虽然图像、预留视角和表面距离均改善，两个 latent 的数值误差增大，因此本状态只证明该有限牙位合成实验的梯度与几何恢复有效，不证明唯一 latent 恢复、全牙列 shape 恢复或照片域性能。

## 分阶段扩大 Shape-only

`configs/shape_expansion.json` 定义 2 牙重复初始化、2 牙更大扰动、4 颗切牙、6 颗前牙、10 颗前牙及前磨牙、14 颗上颌牙六个阶段。前四组各 3 个 seed，后两组各 2 个 seed。牙位扩展使用 0.25 倍训练标准差，单独的扰动扩展使用 0.5 倍。随机扰动按 `(seed, FDI)` 独立固定，使不同活动牙位组中同一颗牙的初始代码一致。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_shape_expansion.py --plan configs\shape_expansion.json
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\inspect_shape_expansion.py runs\<batch-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_shape_expansion.py --resume runs\<batch-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\summarize_shape_expansion.py runs\<batch-id>
```

运行器串行执行准备、拟合及独立评价。每个子实验记录配置、输入包、拟合尝试、日志与评价；`plan.json` 保存预设验收门槛和原始配置，`code_snapshot/` 归档执行代码。恢复时跳过已经完成的子实验，保留中断或失败的拟合目录；运行中修改核心代码会被哈希检查拒绝，需要新建批次。使用 `--stage-limit N` 可在阶段间进行审查后恢复。`status.json` 的 `current_job`、`phase` 和各 job 结果用于实时进度，汇总器保留所有失败，依赖阶段未全部通过时跳过其后续牙位组。

扩展验收在运行前固定：拟合平均 IoU ≥ 0.93，IoU 误差减少 ≥ 30%，预留视角 IoU 误差减少 ≥ 20%，平均 3D 表面误差减少 ≥ 20%；同时要求逐牙表面均值不恶化超过 5%、P95 不恶化超过 10%，以及梯度、冻结代码、网格完整性检查通过。仍保留最小绝对 IoU 提升 0.002。原最小两牙实验的原始阈值和结果没有被改写。

预实验发现原固定采样盒会截断合法的扰动表面，因此扩展批次向外增加 8 个网格单元并同步增加分辨率，保持原网格间距及原内部采样位置。边界穿越检查继续执行。局部求根失败时缩小代码步长重试，所有重试均记录。局部 Jacobian 的部分代码方向对轮廓很弱，扩展批次将初始代码阻尼从 0.01 增到 1.0；它是数值稳定项，不是已校准的统计 prior。

扩展实验完成独立评价后，可生成逐牙语义叠图和 3D 表面误差核查图：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_shape_review.py runs\<batch-id>\jobs\<job-id>\fit_attempt_01 --output semantic_review.png
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_shape_geometry_review.py runs\<batch-id>\jobs\<job-id>\fit_attempt_01
```

语义叠图中白色表示相同 FDI，黄色表示重叠区域的牙位标签不同，红/蓝表示目标独有/预测独有。3D 图为每颗牙 1500 个面积采样点到真值表面的单向距离，初始和最终共用初始 P95 色标；该图仅供空间误差检查，验收仍使用独立评价的双向 6000 点指标。两个工具都要求先完成评价并拒绝覆盖既有图像。

### 已完成的扩大实验

批次 `shape_expansion_20260930T020823Z_4695c457` 完成全部六阶段，16/16 子实验通过预设验收。下表为各阶段不同初始化的范围，sigma 表示每维训练代码标准差的倍数。

| 活动牙数 / latent 维度 | sigma | 通过 | 最终拟合 IoU | 平均表面误差降低 |
| --- | ---: | ---: | --- | --- |
| 2 / 40 | 0.25 | 3/3 | 0.9912–0.9923 | 81.4%–92.0% |
| 2 / 40 | 0.50 | 3/3 | 0.9910–0.9918 | 91.0%–94.7% |
| 4 / 80 | 0.25 | 3/3 | 0.9821–0.9844 | 72.8%–84.0% |
| 6 / 120 | 0.25 | 3/3 | 0.9790–0.9820 | 73.3%–83.6% |
| 10 / 200 | 0.25 | 2/2 | 0.9804–0.9819 | 77.1%–78.4% |
| 14 / 280 | 0.25 | 2/2 | 0.9750–0.9773 | 约 77.3% |

完整上颌阶段的 65° 预留视角 IoU 为 0.9805–0.9807，28 个牙位/seed 记录的平均表面误差全部下降，最小降幅 32.7%。所有活动牙最终网格封闭且单连通；全批次图像到 latent 导数检查最大相对误差 9.50e-6。检查保持局部可见性和对应关系固定。

原采样盒、弱阻尼的 [预实验报告](runs/shape_expansion_20260930T020219Z_852f797c/review/report.md) 记录了 1/3 通过和两个数值失败，原始失败没有删除。修复后使用同一组 seed 开始独立新批次。

完整证据见 [逐 seed 结果](runs/shape_expansion_20260930T020823Z_4695c457/review/report.md)、[评估解释与可视化](runs/shape_expansion_20260930T020823Z_4695c457/review/assessment.md) 和 [源码及模型哈希复核](runs/shape_expansion_20260930T020823Z_4695c457/review/provenance_verification.json)。

这验证了同一训练病例、无噪声逐牙标签、已知相机及位姿下的完整上颌 Shape-only 几何恢复。两牙验证过 0.5 sigma，其余牙位扩展只验证了 0.25 sigma。90 个牙位拟合记录中 50 个的 latent 数值误差增大，不能将通过解释为唯一 `z_GT` 恢复。跨病例、真实照片、未知相机及 Pose+Shape 联合拟合仍需独立实验。

## 最小 Joint：两牙 40 维 latent + 共享 6DoF 位姿

`configs/joint_minimal.json` 使用原训练病例的 11、21 两颗中切牙，三台已知相机。准备器重新渲染只含这两颗牙的场景，排除其他牙齿和牙龈；没有固定于真值的其他牙齿帮助锚定位姿。该场景与前面完整上颌观测的 Shape-only 不同，因此另外运行同输入的固定真值位姿对照。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\prepare_joint.py --config configs\joint_minimal.json
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_joint.py --config configs\joint_minimal.json --fit-input runs\<joint-case-id>\fit_input
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\evaluate_joint.py runs\<joint-run-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_joint_review.py runs\<joint-run-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\diagnose_joint_observability.py runs\<joint-run-id>
```

拟合先用扰动初始 shape 的图像质心做粗位姿对齐，然后用投影轮廓同时优化全部 46 维。每轮刷新 DMM 网格、遮挡和轮廓对应；位姿采用 SO(3) 旋转向量与平移，latent 导数通过隐式 SDF 表面求根传递。Jacobian 有限差分包含 6 个位姿轴、每牙 latent 方向及混合方向；检查的是向量残差导数，不对硬 IoU 求导。初始码阻尼是数值设置，仍未建立经过校准的统计 shape prior。

Joint 输入只含已知相机、逐牙掩码、扰动初始 codes/pose、训练尺度及采样域。GT codes、GT pose、源网格、深度和 65° 预留观测不进入拟合器。生成器和拟合后的独立评估才读取真值。评估将 canonical 形状误差和经过拟合位姿的 world 几何误差分开，防止位姿与形状互相补偿时仅凭图像宣布通过。

### 已完成结果：联合链路有效，独立验收失败

运行 `joint_20260930T101555Z_ec4c7cfe`，seed101、latent 0.25 sigma、初始旋转向量分量 [7,-5,9]°、平移 [0.08,-0.06,0.05] DMM 单位。状态为 **JOINT_MINIMAL_FAIL**：

| 指标 | 初始 | 最终 | 判断 |
| --- | ---: | ---: | --- |
| 三视角逐牙平均 IoU | 0.1444 | 0.9873 | 通过 |
| 65° 预留 IoU | 0.1470 | 0.9602 | 通过 |
| 旋转误差 | 12.45° | 2.41° | 超过 1° 门槛 |
| 平移误差（DMM 单位） | 0.11180 | 0.02378 | 超过 0.01 门槛 |
| canonical 平均表面误差 | 0.004182 | 0.006839 | 增大 63.5%，失败 |
| world 平均表面误差 | 0.08884 | 0.002478 | 降低 97.2%，通过 |

Joint 梯度检查最大相对误差 1.27e-7；两牙网格均封闭且单连通。共 10 轮，按拟合视角 IoU 选第 5 轮，不使用真值或预留观测选择结果。5 次局部求根失败经减半步长重试恢复，日志和失败验收门槛全部保留。

同场景的匹配 Shape-only 对照 `shape_only_20260930T103215Z_603f9922` 通过：初始码、尺度、采样域、相机、掩码及预留视角哈希一致，固定生成真值位姿；canonical 平均表面误差降低 81.8%，拟合 IoU 0.9348 → 0.9918，预留 IoU 0.9151 → 0.9877。这支持优先调查联合求解的初始化、位姿与形状补偿以及局部极小；尚不能据此宣称存在严格的不可辨识性。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\prepare_joint_shape_control.py runs\<joint-case-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_shape_only.py --config runs\<control-case-id>\control_config.json --fit-input runs\<control-case-id>\fit_input
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\evaluate_shape_only.py runs\<control-fit-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_shape_review.py runs\<control-fit-id> --output semantic_review.png
```

详细证据、独立指标、匹配对照及 3D/语义图见 [Joint 核查报告](runs/joint_20260930T101555Z_ec4c7cfe/review.md)。`summary_verification.json` 复核了 6 个拟合依赖/归档代码、9 个 DMM 源码、冻结权重和匹配输入；原 Shape-only 核心脚本未修改。“固定扰动 shape 的轮廓 pose 预优化 → Joint”对照已完成，结果见下节。

### 固定扰动 shape 的轮廓 pose 预优化对照

`configs/joint_warmup.json` 保持原 Joint 配置及验收条件，直接使用原 `joint_case_20260930T101534Z_ec4c7cfe/fit_input`，增加一个固定扰动 shape 的 pose 预优化阶段。独立入口 `run_joint_warmup.py` 在启动时逐项核对原配置和输入 manifest，保留原 `run_joint.py` 及其依赖。新增 `pose_warmup.py` 只投影固定初始网格的轮廓点并优化 6DoF，不更新 latent，不读取真值或预留视角。

每轮刷新遮挡与双向逐牙轮廓对应，使用位姿解析 Jacobian，最多 60 个外层、每次最多 100 个局部函数调用；至少运行 10 轮，连续 12 轮未超过最佳拟合 IoU 1e-5 后停止。按拟合视角 IoU 选择预优化位姿，然后释放全部 46 维，原 Joint 的优化预算、局部/全局边界、初始码阻尼和停止条件保持一致。预优化记录在 `warmup_report.json`，独立保存 `warmup_codes.npz`、`warmup_pose.json`、梯度核对及渲染。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_joint_warmup.py --config configs\joint_warmup.json --fit-input runs\joint_case_20260930T101534Z_ec4c7cfe\fit_input
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\evaluate_joint.py runs\<joint-warmup-run-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_joint_review.py runs\<joint-warmup-run-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\review_joint_warmup.py runs\<joint-warmup-run-id>
```

复用同一独立 Joint 评价器。`review_joint_warmup.py` 在评价结束后比较原基线和新运行，核查全部原 Joint 设置、门槛、输入包、源码、冻结模型、预优化 shape 未变及初始指标一致；此时才读取 GT pose，展示各阶段真实位姿误差，拒绝覆盖旧报告。

2026-10-04 完成运行 `joint_warmup_20261004T043110Z_ddd095ce`：预优化 32 轮，最佳第 20 轮，IoU 0.8530 → 0.9531；固定初始代码和网格核查通过。随后 Joint 10 轮，选第 5 轮，梯度最大相对误差 1.47e-7，无局部求根失败重试。

| 最终指标 | 原 Joint | 增加 pose 预优化后 | 新运行判定 |
| --- | ---: | ---: | --- |
| 三视角拟合 IoU | 0.9873 | 0.9897 | 通过 |
| 65° 预留 IoU | 0.9602 | 0.9731 | 通过 |
| 旋转误差 | 2.41° | 1.98° | 超过 1° 门槛 |
| 平移误差（DMM 单位） | 0.02378 | 0.01572 | 超过 0.01 门槛 |
| canonical 平均表面误差 | 0.006839 | 0.001811 | 相对同一初始 shape 降低 56.7%，通过 |
| world 平均表面误差 | 0.002478 | 0.001400 | 相对同一原始输入降低 98.4%，通过 |

状态仍为 **JOINT_MINIMAL_FAIL**，只剩旋转和平移门槛失败；逐牙 canonical 均值/P95、网格完整性及其他检查全部通过。预优化阶段真实 pose 误差为 1.744° / 0.01054，释放 latent 后增至 1.982° / 0.01572。预优化明显改善了最终形状，但不足以同时恢复正确 pose。下一步可在新对照中调查 latent 的局部刚体成分与共享 pose 的耦合、统计 prior 或分块更新策略，继续沿用验收门槛。

新 [完整报告与可视化](runs/joint_warmup_20261004T043110Z_ddd095ce/review.md)、[独立评价](runs/joint_warmup_20261004T043110Z_ddd095ce/evaluation.json) 和 [输入/门槛/源码/模型复核](runs/joint_warmup_20261004T043110Z_ddd095ce/warmup_comparison.json) 均已保存，旧失败基线保留。这仍是同一训练病例、两牙、单种子的已知相机无噪声实验。

## 第 4 项：Joint 的最小稳健性实验

本节按会话中的“第 4 项稳健性”组织，区别于源文档中的 M4 prior 校准。本次使用两牙 pose 预优化后的 Joint 作为固定控制；该控制仍未通过位姿门槛，因此稳健性试验用于记录敏感性，不推断前一阶段已经通过。

`configs/joint_robustness.json` 预先固定 5 个独立扰动：shape 初始化 seed202/303（同 0.25 sigma、同初始共享 pose），逐牙掩码 1 像素腐蚀/膨胀，以及删除正面视角、仅保留左右 30°。每次只改变一个因素，冻结模型、初始码尺度、所有求解设置和原验收阈值；没有新增统计 prior、牙位或遮挡策略。掩码使用 3×3 核；膨胀的冲突按距原标签区域最近分配，保持原内部 FDI。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_joint_robustness.py --plan configs\joint_robustness.json
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\render_robustness_inputs.py runs\<robustness-batch-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\review_joint_robustness.py runs\<robustness-batch-id>
```

批次新建 `runs/joint_robustness_*`，原始控制不覆盖。输入生成器只在新 case 中改写指定因素并保存 `robustness_contract.json`、原始干净三视角和独立 truth；拟合器不读取干净评价观测及几何真值。运行器最多同时执行 2 个进程，每个使用 CPU 4 线程；单项执行或验收失败不会阻止其他项。计划与 10 个依赖源码在启动前归档，实时状态在 `status.json`，每项保存完整进程日志。`--resume runs\<batch-id>` 继续尚未启动的 prepared 项；记录了运行/中断子进程时拒绝重复启动，应先检查对应 PID 和进程日志。

`evaluate_joint_robust.py` 在拟合后同时记录“受扰动输入上的拟合 IoU”和“所有三个原始干净视角上的 IoU”；绝对图像门槛应用于干净三视角。删去的正面视角只进入事后评价，65° 仍作为另一独立预留视角。GT pose、canonical/world 表面、逐牙均值/P95、闭合连通和梯度检查使用原验收阈值；噪声掩码高 IoU 不能替代干净观测与 3D 恢复。

这是一个病例、两颗牙、小扰动的 pilot；3 个 seed 不提供统计成功率。没有测试未知相机、错 FDI、缺牙或嘴唇/牙龈遮挡。少一视角沿用原未归一化轮廓残差求和，因此相对阻尼权重也变化，不能把结果完全归因于视角几何信息。报告保留全部失败，控制本身未通过时不标记整体稳健性通过。

### 已完成稳健性 pilot

批次 `joint_robustness_20261004T044624Z_1d578127` 完成全部 5 个独立评价，0 次执行失败，完整 Joint 门槛通过 0/5；状态为 **ROBUSTNESS_PILOT_NOT_ESTABLISHED**。

| 扰动 | 干净三视角 IoU | 65° IoU | 旋转误差 | 平移 DMM | canonical 误差减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始化 seed202 | 0.9895 | 0.9655 | 5.595° | 0.03154 | 25.4% |
| 初始化 seed303 | 0.9894 | 0.9732 | 4.782° | 0.03275 | 22.7% |
| 边界腐蚀 1 像素 | 0.9261 | 0.8998 | 1.657° | 0.00777 | -37.1%（恶化） |
| 边界膨胀 1 像素 | 0.9497 | 0.9597 | 3.649° | 0.03678 | 16.8% |
| 只保留左右斜侧 | 0.9884 | 0.9670 | 1.801° | 0.01281 | 57.7% |

新初始化虽然图像分数相近，pose 误差变大；seed303 还有逐牙均值/P95 失败。腐蚀的扰动目标 IoU 达到 0.9770，但干净观测、预留视角及 canonical 形状明显变差；膨胀也未达到形状改善门槛。只保留左右斜侧在本例与控制接近，不能把其位姿失败归因于删除正面，也不能推广为两视角足够。

完整 [报告与可视化](runs/joint_robustness_20261004T044624Z_1d578127/review/report.md)、[结果解释](runs/joint_robustness_20261004T044624Z_1d578127/review/assessment.md) 和 [完整性及逐项结果](runs/joint_robustness_20261004T044624Z_1d578127/review/summary.json) 已保存。所有原输入、控制运行及失败保留；冻结模型、DMM 源码、原求解依赖与本批次归档代码复核通过。

## 重训前：训练数据与模型核查

2026-10-04 完成只读审计 `training_audit_20261004T044936Z`。全量核对 1101 个训练/测试质心、575 个原病例变换及冻结哈希；抽检 23 个 NPZ；检查全部训练 latent 和 126 个病例/牙位的变换场。原数据、DMM、checkpoint 均未修改，未启动训练。

主要发现：原例坐标/尺度处理链路一致，独立相似尺度范围 0.02064–0.02831；镜像未重新对齐、模板平均质心只使用原例，因此相对增强分布存在约 0.028–0.033 DMM 的每牙均值差；牙齿分支是空间变化的 SE(3) 场，latent 可改变局部刚体成分；latent L2 系数硬编码为 1e6，但旧日志不足以认定其过强。此外，SE(3) 零角度转换存在非有限值，center loss 的相对权重随 batch size 改变，配置中的距离 clamp/gradient clip 未在当前训练路径实际执行。

重训前应先确定新模型参数契约及共同刚体模式的约束，再生成镜像/模板参考一致的新数据版本，配置化并归一化训练损失，引入独立验证。历史 WSL 训练数据未与本地逐字节比对，当前 train 源码也没有历史执行归档；核查结论按当前源码、本地冻结数据、日志及 checkpoint 区分证据范围。

详细发现、可视化、实测统计和重训顺序见 [审计报告](runs/training_audit_20261004T044936Z/review.md)。复现：

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\audit_dmm_training.py
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\review_dmm_training_audit.py runs\<training-audit-id>
```

## 跨训练病例的完整上颌 Shape-only

`configs/shape_crosscase.json` 固定三种 latent 距离分位（25%、50%、75%），排除镜像、原病例及不含全部 14 个牙位的条目。距离为 280 维训练代码对训练均值的标准化 RMS，尺度沿用全部训练行的每维标准差。它仅用于预先分层选样，不作为解剖多样性的证据，也不按拟合结果挑选病例。每例完整开放 14 牙，使用 0.25 sigma、seed101/202，沿用上一扩大批次的数值设置与验收条件。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\run_shape_crosscase.py --prepare-only
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' -u scripts\run_shape_crosscase.py --resume runs\<campaign-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\inspect_shape_crosscase.py runs\<campaign-id>
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\summarize_shape_crosscase.py runs\<campaign-id>
```

病例清单与核心代码在开始前归档。生成器在本项目内新建各病例的前向目录；完整性失败的生成形状保留并记为源几何失败，其他预选病例仍继续，失败病例不替换。成功生成的观测依次进入两次独立扰动拟合和事后真值/预留视角评价。中断的前向运行保留为原尝试，新尝试另建目录；恢复时识别已有子批次并跳过已完成结果，已记录子进程尚在运行时拒绝重复启动。

已完成批次 `shape_crosscase_20260930T100642Z_b60557b7`：`C6C00RHE_upper`（50%）2/2、`01F4RGN8_upper`（25%）2/2、`014X0F4K_upper`（75%）1/2 通过，共 **5/6 拟合通过、2/3 病例两次均通过**。候选为 267 个，模型 epoch295，训练映射共 1052 行。这仍属于模型已训练病例的合成自洽实验。

全部最终拟合 IoU 为 0.9762–0.9808，65° 预留 IoU 为 0.9729–0.9826，平均表面误差降低 73.6%–80.4%。失败例 `014X0F4K/seed101` 的 FDI 27 表面均值增加 22.6%、P95 增加 24.3%，尽管整体指标改善，仍按逐牙门槛判失败。当前三视图配置尚未通过全部跨病例验收。

原评价有 3 次因零面积三角形而产生 NaN。所有 6 次统一使用相同面积样本、随机种子、最近 32 候选和原门槛进行有限距离复评：非有限候选使用解析边/顶点及平面距离回退，网格和原评价不改写，全部原有限指标保持不变。原 controller 状态保留在 `status.json`，复评结论见 `review/summary.json` 的 `cases_assessed`。完成旧执行代码核对后，距离修复已接入后续官方评价器，新批次代码归档也包含 `surface_metric_utils.py`；旧执行版本和接入后版本分别归档。

```powershell
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\review_shape_surface_metrics.py runs\<batch-id>\jobs\<job-id>\fit_attempt_01
& 'D:\WorkSpace\Dental\teethDMM\.venv\Scripts\python.exe' scripts\summarize_shape_crosscase.py runs\<campaign-id> --surface-review --output-folder review_v1
```

复评工具拒绝覆盖已有复评目录，汇总输出也要求新目录。详细结果、失败牙位和数值修复证据见 [跨病例核查报告](runs/shape_crosscase_20260930T100642Z_b60557b7/review/assessment.md)、[逐 seed 复评结果](runs/shape_crosscase_20260930T100642Z_b60557b7/review/report.md)、[原始结果](runs/shape_crosscase_20260930T100642Z_b60557b7/review_raw/report.md) 及 [哈希复核](runs/shape_crosscase_20260930T100642Z_b60557b7/review/provenance_verification.json)。84 个牙位记录中 47 个 latent 数值误差增大，不能解释为唯一真值代码恢复；跨训练病例结果也不能解释为真实照片或未见患者泛化。

## Joint 耦合与边界容差对照（2026-10-04）

新增分块 pose/latent 更新、C1 轮廓容差，以及排除当前病例和镜像行的统计 latent 先验对照。完成两牙 9 组和四牙 1 组有效实验，**0/10 按原门槛通过**；另一次统计先验尝试因漏排镜像行中断保留，不计入结果。未修改历史求解器、DMM 网络或 checkpoint。

四牙 FDI 11/12/21/22 的拟合 IoU 为 0.982673，65° IoU 为 0.968043，canonical 平均误差改善 44.32%，但旋转误差 2.05597°、平移误差 0.0217572 DMM，仍只在两个位姿门槛失败。解析梯度和溯源核查通过，不能替代恢复验收；当前不支持进入大规模稳健性扩展。

方法、原门槛、全部结果、可视化、局部 Jacobian 耦合诊断及复现命令见 [本轮报告](docs/synthetic_joint_stabilization_20261004.md)。

## 第 4 项：四牙 Joint 单因素稳健性诊断（2026-10-04）

按最新四牙基线完成干净重放与六项扰动：shape 初值 seed202/303、同模长反向 pose、1 像素腐蚀/膨胀、删除正面。干净重放各预设数值差均为 0；六项拟合和独立评价全部完成，执行失败 **0**，原完整 Joint 门槛 **0/6** 通过。冻结 solver/prior/预算/阈值，每个扰动重新进行图像 pose warmup；这轮为小规模诊断。

两个新 shape 初值新增逐牙均值/P95 失败；腐蚀/膨胀的干净三视角 IoU 分别降到 **0.9021/0.9165**；删除正面新增 FDI12 均值退化 **8.61%**，尽管干净 IoU 仍为 **0.9839**。反向 pose 未新增失败项，所有条件仍未达原位姿门槛。单例四牙结果不足以认定真实照片或整牙弓稳健性。

详细 [结果解释及复现](docs/synthetic_joint_robustness_four_20261004.md)、[全部结果与可视化](runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/report.md)、[完整指标及 78 项溯源核对](runs/joint_robustness_four_20261004T061410Z_de71c9c1/review/summary.json)。历史两牙批次、DMM 模型与源数据均保持冻结；局部遮挡留到显式有效区域契约完成后单独检查。
