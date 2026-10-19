# DMM 多视图牙齿重建

**Git 训练协作入口：[分轮训练、目标机克隆与反馈计划](docs/TRAINING_ITERATION_PLAN.md)。** 项目根目录现作为独立仓库，DMM 和 nvdiffrast 修改源码直接纳入版本管理。数据、权重和环境另行迁移。用户已授权目标机 Codex 自主实施 R0–R4，必读 [自主实验指南](docs/AUTONOMOUS_R0_R4_RESEARCH_GUIDE.md)，其中包含论文、源码阅读路线、调试方法及推进标准。仓库内 [主设计文档](docs/multiview_teeth_reconstruction_semanticxy.md) 随代码同步。

远程仓库：[JohnTitor-elpsykongroo/multiview_teeth_reconstruction](https://github.com/JohnTitor-elpsykongroo/multiview_teeth_reconstruction)。Python 3.10 环境基线使用 `training-r0-v3` 标签（Python 3.10 / PyTorch 2.10.0 / cu130，DMM 源码和训练 JSON 配置不变；重新执行预检）。

RTX 5090 / WSL2 训练交接：[交接入口与修复证据](docs/TRAINING_HANDOFF_20261006.md)。新配置在 `configs/training_handoff_v1/`，使用非零参考初始化与稳定法线分母；脚本提供包校验、目标 GPU 预检、smoke、pilot、显式长训入口。正式几何质量尚待 pilot/训练验证。

**当前主线（2026-10-06）：直接使用Blender现成逐牙/组织标签，暂不开发或训练识别分割。** [oracle观测打包与阶段三接入](docs/stage2_oracle_workflow.md)已在8场景80张图上运行；后续识别模型采用本地SegmentAnyTooth。

2026-10-06 源码整合：[正式训练前的实现与验证说明](docs/training_readiness_v1.md)。新增显式训练配方、冻结 decoder 的验证码拟合与 best_val、训练子集统计约束、decoder 兼容绑定、双像素约定及均值模型候选初始化。配置位于 `configs/training_readiness/`；正式训练与真实重建质量验收尚未完成。

照片流程新增：[阶段二 RGB → 逐牙观测初步实现](docs/stage2_image_understanding_v1.md)；[阶段二 → 阶段三观测契约](docs/tooth_observations_v1.md)；[Blender pilot验收与RGB遮挡分支改进](docs/stage2_blender_acceptance_20261005.md)。两组pilot已用于输入验收和数值联调，25项测试通过；尚无正式分割训练或识别精度结论。

工作站部署与训练：[ThinkStation / RTX 5090 / 128GB 的 WSL2 环境配置与训练手册](docs/WSL_RTX5090_environment_and_training.md)，包含源码迁移、数据校验、GPU 预检、独立双颌训练、续跑、模型导出和可选联合拟合。

当前自主研究交接标签为 `training-autonomy-v1`；环境已经配置的目标机无需重装。

## 当前双颌开发约定

当前目标为**已知相机、静态上下颌、逐牙语义掩码驱动的合成重建**。参数与数据含义以 [双颌参数与数据约定 v1](docs/dual_arch_parameter_data_contract_v1.md) 和 [机器可读固定选项](configs/contracts/dual_arch_static_semantic_v1.json) 为准。已在 `third_party/DMM` 源码中实现单颌数据加载、独立训练入口、模型包导出、双颌场景组装与分阶段图像拟合；使用方式与数据格式见 [双颌接口使用说明](docs/dual_arch_interfaces.md)。真实数据训练尚未启动，历史运行器仍使用原协议。

V1 使用两套单颌 DMM、两个跨视角共享的整体刚体位姿、固定毫米尺度与显式牙位存在性；现有耦合 latent 不额外叠加逐牙位姿。形状与排列解耦另立后续模型版本。已由用户检查认可的 synthetic sanity check 作为参考保留。

SemanticXY 匹配、双颌软语义 / XY 渲染和隐式表面梯度桥已接入原生源码：[匹配协议](docs/semanticxy_matching_v1.md)、[渲染约定](docs/differentiable_rendering_v1.md)。CUDA 前后向与局部数值检查已完成，并修复 SM 12.x 依赖梯度累加问题；已新增 `fit-scene` 分阶段优化、不可见区域恢复和检查点续跑。使用方式、43 项通过的回归测试、细网格 AA 导数限制与全部短预算结果见 [GPU 与分阶段拟合说明](docs/staged_fitting_gpu_v1.md)。正式 DMM 训练仍未启动，安排在所需源码修改与扩展完成后。

可选扩展已提供 `make-dual-config` / `validate-dual-training` / `train-dual`，用于统一校验与调度两套独立训练；联合拟合可显式启用防穿插代理，默认关闭，不强制上下牙接触。详见 [上下颌训练与联合拟合可选扩展](docs/dual_arch_optional_extension_v1.md)。

## 项目入口与目录

2026-10-10 清理后，`scripts/` 保留 19 个当前入口。旧的单颌 CPU 实验与一次性诊断已移到[可恢复源码归档](archive/legacy_experiments_20261010/README.md)，历史结果说明见[完整旧版记录](archive/legacy_experiments_20261010/HISTORICAL_README.md)。[清理记录](docs/CLEANUP_20261010.md)列出范围、核对证据和释放空间。

| 用途 | 入口 |
|---|---|
| WSL 环境与训练 | `setup_training_wsl.sh`、`run_training_wsl.sh` |
| 训练交接与诊断 | `build_training_package.py`、`verify_training_package.py`、`training_target_preflight.py`、`training_smoke.py`、`collect_training_feedback.py` |
| Windows GPU 环境与编译 | `gpu_env.ps1`、`prepare_cuda_redist.py`、`build_gpu_backend.ps1` |
| 当前渲染/拟合回归 | `check_render_bridge.py`、`check_gpu_render.py`、`check_staged_fitting.py`、`check_dual_extension.py` |
| Blender 标注与阶段三 | `stage2_oracle.py`、`stage3_fuse.py` |
| 可复用 RGB 观测工具 | `stage2_observe.py`、`stage2_audit_dataset.py`、`stage2_evaluate.py` |

上述脚本位于 `scripts/`。当前双颌训练、模型导出、场景准备和拟合以 `third_party/DMM/dmm_cli.py` 为统一入口；具体命令见前述训练交接与接口文档。

- `configs/`：当前训练、渲染、双颌契约与 oracle 配置。
- `tooth_observation/`、`observation_fusion/`：阶段二和阶段三实现。
- `third_party/`：纳入版本管理的 DMM、nvdiffrast 源码和依赖。
- `tests/`：现有契约、训练、观测、渲染与拟合回归测试。
- `runs/`：本地运行证据；成功、失败、冻结输入与评估结果继续保留。
- `.runtime/`：本地已安装 Python/CUDA 和 GPU 扩展。
- `archive/`：旧实验源码、原配置、历史说明与恢复清单。

恢复历史脚本后，其协议仍限于各自记录的合成诊断；正式训练和真实照片重建的结论需按当前流程另行验证。

## 当前预处理数据的WSL训练入口（2026-10-18）

`Teeth3DS_DualArch_Preprocess/output/latest` 使用逐病例固定参考相似尺度。
对应入口为 `bash scripts/run_dmm_latest_wsl.sh validate|train upper|lower`，配置位于
`configs/dmm_latest_v2/`。完整命令、队列选择和断点恢复说明见
[DMM README](third_party/DMM/README.md#当前固定参考相似尺度数据集2026-10-18wsl)。
源码直接读取当前NPZ/PKL；该坐标契约与既有固定50mm模型/场景bundle不同，不能混用旧checkpoint。
