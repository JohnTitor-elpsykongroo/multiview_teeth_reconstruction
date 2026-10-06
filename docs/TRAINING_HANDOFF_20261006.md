# RTX 5090 / WSL2 训练交接

**更新：以 [Git 分轮训练计划](TRAINING_ITERATION_PLAN.md) 为当前执行入口。** 当前标签 `training-r0-v3` 使用 Python 3.10，新机器通过本项目 Git 仓库及固定标签同步，先做 R0/R1 后反馈，暂不执行本文的 formal 示例。下文 ZIP 搬运是此前交付方式，保留用于旧包核验；Git checkout 无需 `training_package_manifest.json`，校验脚本会改为检查干净的项目 commit。DMM donor 身份由 `UPSTREAM.json` 提供，不依赖嵌套 `.git`。

当前交接对象：ThinkStation、RTX 5090、128GB 内存、Windows + WSL2。目标是两套独立的 10 维耦合 DMM 先验；训练期间不需要人脸、视频、相机或 nvdiffrast。

## 本版修复与证据

1. **全零参考输出造成初始法线梯度尖峰**。旧 SIREN 的输出层权重和 bias 均为零，法线余弦项的默认 `eps=1e-8` 放大导数。增加显式 `NetworkSpecsRef.output_initialization=siren`，输出权重按 SIREN 范围初始化、bias 为零；旧默认保留兼容。新训练配置使用 `normal_epsilon=0.001`，并显式设置全局梯度裁剪阈值 1000。
2. **固定条件对照**。上颌同病例同种子、16 点/组件 + 32 非表面点，旧初始 reference 梯度范数约 3.31e9；仅改变初始化后约 3915；仅改变 epsilon 后约 33144。组合设置在该病例与非零初始化对照数值一致，没有单靠裁剪掩盖根因。
3. **实际 Adam 更新**。上下颌各一个真实原始训练病例各 20 步，固定采样损失比例分别约 0.324 / 0.339，全程 loss、梯度、参数有限。峰值约 1.11 / 1.18 GB，仅适用于这个小采样检查。完整模型的小样本过拟合并不等于泛化或几何合格。
4. **迁移与续训**。源码哈希改为统一 POSIX 相对路径排序，排除 Windows/Linux 的文件枚举差异。训练记录 Python、Torch、CUDA、GPU 与数值模式，续跑拒绝运行时不匹配；新运行保存源码快照、配置、优化器/调度/RNG、`runtime.json`，异常保存 `failure.json`，不覆盖旧目录。
5. **数据入口**。本机已完整调用上下颌加载器校验原始资源、样本、中心、清单和患者划分。原始冻结数据未改写，新 specs 保存在项目配置目录。

训练与图像拟合的数值门槛须区分：GPU 原生 AA 含浮点原子累加，旧的严格轨迹重放及细网格整图 AA 有限差分门槛仍未通过；本次没有靠放宽阈值宣称通过。这些路径不参与隐式先验训练，交接包中不依赖 nvdiffrast。恢复参数状态已有精确测试；训练完成后的图像重建仍需独立验收。

## 搬运内容与目录

使用本轮生成的 `dmm_training_source_v1.zip`，不要重新 clone 官方仓库替代它。压缩包包含当前全部 DMM 源码及 Git 信息、项目内 torchmeta、训练配置、测试与交接脚本；不包含 Windows 环境、历史权重或完整数据。

另复制下面两个数据目录，保持原相对位置：

```text
~/dental/
  multiview_teeth_reconstruction/      # 解压源码包得到
  data/
    Teeth3DS_DualArch_v1/              # 已处理数据
    Teeth3DS/                         # 清单引用的原始 OBJ/JSON 也必须带上
```

交付目录的 `delivery.json` 记录源码 zip SHA256 和数据资源字节数；`data_files.txt` 列出必须搬运的资源。`training_package_manifest.json` 给出逐文件哈希。数据未打入源码 zip，应另行复制；可完整复制上述数据目录，或按清单选择资源。不要重命名内部文件或修改原始 manifest 的路径。

目标机在 WSL 的 Linux ext4 目录运行，传输盘只作中转。在 Windows 安装 NVIDIA 驱动，WSL 内不要另装 Linux 显卡驱动。[NVIDIA WSL 官方说明](https://docs.nvidia.com/cuda/wsl-user-guide/)

## 安装与目标机检查

保留已有 WSL Ubuntu，按分轮计划准备 Python 3.10；128GB 主机可配置 96GB WSL 内存和 16GB swap。目标机尚未由本对话实际操作，不能将本机 RTX 5050/Windows 结果当成目标验收。

在 Ubuntu 执行：

```bash
sudo apt update
sudo apt install -y git unzip tmux
# Python 3.10 解释器按分轮计划使用系统包或 Conda 准备。
mkdir -p "$HOME/dental"
sha256sum /mnt/e/transfer/dmm_training_source_v1.zip
# 与 delivery.json 的 sha256 一致后解压；目标目录须为新目录。
test ! -e "$HOME/dental/multiview_teeth_reconstruction"
unzip /mnt/e/transfer/dmm_training_source_v1.zip -d "$HOME/dental"
cd "$HOME/dental/multiview_teeth_reconstruction"
nvidia-smi
bash scripts/setup_training_wsl.sh
bash scripts/run_training_wsl.sh preflight
```

当前 Git 版本安装脚本固定 Python 3.10、PyTorch 2.10.0 / torchvision 0.25.0 / cu130（旧 ZIP 环境不自动升级），来自[官方版本组合](https://pytorch.org/get-started/previous-versions/)，辅助依赖版本在 `configs/training_handoff_v1/requirements.txt`。本机验证环境为 Torch 2.14.0+cu130，因此目标机组合必须通过实际预检；不能直接将 Windows Conda 目录或 `.pyd` 拷入 WSL。

预检执行源码包哈希校验、完整数据资源校验、CUDA 前后向、训练单元测试、上下颌真实病例 20 步小采样优化，以及**按配置实际 256 点/组件 + 2048 非表面点运行 2 步**的显存/有限值检查。只在全部通过后输出 `TARGET_READY_FOR_BOUNDED_PILOT`。它不需要编译可微渲染器，也不会启动正式长训。

数据校验会读取大量原始文件，耗时明显长于读取 JSON；日志每 100/250 条报告进展。失败报告和已完成检查保存在新运行目录，修复问题后另建目录重试。

## 冒烟、pilot 与正式训练

先将下列 `PROOF` 改为目标机预检成功报告的实际路径：

```bash
PROOF=runs/target_preflight_实际时间戳/report.json
bash scripts/run_training_wsl.sh smoke upper "$PROOF"
bash scripts/run_training_wsl.sh smoke lower "$PROOF"
bash scripts/run_training_wsl.sh pilot upper "$PROOF"
bash scripts/run_training_wsl.sh pilot lower "$PROOF"
```

smoke 配置仅 2 个 train 病例、1 epoch、1 个 val 的极小预算，验收数据→优化器→验证→保存链路；刚开始训练没有零水平面或尚不能提取有效网格时，验证会保留失败、`best_val_score=null`，不会伪造 `best_val.pth`。这种 smoke 不能验收模型质量。

pilot 配置每颌 32 个原始 train、4 个原始 val、2 epoch。检查 `progress.jsonl` 的逐项损失、梯度、有效点和学习率，检查 `validation_*.json` 中的逐牙几何、语义、失败病例及预算。两轮仅是启动检查；若验证始终失败或几何无改善，需要继续小规模诊断，不能因训练 loss 下降就进入长训。

确认短程训练和独立验证有效后，显式启动完整数据训练：

```bash
tmux new -s dmm-training
bash scripts/run_training_wsl.sh formal upper "$PROOF"
bash scripts/run_training_wsl.sh formal lower "$PROOF"
```

当前 formal_candidate 是 100 epoch 起始预算，不是保证足够的优化轮数。正式配置含各自的镜像训练样本，上下颌顺序运行以控制显存；不共享 latent 或解码器。shell 入口要求当前包/源码和实际目标 GPU 预检一致，但**不自动替用户宣布 pilot 几何质量通过**。

## 中断、续跑、导出与故障

训练轮末写检查点；异常时保存 `failure.json`。保留原运行目录，从最后完整的 `epoch_*.pth` 或 `final.pth` 恢复，使用同一源码、同一运行时和相同配置；仅可增加 epochs 或改变保存频率。将修改配置保存为新文件时，必须正确重建相对资源引用。

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/third_party/DMM"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python third_party/DMM/dmm_cli.py train --config configs/training_handoff_v1/upper_formal_candidate.json \
  --resume runs/原运行/epoch_000010.pth --output runs/upper_resumed_新编号 --device cuda
```

所有权重导出继续使用训练时源码。没有 `best_val.pth` 时先查看验证失败，不应把 `best_train.pth` 称为验证最优。bundle 成功导出也仍是 `NOT_QUALITY_ACCEPTED`。

- 哈希或路径错误：按 `data_files.txt` 补齐资源，保持父目录结构；不改哈希来绕过校验。
- CUDA OOM：新建较小采样配置重跑预检；不要把 128GB RAM 当作显存。不在现有续跑中静默改变采样。
- 非有限值：保留 `failure.json`、配置与最后检查点，停止该配置；检查单位、数据、法线和数值项，不先禁用报错。
- 源码/运行时不一致：使用冻结快照和记录环境；不伪造来源字段。
- 新数据或配方需要重新开始实验；当前只保存完成轮的续跑状态，不支持精确恢复到被中断轮中间的任意病例。
