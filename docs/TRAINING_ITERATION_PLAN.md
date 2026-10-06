# Git 协作与分轮训练计划

本项目第一阶段：已知相机、静态上下颌、逐牙语义掩码驱动的合成重建。先训练上下颌各一套 10 维耦合 DMM 先验，再验证多视图拟合。训练先验使用已处理的三维数据；本轮不训练照片分割器。目标机：RTX 5090 / 128GB / Windows + WSL2 Ubuntu，Python 3.10（系统解释器或 Linux Conda 均可）。

**当前只安排 R0、R1。R1 完成后交回反馈，停止继续长训。** 后续轮次由结果决定修改，提交新代码和配置，再在目标机更新。源码和接口已具备启动小规模验证的入口，模型质量与目标机环境尚待实际验证。

## 仓库边界

本地仓库根目录为 `D:\WorkSpace\Dental\multiview_teeth_reconstruction`，主分支 `main`，远程为 [JohnTitor-elpsykongroo/multiview_teeth_reconstruction](https://github.com/JohnTitor-elpsykongroo/multiview_teeth_reconstruction)。本次部署标签为 `training-r0-v3`：改用 Python 3.10 及兼容的依赖，DMM 模型源码、损失和训练 JSON 配置不变，旧标签保留。环境改变后必须重新运行 R0，不能复用旧预检或跨环境续训。不要使用官方 DMM 仓库替换本项目。

- 纳入：DMM 修改源码、内置 torchmeta、nvdiffrast 源码及 SM12 修复、项目脚本、配置、接口、测试、研究和部署文档。
- 不纳入：数据、权重、运行输出、Python/CUDA 环境、编译产物、访问凭据。`runs/` 和 `.venv/` 是本地输出。
- DMM 与 nvdiffrast 作为普通源码目录跟踪，不使用子模块。上游地址和 donor commit 记录在各自 `UPSTREAM.json`；项目 commit 与上游 commit 分开记录。原有嵌套 Git 元数据保存在本机忽略目录 `.vendor_git_history/`。
- 仓库内 [主设计文档](multiview_teeth_reconstruction_semanticxy.md) 是今后跨机器同步的版本；它从原项目外部文档复制而来。外部文件保留，后续修改以仓库内版本为准。
- 历史文档提到的本机绝对路径和历史 `runs/` 不随 clone 搬运。本轮仅使用 `configs/training_handoff_v1/` 及本页命令；历史入口不代表可直接在新机器运行。

## 目标机首次部署

如果尚未安装 WSL，在 Windows 管理员 PowerShell 执行以下命令，按提示重启并完成 Ubuntu 用户创建。已有 Ubuntu 22.04/24.04 不必重新安装；当前只需要可用的 Linux Python 3.10。

```powershell
wsl --update
wsl --install -d Ubuntu-24.04
wsl --list --verbose
```

确认 Ubuntu 使用 WSL 2；若列表显示版本 1，执行 `wsl --set-version Ubuntu-24.04 2`。在 Windows 安装支持 RTX 5090 的 NVIDIA 驱动。128GB 主机可在 Windows 用户目录的 `.wslconfig` 中合并以下配置，保留已有其他设置；执行 `wsl --shutdown` 后重新打开 Ubuntu 生效，注意该命令会停止所有 WSL 任务。

```ini
[wsl2]
memory=96GB
swap=16GB
```

参考：[Microsoft WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)、[WSL 配置说明](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)、[NVIDIA WSL CUDA 说明](https://docs.nvidia.com/cuda/wsl-user-guide/)。

以下命令全部在 Ubuntu 终端执行。`$HOME/dental` 是新部署示例；已放在 `/mnt/e/实际目录/dental` 的项目可以保留当前位置，只需进入对应项目目录，代码与 data 维持同级关系。已有 checkout 跳过 clone，按下面“已有项目更新”操作。

```bash
sudo apt update
sudo apt install -y git tmux rsync
mkdir -p "$HOME/dental"
git clone https://github.com/JohnTitor-elpsykongroo/multiview_teeth_reconstruction.git "$HOME/dental/multiview_teeth_reconstruction"
cd "$HOME/dental/multiview_teeth_reconstruction"
git switch --detach training-r0-v3
git status --short                 # 应无输出
git rev-parse HEAD                 # 记录到实验笔记
```

另行复制两个数据目录，保留全部内部路径：

```text
~/dental/
  multiview_teeth_reconstruction/
  data/
    Teeth3DS_DualArch_v1/
    Teeth3DS/
```

训练清单还引用原始 Teeth3DS OBJ/JSON，不能只复制处理后样本。既有移交资源清单统计约 75.3 GB；完整复制数据目录可能更大，另外为环境、检查点和后续实验预留空间。冻结数据不通过 Git 搬运，不修改其中的 manifest 或资源哈希。首次预检由加载器完整检查哈希与患者划分。

例如，已把这两个目录搬到目标电脑 Windows 的 `D:\WorkSpace\Dental\data` 时，可以在 Ubuntu 执行下面的复制命令。若中转盘位置不同，只修改 `TRANSFER`，不要修改训练配置中的相对路径。

```bash
TRANSFER=/mnt/d/WorkSpace/Dental/data
test -d "$TRANSFER/Teeth3DS_DualArch_v1" && test -d "$TRANSFER/Teeth3DS"
mkdir -p "$HOME/dental/data"
rsync -a --info=progress2 "$TRANSFER/Teeth3DS_DualArch_v1" "$HOME/dental/data/"
rsync -a --info=progress2 "$TRANSFER/Teeth3DS" "$HOME/dental/data/"
```

确认数据复制完成、`nvidia-smi` 能看到 RTX 5090 后再安装环境。脚本建立项目 `.venv`，固定 Python 3.10、PyTorch 2.10.0 / torchvision 0.25.0 / cu130 以及 requirements 中的依赖；版本组合见 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)。现阶段使用预编译 PyTorch，无需先装 CUDA Toolkit 或编译渲染器。已有 `.venv` 时脚本会拒绝覆盖，避免破坏环境；安装失败先保留报错再处理。

先准备 Python 3.10，以下两种方法选其一。如果当前系统或已激活 Conda 的 Python 已是 3.10，可直接运行安装脚本，无需另建 bootstrap 环境。

- Ubuntu 22.04 系统解释器：`sudo apt install -y python3.10 python3.10-venv`。
- 已安装 Linux Conda（适用于当前截图中的 base 环境）：

```bash
conda create -n dental-bootstrap310 python=3.10 pip -y
conda activate dental-bootstrap310
python --version
```

Conda 环境仅提供解释器，训练依赖由脚本装到项目 `.venv`，不污染 base；保留 bootstrap 环境，因为 `.venv` 依赖它的基础解释器。脚本依次检测 `python3.10`、`python3`、`python`，必须是 Linux 3.10；也可用 `PYTHON_BIN="$(command -v python)" bash scripts/setup_training_wsl.sh` 显式指定。找不到 3.10 或缺少 venv/ensurepip 时会在创建环境前明确报错。依赖固定为 NumPy 2.2.6、SciPy 1.15.3、scikit-image 0.25.2、Pillow 11.3.0、ordered-set 4.1.0，pip 固定 25.3。参考：[Conda 环境管理](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)、[Ubuntu 22.04 venv 包](https://packages.ubuntu.com/jammy/python3.10-venv)。

已有项目更新（先结束所有训练；在实际项目根目录执行）：

```bash
git fetch origin --tags
git switch --detach training-r0-v3
```

此前 `python3.12: command not found` 发生在创建 `.venv` 前，通常没有残留环境。若后来已创建旧 `.venv`，先退出激活状态，保留旧环境备份再安装，不删除历史产物：

```bash
# 仅在 .venv 已存在时执行；移动前确保已 deactivate 且没有训练进程。
if [ -e .venv ]; then
  mkdir -p runs/environment_backups
  mv -- .venv "runs/environment_backups/venv_before_py310_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
```

备份仅供恢复/排错，虚拟环境移动后不可直接作为新路径下的可运行环境。安装完成后，后续训练入口会自动激活项目 `.venv`。

本次兼容性验证：在全新 Windows Python 3.10.21 隔离环境安装同版 CPU PyTorch 和上述科学计算依赖，`pip check` 通过，26 项训练与交接回归测试全部通过、无跳过；另成功解析官方 Python 3.10 / Linux cu130 的 torch、torchvision wheel 及科学计算依赖。该证据不替代目标 RTX 5090 / WSL 的 R0 实测，也不代表模型训练质量通过。

```bash
nvidia-smi
bash scripts/setup_training_wsl.sh
bash scripts/run_training_wsl.sh preflight
```

安装脚本与固定依赖见 [环境交接说明](TRAINING_HANDOFF_20261006.md)。Windows 安装 NVIDIA 驱动，WSL 中不另装 Linux 显卡驱动。训练本身无需 nvdiffrast；图像拟合的编译及梯度验收留到后续。

## 分轮实验与推进条件

| 轮次 | 预算与目的 | 需要查看的证据 | 下一步 |
| --- | --- | --- | --- |
| R0 目标机预检 | CUDA 前后向；训练测试；完整数据校验；每颌真实病例固定采样 20 步，再按 pilot 的 256 点/组件 + 2048 非表面点跑 2 步 | `report.json`、测试日志、优化器与配置采样 smoke JSON、显存峰值、运行时版本 | 全部通过且状态为 `TARGET_READY_FOR_BOUNDED_PILOT` 才进入 R1；否则带失败报告反馈 |
| R1 链路与短程 pilot | 每颌先 2 个训练病例、1 epoch；再 32 个原始训练病例、4 个原始 val、2 epoch；上下颌顺序运行 | 逐项损失、有效点数、梯度范数、裁剪前范数、学习率、非有限值、每个 val 的场与几何诊断、检查点完整性 | **打包反馈后停止**；2 epoch 不足以证明收敛或泛化 |
| R2 受控诊断（反馈后发配置） | 优先将同一 32 病例配置延长至约 20 epoch；若 R1 暴露实现/数值问题，先固定少数病例做记忆实验 | 同一验证预算下的学习曲线、逐牙误差、网格边界/缺牙/连通性、训练病例能否重建；每次只改一项主要因素 | 明确是训练不足、实现/尺度问题还是泛化不足后再扩展 |
| R3 中等规模（反馈后发配置） | 候选 128 个 train / 16 个 val、20 epoch 起步；固定患者划分与种子 | 有效网格比例、病例与逐牙均值/P95/最差值、失败清单、缺牙分组、显存与耗时 | 冻结配方、训练/验证预算及验收表后才开始全量 |
| R4 全量先验 | 上颌 1290、下颌 1276 个 train（含镜像）；原始 val 88/86；100 epoch 仅为候选起点 | 冻结预算选择 best_val；原始训练病例统计；导出检查；模型与逐牙几何评审 | 所有选定 val 有效才产生 best_val；测试集只在最终方案冻结后做一次独立评估 |
| R5 重建接入 | 两颌先验各自通过后，已知相机合成掩码；依次 Pose-only、Shape-only、Joint | 位姿绝对误差、逐牙几何、遮挡/缺牙、梯度、失败率、防穿插门槛 | 对重建另行验收；先验训练通过不代表图像拟合已通过 |

R2 以后为计划，不是本轮已经发放的执行指令。R2 小病例实验不直接导出最终先验：10 维 latent 的协方差统计要求每个存在组件有至少 11 个原始训练病例支撑。当前固定 canonical 参考来自冻结的原始 train 集；pilot 的 decoder 更新及验证 latent 均值仅使用所选训练病例。验证时只拟合临时代码，不更新 decoder，fit/evaluation 点与中心监督分开，test 不进入调参。

初期 `no finite zero surface` 或 `non-positive domain boundary` 必须保留失败。若优化器和保存链路正常，允许继续完成 R1 的短预算以采集证据；不能以 loss 下降或聚合 IoU 变好替代逐牙几何通过。出现 NaN/Inf、CUDA OOM、哈希/患者泄漏、保存或续跑错误，应停止该配置并反馈，不静默改采样/阈值或跳过病例。

## 本轮 R1 的执行命令

把 PROOF 改成 R0 成功报告实际路径。每一步失败后先停止并反馈；建议逐条执行，以便检查退出状态。

```bash
# 先进入实际项目根目录（包括 /mnt/e 上的项目也一样）
PROOF=runs/target_preflight_实际时间戳/report.json
bash scripts/run_training_wsl.sh smoke upper "$PROOF"
bash scripts/run_training_wsl.sh smoke lower "$PROOF"
bash scripts/run_training_wsl.sh pilot upper "$PROOF"
bash scripts/run_training_wsl.sh pilot lower "$PROOF"
```

运行输出都是新目录，控制台日志在 `runs/logs/`。Git 检查要求源码目录干净，并把预检绑定到项目 commit、源码 SHA 和运行时。未跟踪的脚本/配置也会拒绝；个人实验笔记写入 `runs/`。不要临时改受控配置之后复用旧 PROOF。

## 反馈包

运行停止后，用实际目录替换占位符；有哪一步结果就收集哪一步，失败目录同样收集。不要等待“全部成功”才反馈。

```bash
source .venv/bin/activate
python scripts/collect_training_feedback.py \
  --run runs/target_preflight_实际时间戳 \
  --run runs/upper_smoke_实际时间戳 \
  --run runs/lower_smoke_实际时间戳 \
  --run runs/upper_pilot_实际时间戳 \
  --run runs/lower_pilot_实际时间戳 \
  --log runs/logs/pilot_upper_实际时间戳.log \
  --log runs/logs/pilot_lower_实际时间戳.log \
  --output runs/feedback_R1_001.zip
```

反馈包包含每个运行目录顶层的 JSON/JSONL/日志/图像、环境安装清单、检查点文件名/大小/SHA256。不会递归收集数据和 truth，也不包含权重；原机器保留全部检查点，必要时再单独传 `final.pth` / `best_val.pth`。预检或 smoke 的控制台日志也可重复添加 `--log`。每个文件默认上限 64 MiB，超限会明确列在 `feedback.json` 并返回非零退出码，需另传或提高 `--max-file-mb`，不会悄悄截断。

一起告诉本对话：执行了哪几步、每步大致耗时、是否中断/重试/手工修改、`nvidia-smi` 显存观察、任何异常图像。只发终点 loss 或挑选成功截图不足以定位问题。

## 本机修改 → 目标机更新 → 下一轮

1. 本对话检查反馈并给出原因、变更范围、下轮固定配置与通过条件。源码和配置变更先测试、提交，再创建新的训练标签（如 `training-r2-v1`）；旧标签不移动。
2. 等目标机训练结束、保存反馈及权重后再更新。**正在训练的 checkout 禁止 pull/switch 或编辑源码**，避免混用运行中 import 的不同版本。
3. 目标机执行：

```bash
git status --short
git fetch origin --tags
git switch --detach <本对话批准的下一轮标签>
# 依赖未变则复用环境；依赖变更按新版本说明建立新环境。
bash scripts/run_training_wsl.sh preflight
```

4. 按新一轮命令运行，再反馈。`runs/` 被忽略，旧运行不会因更新消失；不要执行 `git clean -fdx`。
5. 当前续跑要求 DMM 源码指纹、运行时和配置兼容，只允许既定的 epochs/保存频率变更。修改模型、损失、采样或数据后通常重新训练；不得手工改检查点哈希绕过校验。跨版本热启动需要单独实现和评审迁移路径。

本轮不自动启用 `formal`。后续训练时长和最终误差阈值根据 R1–R3 证据制定，不预设“100 epoch 一定收敛”或未经校准的临床精度承诺。
