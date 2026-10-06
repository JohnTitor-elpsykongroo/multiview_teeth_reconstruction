# Git 协作与分轮训练计划

本项目第一阶段：已知相机、静态上下颌、逐牙语义掩码驱动的合成重建。先训练上下颌各一套 10 维耦合 DMM 先验，再验证多视图拟合。训练先验使用已处理的三维数据；本轮不训练照片分割器。目标机：RTX 5090 / 128GB / Windows + WSL2 Ubuntu 24.04。

**当前只安排 R0、R1。R1 完成后交回反馈，停止继续长训。** 后续轮次由结果决定修改，提交新代码和配置，再在目标机更新。源码和接口已具备启动小规模验证的入口，模型质量与目标机环境尚待实际验证。

## 仓库边界

本地仓库根目录为 `D:\WorkSpace\Dental\multiview_teeth_reconstruction`，主分支 `main`，初始训练版本标签 `training-r0-v1`。当前不设置远程。提供远程地址后再配置 origin 和推送；不要使用官方 DMM 仓库替换本项目。

- 纳入：DMM 修改源码、内置 torchmeta、nvdiffrast 源码及 SM12 修复、项目脚本、配置、接口、测试、研究和部署文档。
- 不纳入：数据、权重、运行输出、Python/CUDA 环境、编译产物、访问凭据。`runs/` 和 `.venv/` 是本地输出。
- DMM 与 nvdiffrast 作为普通源码目录跟踪，不使用子模块。上游地址和 donor commit 记录在各自 `UPSTREAM.json`；项目 commit 与上游 commit 分开记录。原有嵌套 Git 元数据保存在本机忽略目录 `.vendor_git_history/`。
- 仓库内 [主设计文档](multiview_teeth_reconstruction_semanticxy.md) 是今后跨机器同步的版本；它从原项目外部文档复制而来。外部文件保留，后续修改以仓库内版本为准。
- 历史文档提到的本机绝对路径和历史 `runs/` 不随 clone 搬运。本轮仅使用 `configs/training_handoff_v1/` 及本页命令；历史入口不代表可直接在新机器运行。

## 目标机首次部署

远程就绪后，替换下列地址占位符。训练代码和数据放 WSL Linux ext4 文件系统，不直接在 `/mnt/d` 上长训。

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev git tmux
mkdir -p "$HOME/dental"
git clone <稍后提供的远程地址> "$HOME/dental/multiview_teeth_reconstruction"
cd "$HOME/dental/multiview_teeth_reconstruction"
git switch --detach training-r0-v1
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
cd "$HOME/dental/multiview_teeth_reconstruction"
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
