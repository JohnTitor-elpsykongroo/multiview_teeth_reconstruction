# ThinkStation / RTX 5090 / 128GB：WSL2 环境配置与 DMM 训练手册

**2026-10-06 Python 3.10 更新：当前执行以 [分轮训练计划](TRAINING_ITERATION_PLAN.md) 和 `training-r0-v3` 为准，使用 PyTorch 2.10.0 / cu130；以下旧 Python 3.12 手工环境命令不再用于本轮安装。**

更新：2026-10-05。适用当前项目 `third_party/DMM` 的 **manifest 独立单颌训练入口**，及可选双颌训练调度、已知相机联合拟合。

2026-10-06 交接更新：新机器请优先使用 [训练交接入口](TRAINING_HANDOFF_20261006.md) 和 `configs/training_handoff_v1/`。它包含本次修复后的非零 SIREN 参考初始化、稳定法线分母、版本化验证及可执行 WSL 脚本。下文保留通用环境和旧手工命令供排错；不要用旧默认 specs 重新生成配置替代交接配置。

## 0. 先确认范围

本手册按当前源码核对命令和数据格式，并查阅官方安装说明。**目标 ThinkStation 尚未实际执行这些步骤**，本文的 Linux 版本组合是待目标机验证的推荐基线，不能把本机 Windows GPU 测试视为 WSL 验收。

当前代码可执行：

- 上颌、下颌独立训练与续跑，独立导出模型包。
- 可选 `train-dual` 顺序调度两颌，检查跨颌患者划分，不共享 decoder 或 latent。
- 冻结两套模型后进行 pose → shape → joint 多视图图像拟合，以及可选防穿插。

当前已提供训练中冻结 decoder 的验证码拟合、验证打分及 best_val 选模。尚不提供早停、TensorBoard 日志、多病例 batch、多卡 DDP、AMP/混合精度、完整的数据预处理流水线。不要往 JSON 添加旧版 `ScenesPerBatch`、`SamplesPerScene`、`DataLoaderThreads` 等字段期待它们生效。

本文没有启动正式训练。已有原始扫描或旧 `SdfSamples/*.npz + *.pkl` 并不等于已有本版合规训练集；先按第 6 节验证。

### 推荐执行顺序

1. Windows 驱动与 WSL2 → Linux 文件系统中的项目副本。
2. Python/PyTorch → GPU 基础检查。
3. 核验数据 → 单病例真实损失/反传/显存检查 → 单 epoch 冒烟。
4. 固定配置及环境 → 正式单颌训练；需要时启用双颌调度。
5. 导出模型包 → 编译可微渲染依赖 → GPU 渲染检查 → 联合拟合。

第 5 步的渲染编译也可提前完成；DMM 隐式模型训练本身不依赖 nvdiffrast。

## 1. 推荐版本和资源安排

| 项目 | 本手册基线 | 说明 |
| --- | --- | --- |
| 主机系统 | Windows 11 + WSL2 | 驱动安装在 Windows |
| Linux | Ubuntu 24.04 LTS，x86_64 | 使用发行版 Python 和 GCC |
| Python | 3.12，独立 venv | 不复用 Windows Conda/venv |
| PyTorch | 2.13.0，官方 cu130 wheel | 固定可查证版本，非声称最新 |
| torchvision | 0.28.0，同一 cu130 索引 | 内置 torchmeta 的导入链依赖它 |
| CUDA Toolkit | 13.0 系列，仅编译渲染时需要 | 与 `torch.version.cuda == 13.0` 对齐 |
| GCC/G++ | Ubuntu 24.04 的 13 系列 | 不使用 Windows MSVC |
| nvdiffrast | 项目内 0.4.0 源码及 SM 12.x 补丁 | 必须在 Linux 重新编译 |
| WSL 内存 | 先设 96GB，swap 16GB | 给 Windows 留约 32GB；工程建议 |
| GPU | RTX 5090，预期 SM 12.0 | 显存以 `nvidia-smi` 实际报告为准 |

PyTorch 官方列出了 2.13.0 / torchvision 0.28.0 的 cu130 安装组合；NVIDIA 列出 RTX 5090 的计算能力为 12.0。CUDA 13.0 的 Linux 支持表包含 Ubuntu 24.04 与 GCC 13。这里选择这些版本是为了固定部署基线，仍需通过后文的实际计算检查。[PyTorch 安装组合](https://pytorch.org/get-started/previous-versions/)、[NVIDIA GPU 计算能力](https://developer.nvidia.com/cuda/gpus)、[CUDA 13.0 Linux 安装指南](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-installation-guide-linux/index.html)

不承诺训练耗时或显存占用。当前损失含高阶导数与形变 Jacobian；128GB 系统内存不能代替 GPU 显存。先用完整网络、实际数据测量，再定采样规模。

## 2. Windows：安装 WSL2、驱动和内存配置

先安装支持 RTX 5090 的 NVIDIA Windows 驱动，重启。管理员 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --list --verbose
nvidia-smi
```

Ubuntu 对应的 VERSION 应为 2；如果已有其他发行版，不必删除，可新建此环境。按提示重启并创建 Linux 用户。[Microsoft WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)

编辑 Windows 用户目录下 `%UserProfile%\.wslconfig`，与已有配置合并：

```ini
[wsl2]
memory=96GB
swap=16GB
localhostForwarding=true
```

`processors` 暂不指定，避免在未知 CPU 核数下错误限制。训练前确认 Windows 不会自动睡眠；`tmux` 无法抵御主机关机、休眠或 WSL 被关闭。

保存后，在没有运行中任务时执行：

```powershell
wsl --shutdown
wsl -d Ubuntu-24.04
```

`wsl --shutdown` 会停止全部 WSL 发行版中的任务。WSL 的 memory 默认是主机内存的 50%；配置属于整个 WSL2 VM，重启后生效。[Microsoft WSL 配置说明](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

进入 Ubuntu 后执行：

```bash
uname -a
cat /etc/os-release
free -h
df -h "$HOME"
nvidia-smi
# 如果命令未在 PATH，但 Windows 驱动已安装，可尝试：
/usr/lib/wsl/lib/nvidia-smi
```

**不要在 WSL 内安装 NVIDIA Linux 显卡驱动。** WSL 使用 Windows 驱动映射。后面只安装 toolkit；避免 `cuda`、`cuda-drivers`、`nvidia-driver-*` 等会拉入 Linux 驱动的安装方式。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 3. 迁移项目和数据

### 3.1 必须复制本项目的修改版源码

只重新 clone 官方 DMM 会丢失本次所有扩展。复制当前项目的完整源码快照，至少包括：

```text
multiview_teeth_reconstruction/
  third_party/DMM/          # 包含 UPSTREAM.json、dmm、training、data、networks、utils、内置 torchmeta
  third_party/nvdiffrast/   # 包含本地修改的 csrc/common/common.h
  configs/
  scripts/
  tests/
  docs/
  README.md
```

当前改为项目根目录独立 Git 仓库，按 [分轮训练计划](TRAINING_ITERATION_PLAN.md) clone 固定标签。DMM 上游身份读取 `UPSTREAM.json`，项目 commit 另行记录，不再搬运嵌套 `.git`。使用项目提交同步所有修改；不能只 clone 官方上游。

可在源电脑 PowerShell 打包到自行准备的传输盘目录。下面的 `E:\transfer` 必须先存在：

```powershell
tar.exe -czf E:\transfer\dental_multiview_source.tar.gz `
  --exclude=.runtime --exclude=.venv --exclude=runs `
  --exclude=__pycache__ --exclude=build --exclude=dist `
  --exclude=*.pyd --exclude=*.so --exclude=*.egg-info `
  -C D:\WorkSpace\Dental\multiview_teeth_reconstruction .
Get-FileHash E:\transfer\dental_multiview_source.tar.gz -Algorithm SHA256
```

排除的 `runs/` 中如有待继续的训练 checkpoint，应**另行完整复制对应运行目录**；不要把历史实验混进新输出。不要传输 Windows `.runtime/gpu-python`、CUDA 编译器、`.pyd` 或虚拟环境来充当 Linux 环境。

目标 Ubuntu 安装基础工具并解包。以下所有 Bash 命令在 Ubuntu 中运行，不在 PowerShell 中运行：

```bash
sudo apt update
sudo apt install -y git curl ca-certificates build-essential gcc-13 g++-13 \
  python3.12 python3.12-venv python3.12-dev rsync tmux htop unzip

mkdir -p "$HOME/dental/multiview_teeth_reconstruction"
sha256sum /mnt/e/transfer/dental_multiview_source.tar.gz
tar -xzf /mnt/e/transfer/dental_multiview_source.tar.gz \
  -C "$HOME/dental/multiview_teeth_reconstruction"
cd "$HOME/dental/multiview_teeth_reconstruction"
git -C third_party/DMM rev-parse HEAD
git -C third_party/DMM status --short
```

SHA256 应与源电脑一致。Git 有未提交修改是预期现象；不要 `reset --hard` 或重新 checkout 覆盖修改。

### 3.2 数据也放在 Linux 文件系统

建议结构：

```text
~/dental/multiview_teeth_reconstruction/   # 项目与新运行
~/dental/data_v1/upper/                    # manifest.json、canonical.json、samples、centers
~/dental/data_v1/lower/
~/dental/source_geometry/                  # manifest 引用的原始扫描和标注
```

训练读取大量 NPZ，优先用 Linux home 所在的 ext4 虚拟磁盘；`/mnt/d`、`/mnt/e` 适合作为传输入口。Microsoft 建议 Linux 工具工作负载将文件保存在 Linux 文件系统。[文件系统建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

所有 manifest 引用均为**相对其所属 JSON 的路径 + 文件 SHA256**。必须连同 `source_geometry`、`source_annotation`、centers、samples、canonical 引用一起迁移；加载器也会核验原始源文件。保持相对目录结构，避免出现 `D:\...` 或反斜杠路径。若确需重排目录，在数据副本中重建引用后重新校验，不编辑冻结原始数据。

不要估算一个固定磁盘容量后盲目复制：先用 `du -sh` / `df -h` 统计数据、源码、环境和 checkpoint。checkpoint 含模型、embedding 和 Adam 状态，运行后再实测单个文件大小，乘以计划保留数量。

## 4. Python 与 GPU 运行环境

在项目根目录：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy scipy pillow scikit-image ordered-set plyfile trimesh ninja
python -m pip check

export PYTHONPATH="$PWD/third_party/DMM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

NumPy 等辅助包由此次安装解析，安装成功并通过检查后用第 5 节的 `pip freeze` 固定实际版本。不要直接安装 PyPI 的 torchmeta，项目使用自带版本；也不要在这个 venv 中混装 Conda CUDA 或随意升级 torch/torchvision。匹配的 torchvision 对原生 DMM 导入是必要的，即使训练没有使用视觉 backbone。

保存一个每次开新终端可加载的环境文件（首次创建；已有文件先检查）：

```bash
test ! -e .venv/dmm_env.sh
cat > .venv/dmm_env.sh <<'BASH'
# 从项目根目录 source .venv/dmm_env.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/third_party/DMM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
BASH
source .venv/dmm_env.sh
```

### GPU 基础检查

```bash
python - <<'PY'
import torch, torchvision
from networks.dmm_net import DMM
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('runtime CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('compiled architectures:', torch.cuda.get_arch_list())
assert torch.cuda.get_device_capability(0) == (12, 0), 'check the actual target GPU'
assert torch.version.cuda == '13.0', 'this guide expects cu130'
x = torch.randn(512, 512, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('GPU_BASIC_FORWARD_BACKWARD_OK')
PY
python -B third_party/DMM/dmm_cli.py --help
```

`nvidia-smi` 的 CUDA Version 是驱动支持能力，`torch.version.cuda` 是 wheel 使用的版本，`nvcc --version` 是独立 toolkit 编译器版本；三者含义不同。只有 `torch.cuda.is_available()` 不足以确认训练能运行，后面还要运行原生损失的高阶反传。

## 5. 保存环境和源码身份

在本次环境配置完成后创建新的记录目录，后续增加渲染依赖时另存新记录，不覆盖旧锁定文件：

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
AUDIT="runs/environment_$STAMP"
mkdir -p "$AUDIT"
python -m pip freeze > "$AUDIT/pip-freeze.txt"
python -m pip inspect > "$AUDIT/pip-inspect.json"
python -m torch.utils.collect_env > "$AUDIT/torch-environment.txt"
nvidia-smi -q > "$AUDIT/nvidia-smi.txt"
uname -a > "$AUDIT/kernel.txt"
git -C third_party/DMM rev-parse HEAD > "$AUDIT/dmm-upstream.txt"
git -C third_party/DMM status --short > "$AUDIT/dmm-status.txt"
python - <<'PY' > "$AUDIT/dmm-source-sha256.txt"
from dmm.provenance import source_fingerprint
print(source_fingerprint())
PY
```

本文编写时 native DMM 源码指纹为 `3e8ef821758358cd054ffcb65f801829380e61ba49f408c61dffa5d5d15b4af4`。无后续源码改动时，迁移后应相同。该指纹按文本归一化读取源码；训练数据 JSON/NPZ 等引用哈希则按原始文件字节计算。

保存**实际源码快照和未跟踪文件**，仅保存上游 commit 不足以重现。训练、checkpoint 导出、bundle 加载、拟合恢复都包含严格的源码绑定。环境升级要另开环境并验证，不能仅改包版本后宣称实验连续一致。

## 6. 训练数据与配置

### 6.1 本版数据要求

详见 [参数与数据约定](dual_arch_parameter_data_contract_v1.md) 和 [接口格式](dual_arch_interfaces.md)。关键要求：

- 两颌各有自己的 manifest、canonical 参考和 specs；无需成对扫描，但同患者跨颌不能跨 train/val/test。
- 每颌固定 14 个牙位，牙龈为 0；显式 presence 表示实际缺牙。
- 单位转换、LPS 方向、canonical 刚体对齐必须在预处理时完成；模型坐标为毫米除以固定 50，不做病例独立缩放。
- NPZ 精确包含 `surface_points`、`surface_normals`、`surface_labels`、`offsurface_points`；不接受旧格式的 `surf/pos/neg/normals` 直接替换。
- 所有原始、镜像与同患者病例保持同一划分；canonical 参考只来自原始 train 病例。
- 用于后续 prior 导出时，每个组件至少有 `latent_dim + 1` 个原始 train 病例实际包含它，且 latent 方差非零；镜像不计入这个数量。

`make-config` 不会制作数据、推断单位或自动转换旧模型。它会加载并校验已有数据，所以数据不合规时应先修正预处理。

### 6.2 创建两份独立配置

以下示例假定第 3 节的目录布局已经成立：

```bash
mkdir -p configs/workstation
python -B third_party/DMM/dmm_cli.py make-config \
  --arch upper --manifest ../data_v1/upper/manifest.json \
  --canonical-reference ../data_v1/upper/canonical.json \
  --specs third_party/DMM/examples/dual_arch/upper_specs.json \
  --epochs 100 --output configs/workstation/upper_train.json
python -B third_party/DMM/dmm_cli.py make-config \
  --arch lower --manifest ../data_v1/lower/manifest.json \
  --canonical-reference ../data_v1/lower/canonical.json \
  --specs third_party/DMM/examples/dual_arch/lower_specs.json \
  --epochs 100 --output configs/workstation/lower_train.json

python -B third_party/DMM/dmm_cli.py validate-training \
  --config configs/workstation/upper_train.json \
  --other-config configs/workstation/lower_train.json
```

只训练一颌时，运行该颌的 make-config，并省略 `--other-config`。两份清单都可用时务必做跨颌检查。

### 6.3 RTX 5090 起始参数

| 参数 | 首轮采用 | 如何理解 |
| --- | ---: | --- |
| epochs | 100 | 预算初值，不是已证实足够的收敛轮数 |
| points_per_component | 256 | 每个存在牙位及牙龈的表面点数 |
| offsurface_points | 2048 | 单病例共享的非表面点数 |
| learning_rate | 0.0001 | Adam，当前无调度器 |
| seed | 42 | 固定随机种子 |
| checkpoint_every | 10 | 建议首轮长训前改为 1，方便恢复；实测大小后再调整 |

完整 14 牙加牙龈的病例每次查询点数为 `15 × 256 + 2048 = 5888`。训练器逐病例优化，各组件会在这些查询点上计算损失；不能把 5888 当成总计算量或显存的线性保证。

首轮保留示例网络结构、latent 维数和损失系数。旧版经验中的 `16384 点 × batch 2` **不适用于当前入口**。本版逐病例读取，不缓存整个数据集，没有可调的 worker/batch 参数。当前 full specs 的目标机显存需要实测。

若 OOM，可在**新实验配置**中将配额降到 `128/1024`；有足够余量后再单独试 `512/4096`。改变采样量会改变训练过程，不能加载旧 optimizer 当成完全一致的续跑。不要通过增加 swap 处理显存 OOM，也不要未验证就加入 AMP。

编辑训练配置应在生成双颌任务配置之前完成。修改 specs/manifest/canonical 内容后，必须重新生成对应引用 hash；通常新建配置重新 `make-config` 最清楚。

## 7. 正式训练前：真实单病例 GPU 检查

这个步骤使用目标数据中存在组件最多的一个 train 病例及完整 specs，实际运行损失、反传、Adam step，并检查梯度和参数。只在内存内更新一次，不写训练 checkpoint，不修改数据。

```bash
python - configs/workstation/upper_train.json <<'PY'
import json, sys, time, torch
from networks.dmm_net import DMM
from training.arch_training import load_training_config, ArchTrainingSystem

config, dataset, centers, specs = load_training_config(sys.argv[1])
torch.manual_seed(config['seed'])
system = ArchTrainingSystem(DMM(specs, arch=config['arch']), dataset, centers).cuda()
system.train()
optimizer = torch.optim.Adam(system.parameters(), lr=config['learning_rate'])
index = max(range(len(dataset)), key=lambda i: len(dataset.selected[i].components))
sample = dataset[index]
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
optimizer.zero_grad(set_to_none=True)
loss, terms = system.loss(sample)
assert torch.isfinite(loss), 'nonfinite loss'
loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in system.parameters())
optimizer.step()
assert all(torch.isfinite(p).all() for p in system.parameters())
torch.cuda.synchronize()
print(json.dumps(dict(status='ONE_CASE_GPU_STEP_OK', arch=config['arch'],
    case_id=sample['case_id'], components=len(sample['components']), terms=terms,
    seconds=time.perf_counter()-started,
    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3), indent=2))
PY
```

同样检查 lower，将第一行的配置路径换成 `lower_train.json`。将输出保存到独立日志，记录同时运行的其他 GPU 程序。一个病例通过仍不能证明整个数据集可训练，因此继续下一步。

### 单 epoch 冒烟

在相同目录创建只改变 epochs 和保存频率的配置；这样其相对文件引用保持有效：

```bash
python - <<'PY'
import json
from pathlib import Path
for arch in ('upper', 'lower'):
    source = Path(f'configs/workstation/{arch}_train.json')
    target = source.with_name(f'{arch}_smoke.json')
    assert not target.exists(), target
    config = json.loads(source.read_text())
    config.update(epochs=1, checkpoint_every=1)
    target.write_text(json.dumps(config, indent=2) + '\n')
PY

python -u -B third_party/DMM/dmm_cli.py train \
  --config configs/workstation/upper_smoke.json --device cuda:0 \
  --output runs/workstation_upper_smoke_001
python -u -B third_party/DMM/dmm_cli.py train \
  --config configs/workstation/lower_smoke.json --device cuda:0 \
  --output runs/workstation_lower_smoke_001
```

这会各自遍历全部 train 病例一轮，不是“一批”快速测试。先看单病例耗时再决定什么时候执行。需要确认运行完成、loss/gradient/parameter 均有限、final checkpoint 可加载；不能只看到进程启动就算通过。

最小 checkpoint 核查：

```bash
python - <<'PY'
import torch
from networks.dmm_net import DMM
from training.arch_training import load_training_config, ArchTrainingSystem
from dmm.provenance import source_fingerprint
for arch in ('upper', 'lower'):
    cfg, data, centers, specs = load_training_config(f'configs/workstation/{arch}_smoke.json')
    saved = torch.load(f'runs/workstation_{arch}_smoke_001/final.pth', map_location='cpu', weights_only=True)
    assert saved['arch'] == arch and saved['epoch'] == 1
    assert saved['source_sha256'] == source_fingerprint()
    system = ArchTrainingSystem(DMM(specs, arch=arch), data, centers)
    system.load_state_dict(saved['system_state'], strict=True)
    assert all(torch.isfinite(v).all() for v in saved['system_state'].values())
    print(arch, 'CHECKPOINT_LOADED', saved['best_train_loss'])
PY
```

## 8. 正式训练与运行管理

### 8.1 单颌训练

首轮建议按顺序运行上颌和下颌；不要在同一 GPU 上同时启动两份完整训练争抢显存。

```bash
tmux new -s dmm-upper
```

在 tmux 会话内回到项目根目录，加载环境并执行：

```bash
cd "$HOME/dental/multiview_teeth_reconstruction"
source .venv/dmm_env.sh
mkdir -p logs
set -o pipefail
python -u -B third_party/DMM/dmm_cli.py train \
  --config configs/workstation/upper_train.json \
  --output runs/workstation_upper_train_001 --device cuda:0 \
  2>&1 | tee logs/workstation_upper_train_001.log
```

`Ctrl+B` 后按 `D` 脱离；返回用 `tmux attach -t dmm-upper`。下颌将配置、会话、输出目录、日志名都换成 lower。所有输出目录必须全新；重复执行应换新的 run 名，不删除旧结果。

### 8.2 可选：统一双颌调度

两份单颌配置已固定且都通过冒烟后：

```bash
python -B third_party/DMM/dmm_cli.py make-dual-config \
  --upper-config configs/workstation/upper_train.json \
  --lower-config configs/workstation/lower_train.json \
  --output configs/workstation/dual_train.json
python -B third_party/DMM/dmm_cli.py validate-dual-training \
  --config configs/workstation/dual_train.json

# 在 tmux 内执行；本条命令会真正开始训练。
set -o pipefail
python -u -B third_party/DMM/dmm_cli.py train-dual \
  --config configs/workstation/dual_train.json \
  --output runs/workstation_dual_train_001 --device cuda:0 \
  2>&1 | tee logs/workstation_dual_train_001.log
```

该入口顺序训练 upper → lower，各自独立输出，不是一个跨颌联合训练网络。拟合时的防穿插不会自动用于训练。详见 [可选扩展](dual_arch_optional_extension_v1.md)。

### 8.3 看什么日志

```bash
# 另一终端；WSL 的 nvidia-smi 某些统计可能不完整
watch -n 2 nvidia-smi
htop
tail -f runs/workstation_upper_train_001/progress.jsonl
```

原生训练每个 epoch 完成后记录 `mean_train_loss`、`best_train_loss`、`mean_terms`、`term_case_counts`、三组梯度/学习率及可用的 `best_val_score`。一整轮内没有新的 progress 行不代表卡死。`mean_terms` 按该项实际参与病例数计算，另保留有效点计数。

本训练路径**不写 TensorBoard events**；安装 TensorBoard 后也不会自动出现曲线。当前以 JSONL 和终端日志为准。

| 输出 | 含义 |
| --- | --- |
| `config.json` / `provenance.json` | 配置、数据与源码身份、恢复来源 |
| `progress.jsonl` | 每轮训练损失 |
| `epoch_000010.pth` 等 | 周期 checkpoint |
| `best_train.pth` | 最低训练损失，不是最佳验证集模型 |
| `final.pth` | 计划最后一轮 |
| `report.json` | 训练流程完成，不代表质量验收 |
| 双颌目录中的 `workflow.json` | 两颌任务状态和已完成结果引用 |

训练已有验证集自动评估，完整选定 val 集合成功并改善时保存 `best_val.pth`；失败病例不剔除算平均。不要单凭训练损失选择模型。正式质量评估仍应检查表面、语义、未观察区域和多视图拟合结果。

## 9. 中断恢复与延长训练

### 9.1 单颌恢复

保持原配置、数据、网络和源码一致，从已经保存的完整 checkpoint 恢复到新目录：

```bash
python -u -B third_party/DMM/dmm_cli.py train \
  --config configs/workstation/upper_train.json --device cuda:0 \
  --resume runs/workstation_upper_train_001/epoch_000010.pth \
  --output runs/workstation_upper_train_002
```

恢复粒度是完整 epoch，不含上次中断那一轮的部分进度。周期设为 10 时，可能回退多轮；需要更细的恢复间隔，应在正式开始前采用 `checkpoint_every=1`。

单颌签名允许改变 `epochs` 和 `checkpoint_every`，其他配置变化会拒绝续跑。延长训练时，复制 JSON 到**同一目录的新文件**，只增大 epochs（如 100 → 200），然后从 final 恢复；输出总 epoch 数是 200，不是再加 200。不要改原始已记录的配置文件。

### 9.2 双颌任务恢复

```bash
python -u -B third_party/DMM/dmm_cli.py train-dual \
  --config configs/workstation/dual_train.json --device cuda:0 \
  --resume runs/workstation_dual_train_001/workflow.json \
  --lower-resume runs/workstation_dual_train_001/lower/epoch_000010.pth \
  --output runs/workstation_dual_train_002
```

示例假定上颌已完成、下颌中断。已完成上颌只验证并引用其文件，不重训；未提供 `--lower-resume` 时，未完成的下颌从头训练。第一次上颌就中断则用 `--upper-resume`。

双颌 workflow 绑定整个任务配置、子配置 hash、设备字符串和源码。不要在恢复时把 `cuda:0` 改成 `cuda`，也不要改双颌任务的子配置。需要延长已完成任务时，分别用单颌入口及新配置恢复；已完成 workflow 不能作为未完成任务重启。

断电发生在一颌完成但 workflow 尚未写入的瞬间时，应先核对该颌 `final.pth`、`report.json`；不要编辑 workflow 伪造完成状态，可直接独立运行剩余颌并保留来源记录。

## 10. 导出模型包

两个独立 run 的示例：

```bash
python -B third_party/DMM/dmm_cli.py export-bundle \
  --checkpoint runs/workstation_upper_train_001/final.pth \
  --output runs/scene_001/model_bundle/upper --model-id upper_workstation_v1
python -B third_party/DMM/dmm_cli.py export-bundle \
  --checkpoint runs/workstation_lower_train_001/final.pth \
  --output runs/scene_001/model_bundle/lower --model-id lower_workstation_v1
```

双颌调度的 checkpoint 路径换成 `<dual-run>/upper/final.pth` 和 `<dual-run>/lower/final.pth`；已完成颌可能引用旧 run，以 workflow 为准。

每颌输出 `model.json`、`weights.pth`、`latent_statistics.npz`，后者仅用原始、存在该组件的 train 病例 latent 统计。若报样本数量不足/零方差，不要通过随机码或镜像充数，也不要跳过导出校验。

这一步不自动生成相机、逐牙掩码、初始位姿或场景 manifest。场景需按 [双颌接口](dual_arch_interfaces.md) 的 `fit_input/`、`initialization/` 结构制作。

## 11. 可选：在 Linux 编译渲染后端并接入拟合

### 11.1 Toolkit 13.0 与 nvdiffrast

仅在需要渲染/图像拟合及其验证时执行。WSL-Ubuntu 官方仓库提供 toolkit-only 包：[NVIDIA 包索引](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/)。

```bash
cd "$HOME/dental/multiview_teeth_reconstruction"
source .venv/dmm_env.sh
mkdir -p .runtime/downloads
curl -fL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
  -o .runtime/downloads/cuda-keyring_1.1-1_all.deb
sudo dpkg -i .runtime/downloads/cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-13
export CXX=g++-13
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
nvcc --version
g++-13 --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"

grep -n '__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 1200' \
  third_party/nvdiffrast/csrc/common/common.h
python -m pip install --no-build-isolation --no-deps ./third_party/nvdiffrast
python -c "import nvdiffrast.torch, _nvdiffrast_c; print(_nvdiffrast_c.__file__)"
```

`grep` 应找到项目保留的 SM 12.x 条件。项目在 Windows SM 12.0 上发现并修复了共享原子梯度累加偏差，Linux 也保留这个源码补丁，但仍需目标机实测。不要改回未修补的在线仓库版本，不要复制 Windows 编译出的 `.pyd`。本版本 setup.py 使用 PyTorch CUDAExtension，并要求 `--no-build-isolation`；上游文档可作背景参考，实际构建以项目内源码为准。[nvdiffrast 官方文档](https://nvlabs.github.io/nvdiffrast/)

不设置 Windows `.runtime` 到 `PYTHONPATH`，不 source `scripts/gpu_env.ps1`。这里使用 CUDA 光栅化，不需要为了本项目配置 OpenGL 窗口或安装 FLAME、pointops、Blender。

如新终端还要重新编译，重新导出上述 CUDA_HOME、PATH、CC/CXX 和架构变量。记录 `nvcc --version`、`dpkg-query -W 'cuda*'`、包锁与编译日志。不要用 `--allow-unsupported-compiler` 跳过编译器兼容检查。

### 11.2 运行现有 GPU 验证

```bash
python -B scripts/check_gpu_render.py
python -B scripts/check_dual_extension.py
```

两脚本创建新 `runs/` 目录。当前源码预期渲染检查 14 项通过，扩展回归 50 项通过、无跳过；以目标机日志为准。扩展脚本包含两个临时小数据单 epoch 训练测试和未训练几何夹具拟合，不会训练实际牙齿数据。

要求核对 JSON、退出码和 PNG，不只看“无异常导入”。Windows 已知的**细密网格整图 AA 严格有限差分未通过**限制仍存在；局部导数门槛通过不等于完整重建或临床验收。源代码内的启用防穿插诊断也可能保留 `COLLISION_GATE_NOT_MET`，不能将其改名为成功。

### 11.3 联合拟合命令

准备好完整 scene 后：

```bash
python -B third_party/DMM/dmm_cli.py validate-scene \
  runs/scene_001/fit_input/manifest.json --device cuda:0

python -u -B third_party/DMM/dmm_cli.py fit-scene \
  runs/scene_001/fit_input/manifest.json --device cuda:0 \
  --fit-config configs/fit_static_v1.json \
  --surface-config configs/surface_static_v1.json \
  --render-config configs/render_nvdiffrast_v1.json \
  --matching-config configs/semanticxy_static_v1.json \
  --output runs/workstation_fit_001
```

需要可选防穿插时追加 `--collision-config configs/collision_optional_v1.json`；不传即关闭。这里冻结 decoder，只优化场景参数，不会继续训练 DMM 权重。相机固定、所有有效视图共同参与，静态开口不要求上下牙接触。

渲染配置默认 4×4 超采样，`max_pixels` 统计超采样后的像素数；例如 512×512 图像需要 4,194,304 个内部像素，超过当前 1,048,576 默认上限。先检查图像尺寸和显存，再在新配置中明确调整预算；不能把缩放图片后不改 K 当成等价输入。

## 12. 故障定位表

| 现象 | 优先检查 |
| --- | --- |
| WSL 看不到 GPU | Windows 驱动、WSL2 而非 WSL1、`wsl --update`；不要安装 Linux 显卡驱动 |
| `no kernel image` / `sm_120` 不支持 | 是否误装 CPU/旧 CUDA wheel，是否真在 Linux venv；运行实际矩阵与模型反传检查 |
| torchvision 导入报 operator 缺失 | torch/torchvision 是否同一 cu130 来源和匹配版本 |
| `CUDA_HOME` / nvcc 找不到 | 仅渲染编译需要 toolkit；检查 `/usr/local/cuda-13.0` 和激活环境 |
| 编译器或 CUDA 版本不匹配 | torch 13.0、nvcc 13.0、GCC 13 是否一致；检查 PATH 中旧 toolkit |
| 抗锯齿梯度检查失败 | SM 12.x 补丁是否搬运，导入的是否本次 Linux `.so`，保存诊断而非放宽阈值 |
| 训练 CUDA OOM | 当前 GPU 是否被其他进程占用；用新配置降低点配额并重新测量，不改 worker 字段 |
| WSL 被 OOM kill / `Cannot allocate memory` | `free -h`、`dmesg`、`.wslconfig`、磁盘剩余；系统内存与显存分别排查 |
| 文件 SHA256 不匹配 | 文件传输完整性、JSON 换行/编码变化、迁移后相对路径是否仍对应正确文件 |
| `not a git repository` | 是否 clone 了本项目并进入项目根目录；DMM 的 donor 身份由 `UPSTREAM.json` 提供 |
| canonical、presence、单位校验失败 | 修正数据生成流程，不从采样或照片猜 presence，也不放宽加载器 |
| source/config mismatch | 是否换过源码、数据、采样/学习率、模型统计或设备标识；使用对应快照恢复 |
| 长时间无 progress 行 | 当前实现按 epoch 写日志，先看 GPU 利用率和单病例耗时 |
| 没有 TensorBoard 曲线 | 当前原生入口没有 event writer，读取 progress.jsonl |
| 导出 prior 失败 | 每组件原始 train 数量是否 ≥ d+1、是否实际存在、latent 是否有有效方差 |

## 13. 目标机交接记录

开始长训前记录：Windows 驱动、WSL/Ubuntu 版本、GPU 型号和显存、pip 锁、源码指纹、两颌数据 manifest hash、配置、单病例峰值显存/耗时、单 epoch 冒烟结果和 checkpoint 核查结果。任一步失败先保留日志。

训练结束记录：实际完成 epoch、所选 checkpoint 及其哈希、原始训练损失、模型包、训练/验证/测试划分、后续几何与图像评估。过程完成、接口通过、图像拟合和模型质量是不同状态。

相关项目文档：[双颌数据接口](dual_arch_interfaces.md)、[GPU 与分阶段拟合](staged_fitting_gpu_v1.md)、[可选双颌扩展](dual_arch_optional_extension_v1.md)。
