# Teeth3DS → DMM 可搬运预处理工程

把本文件夹或同名 ZIP 拷贝到另一台电脑，利用那台电脑已有的**未经处理的 Teeth3DS**，复现 2026-10-17 这版固定参考相似对齐数据。运行代码、旧版数值依赖、病例清单、参考和验收基准都在本工程内。无需原电脑的工作区、归档工程、旧模型、GPU 或 CUDA。

另一台电脑的 AI Agent 请先阅读 **[AGENTS.md](AGENTS.md)**。

## 快速使用

### 1. 选择 CPU 预处理环境

原处理环境是 Python **3.12.14**、NumPy **2.5.2**、SciPy **1.18.0**。优先使用 Python 3.12 并安装 `requirements.txt`；尽量保持相同补丁版本。可在 Windows、Linux 或 WSL 运行。GPU 训练环境和这个 CPU 预处理环境可以分别保留。

例如已有 Conda 时：

```text
conda create -n teeth3ds-preprocess python=3.12 pip
conda activate teeth3ds-preprocess
python -m pip install -r requirements.txt
```

也可以使用现有匹配环境或在本项目创建 `.venv`。已经存在的环境不要覆盖。网络下载中断时保留已完成的依赖缓存。

最低可执行 Python 版本为 3.10；使用其他 NumPy/SciPy 版本时，`sample` 会记录版本并进行同样的严格内容核对。**替代环境先通过 sample**，然后 batch 显式加 `--allow-environment-mismatch`；这个参数只允许环境不同，不能绕过数据内容校验。若 sample 的数组或 QC 不一致，优先恢复参考环境。

### 2. 指定原始数据根目录并抽查

标准目录结构是：

```text
Teeth3DS/
  upper/<patient>/<patient>_upper.obj
  upper/<patient>/<patient>_upper.json
  lower/<patient>/<patient>_lower.obj
  lower/<patient>/<patient>_lower.json
```

在本工程根目录运行，路径换成那台电脑的实际目录：

```text
python -B run.py sample --raw-root "D:/datasets/Teeth3DS" --workers 2
```

Linux/WSL 例子：

```bash
python -B run.py sample --raw-root /data/Teeth3DS --workers 2
```

目录层次不同时加 `--locate-files`，只定位清单中已有文件名，**不会追加病例**。重复同名文件会报错，必须明确正确的数据目录。文件内容由原始 OBJ 和标注 JSON 的 SHA256 核对。

`sample` 只处理固定的 **6 个原例 / 8 组 NPZ+PKL**，含上下颌参考、原例/训练镜像、val/test、缺牙、REVIEW、HOLD。参考病例自动在抽查范围内，不会触发全量。

成功会打印 `PASS_REPRODUCED_DATA_CONTENT`，随后自动删除本次 `.work/sample_*` 临时输出和几何缓存。失败在终端打印原因并保留该次工作目录，便于定位；需要保留成功样本调试时才加 `--keep-work`。所有命令均不生成报告、审查图片或 HTML。

### 3. 在训练电脑批量生成

```text
python -B run.py batch --raw-root "D:/datasets/Teeth3DS" --work-dir "D:/datasets/Teeth3DS_DMM_Reproduced" --workers 6
```

```bash
python -B run.py batch --raw-root /data/Teeth3DS --work-dir /data/Teeth3DS_DMM_Reproduced --workers 6
```

工作目录 `--work-dir` 可以省略，默认使用本工程目录。输出都在 `<work-dir>/output/latest`；建议为完整数据预留 **至少 100 GB** 可用空间，空间占用取决于原网格大小。不要把工作目录放进原始 Teeth3DS 内。内存紧张时用 `--workers 1` 或 `2`，随机种子不随并发数改变。

同路径再次运行会复用这个工作目录中身份匹配的**最新缓存**。每例仅一份几何/采样缓存，没有逐轮完整副本或阶段冻结。已有输出必须属于本工程、同一原始路径和同一病例范围；不同范围请用新的工作目录，防止抽查结果覆盖整库。完整 batch 会自动执行独立几何核验、全部导出内容基准比较、划分和平均质心检查。

可单独重新验证已完成的完整输出：

```text
python -B run.py verify --work-dir "D:/datasets/Teeth3DS_DMM_Reproduced"
```

如果被强制中断留下 `output/latest/running.lock`，先核查记录 PID 对应进程确实不再运行，再处理这个本工程的遗留锁；不要并发写同一工作目录。

## 复现目标与固定规则

- 1,793 个有标注候选：upper 899，lower 894。原始 7 个具体颌扫描及增强血缘排除；配对的另一颌仍保留。100 份缺标注扫描未列入本轮候选。
- 完整基准：**PASS 755、REVIEW 916、HOLD 122；原例导出 1,671、含训练镜像共 3,001 组 NPZ+PKL**。
- upper 导出 1,513，lower 1,488；完整划分清单和计算 PASS 子集均随包携带。HOLD 保留说明，不生成训练监督；REVIEW 按当前逻辑导出，不扩大永久排除。
- 固定参考 upper `01328DDN_upper`、lower `01JNZKAX_lower`，参考全网格最大半边长 0.95；逐例 `q=sRx+t`。所有结果保留逐例正逆矩阵，没有统一的 50 mm/model-unit。
- 原始顶点顺序不变；只在派生缓存中过滤退化面和重复面。strict 面面积质心用于对齐；法线来自实际观察面叉积。
- 每个存在牙号 25,000 表面点，牙龈 100,000 点；每例 500,000 规范立方体均匀空间点。按原种子和 NumPy 算法采样。
- 仅 train 镜像：反射、FDI 对换、重新对齐同一参考；空间点在镜像最终规范坐标中独立生成。
- `pos/neg` 各 250,000，是 DMM 非表面兼容槽，第四列 0 占位、牙号 -1；不是真实内外/SDF标签。
- val/test 不参与参考或平均质心拟合。平均质心只由计算 PASS 的 train（含镜像）生成，不能用 sample 的均值替代完整 batch 均值。
- 时间、绝对路径和压缩容器字节不是数值复现的主判据。数据内容保持严格校验。

## 验收方法和边界

代码入口先检查 `PACKAGE_MANIFEST.json` 中源码、配置和基准的 SHA256。独立几何核验读取全部当前输出数组，并检查面索引/重心采样回放、法线、牙号、配额、空间域、均值、原始输入 SHA 和患者划分。

程序内部进一步核对既有病例/QC/镜像清单、参考和逐例变换（数值容差 1e-12）、划分及完整 batch 的平均质心。所有 3,001 个基准 NPZ 只携带原有文件 SHA 和 NPY 成员 CRC32/原始长度；CRC32 用于非恶意的意外数据差异检测，**不是密码学身份保证**。NPZ 成员的实际 CRC 会在独立数组读取时校验。

抽查的 8 组导出还携带 **32 个完整数组 SHA256**（包含 dtype/shape）和每数组 257 个数值探针；PKL 的全部中心 float32 值逐项核对。压缩库差异导致容器 SHA 不同时，终端分别展示容器匹配数和内容核验结果；数组 CRC、完整抽查数组 SHA、QC 或数据规则不一致都会报错，不会自动放宽。

抽查通过只证明这些样本和工程执行链路可复现，不能冒称已在另一台电脑全库复现；那台电脑需完成 batch 的全部内容核验。

实测边界：参考 Windows 环境的 8 组 NPZ/PKL 文件 SHA、32 个完整数组 SHA 全部一致。另测本机旧 WSL 环境（Python 3.10.12 / NumPy 2.2.6 / SciPy 1.15.3 / Pillow 11.3.0），31/32 数组一致，剩余一个表面数组仅 1 个坐标相差约 1.82e-12；面采样蓝图一致，变换矩阵差异约 1.78e-15。**严格校验在该旧环境正确失败，未改基准或放宽通过条件。** 其他操作系统/CPU即使使用参考版本也须先运行 sample；本包不承诺未经验证的跨平台逐位一致。

## 输出和训练衔接

```text
<work-dir>/
  .inputs/config.json + candidates.json
  output/latest/
    .portable_owner.json / inputs.json / run_config.json
    references.json / case_results.json
    cache/<case>/geometry.npz + geometry.json + sampling.npz + processed.json
    data/<upper|lower>/SdfSamples/<case>.npz + <case>.pkl
    data/<upper|lower>/manifest.json + specs.json + avg_centroids.txt
    data/<upper|lower>/splits/*.json
```

独立验证后的 `data/`、`inputs.json`、`references.json` 等共同组成 `dmm_fixed_similarity/2` 数据根。训练端使用支持该契约的 DMM 加载器（当前项目的 `FixedSimilarityDataset` / `train-latest` 适配），指向 `<work-dir>/output/latest`。旧的固定 50 mm 约定和旧 embedding checkpoint 不能直接套用。

本工程只生成和核验数据，不生成报告、图片或 HTML，也不启动训练。运行进度和校验结论只打印到终端；case_results.json、inputs.json、参考、变换和最新缓存元数据用于身份校验、增量恢复与数据加载，予以保留。PASS 是计算一致性和规则检查结果，人工视觉/解剖验收仍待完成，`accepted_for_training` 保留 false。
