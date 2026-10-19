# 给另一台电脑 AI Agent 的执行说明

用户已经授权：使用本机已有原始 Teeth3DS，运行本独立工程的小样本复现检查及完整批量预处理，尽量一比一复现随包基准，最终交付数据集路径；用户不需要报告。项目来源电脑的数据已完成处理，本包负责在新电脑重新生成。先读 README.md；不用寻找原电脑的其他项目或聊天记录，也不用等待逐阶段批准。

## 先确定输入和环境

1. 定位本机未经处理的 Teeth3DS 根目录，原件只读。标准结构为 `upper|lower/<patient>/<patient>_<arch>.obj|json`。路径不同可用 `--locate-files`，它只重定位既定文件名，不增加样本。OBJ 和标注 SHA 必须匹配基准；不以网格外形相似代替原件身份。
2. 本工程只需要 CPU。原处理环境：Python 3.12.14 / NumPy 2.5.2 / SciPy 1.18.0。优先复用匹配环境，或创建隔离预处理环境并安装 requirements.txt；不要替换已有训练环境，不安装 CUDA/PyTorch 来做预处理。
3. `run.py` 自动检查源码、配置和基准完整性。已有其他 Python >=3.10 环境时可以先直接运行 sample；若完整样本内容检查通过，可明确记录该替代运行时，再用 batch 的 `--allow-environment-mismatch`。这个参数不允许任何内容差异。

已知实测边界：本机旧 WSL 的 Python 3.10.12 / NumPy 2.2.6 / SciPy 1.15.3 / Pillow 11.3.0 中，抽查有一个表面坐标相差 1.82e-12，严格核对正确失败。不要默认 DMM 的 Python 3.10 训练环境也能逐位复现原预处理；优先单独使用参考预处理环境。遇到同类数值差异，根据终端诊断区分完全一致与近似一致，不以数值探针为 0 代替完整数组哈希匹配，不擅自添加坐标补丁或放宽验收。

## 实际执行

在本文件所在目录执行，路径换成本机路径：

```text
python -B run.py sample --raw-root <原始Teeth3DS根目录> --workers 2
python -B run.py batch --raw-root <原始Teeth3DS根目录> --work-dir <独立生成目录> --workers 6
```

小样本固定 6 原例：`01328DDN_upper`、`01JNZKAX_lower`、`09HRNJZ8_lower`、`01343APK_upper`、`014X0F4K_lower`、`013NUWYR_lower`。其中 5 原例输出、3 例为 train 并有镜像，共 8 组导出；最后一例是 HOLD。sample 默认在成功后自动清理本次临时工作目录，结果只打印到终端，不落盘报告。

检查 sample 是否为 `PASS_REPRODUCED_DATA_CONTENT`，32 个完整数组哈希是否匹配、数值探针误差是否为 0、QC/参考/镜像/held-out划分是否一致，再在这台**训练电脑**运行完整 batch。完整输出约数十 GB，工作区预留至少 100 GB 可用空间；内存不足时降低 workers。批处理无阶段冻结和人为批准关卡，正常继续到自动核验完成即可。

同一 batch 命令可恢复身份匹配的最新缓存。不要把 sample 的 output/latest 移入完整 batch 工作目录；sample 的平均质心是子集计算值。不同原始根或不同包版本请用新工作目录。锁文件须结合 PID 的实际进程身份判断，不能盲删运行中的锁。

## 固定的数据边界

- 永久排除仅这 7 个扫描及增强：00OMSZGW_lower、5J9O0LG3_lower、6L4N5MA2_lower、NQJYZS60_lower、Q5GSR6N2_upper、THGBYHX3_lower、X3EW8O0A_lower。配对另一颌保留。
- 只用 assets/candidates.json 中 1,793 候选，沿用 patient 级 train/val/test。原来 100 份缺标注样本不加入。
- 保留 previous_status / previous_reason 中历史信息，复用现行 QC 规则；REVIEW/HOLD 不自动变成新的永久排除。
- 使用固定参考、相似尺度、25k/牙+100k/龈表面点、500k 均匀空间点，以及 train 镜像重对齐规则。不要“优化”对齐、补洞、修网格、改采样、改法线或改阈值来追求通过。
- val/test 不用于修改参考、参数或诊断阈值。
- 只修改本机路径和 workers 等执行配置；数值代码、基准和样本身份需要保持一致。遇到原件 SHA 不符、样本缺失、重复文件、内容校验失败等实际问题，保留证据并向用户明确报告，不悄悄改基准或扩大排除。

## 完成后交付数据集

完整基准应为：1,793 原例；PASS 755、REVIEW 916、HOLD 122；1,671 原例导出、含镜像共 3,001 组；上颌 1,513、下颌 1,488。具体 train/val/test 与计算 PASS 子集见 assets/tables.json，所有病例结果应匹配 assets/baseline_cases.json。

交付完整数据根 `<work-dir>/output/latest`，简短告知用户路径及校验是否通过即可。不要另写报告，不生成审查图片、HTML 或运行报告 JSON；保留训练清单、参考/变换、输入身份与必要的最新缓存元数据。进度和异常只打印到终端。

完整数组 SHA256 基准仅覆盖 8 个抽查导出，其余完整导出由 NPY 成员 CRC32/长度、既有容器 SHA、全部 PKL 中心值及几何规则回放核对。准确说明校验层级；不把 CRC32 称为密码学全库数组 SHA 保证，不把小样本通过冒称另一台电脑已全库通过。

sample 失败材料只在排查需要时保留；确认问题解决后，清理明确属于本工程的失败 sample_* 临时目录。对任何递归清理先核实绝对目标在预期 .work 内并有相应 .sample_owner.json；不要删除原件或用户已有输出。完整 batch 只留最新一套当前缓存，不积累逐轮副本。

本授权不包含启动模型训练；数据生成和验证完成后直接交付路径。计算 PASS 和复现一致性不能代替人工视觉/解剖/训练效果验收。
