# 可选扩展：上下颌训练任务与联合拟合

## 范围与启用方式

此扩展直接位于 DMM 源码中，新增显式命令与可选配置；既有 `train`、`fit-scene` 使用方式继续有效。没有启动真实数据的正式训练。

- 两颌分别训练独立 decoder、embedding、统计 prior；不要求病例逐行配对、不共享权重、不学习跨颌咬合 prior。
- 联合图像拟合沿用两颌各自的整体 SE(3) 和 q、已知相机、全部有效视图、共同深度缓冲与固定牙龈码。
- 可选防穿插项默认关闭。启用时只惩罚穿入，不吸引分开的上下牙，不强制闭合咬合。
- 不增加逐牙刚体变量、动态下颌运动或形状/排列解耦。它们需要不同的数据和模型约定。

## 1 双颌训练任务入口

[dual_training.py](../third_party/DMM/training/dual_training.py) 组织两个原生 `train_arch` 任务，按 upper → lower 顺序执行。先校验两份 manifest、canonical 参考、网络参数和跨颌患者划分，全部通过后才创建训练输出。同一患者即使出现在不同颌的数据中，也不得跨 train/validation/test 划分。

以下命令中的路径需要替换成合规数据的实际配置。**创建配置、校验配置不会训练；`train-dual` 才会启动训练。**

```powershell
. scripts/gpu_env.ps1
# 只创建任务配置并校验输入
& $gpuPython -B third_party/DMM/dmm_cli.py make-dual-config `
  --upper-config <upper-config.json> --lower-config <lower-config.json> `
  --output <dual-training.json>
& $gpuPython -B third_party/DMM/dmm_cli.py validate-dual-training --config <dual-training.json>

# 仅在后续正式决定训练时执行
& $gpuPython -B third_party/DMM/dmm_cli.py train-dual `
  --config <dual-training.json> --output runs/<new-dual-run> --device cuda
```

输出包含 `upper/`、`lower/` 和原子更新的 `workflow.json`。各颌仍保存原生训练配置、日志、optimizer、随机数状态与 checkpoint。任务完成状态为 `DUAL_TRAINING_COMPLETED_NOT_QUALITY_ACCEPTED`，不表示模型质量验收。

### 中断续跑

```powershell
& $gpuPython -B third_party/DMM/dmm_cli.py train-dual `
  --config <dual-training.json> --output runs/<new-resumed-run> --device cuda `
  --resume runs/<old-run>/workflow.json `
  --lower-resume runs/<old-run>/lower/epoch_000010.pth
```

- 新运行目录引用已完成颌的 final checkpoint 和 report，逐一验证哈希，跳过该颌；不会修改旧目录。
- 未完成颌可用 `--upper-resume` / `--lower-resume` 指定已有的周期 checkpoint。未提供时，仅该未完成颌从头开始。
- 续跑绑定源码、双颌配置及设备；单颌 checkpoint 继续使用现有输入/源码签名校验。
- 已完成颌不能再次传入 resume checkpoint；输入改变或历史结果被篡改时拒绝恢复。
- 突然断电时，单颌工作最多回退到最后一个已保存的周期 checkpoint；调度状态以最后完整的 `workflow.json` 为准。

训练后分别用既有 `export-bundle` 导出两套 bundle，再按 [双颌接口](dual_arch_interfaces.md) 组装场景。导出继续要求足够数量的原始、存在牙位训练病例计算统计量。两套 bundle 不自动证明同患者真实咬合。

## 2 联合拟合的可选防穿插项

[collision.py](../third_party/DMM/dmm/collision.py) 在 upper → lower 与 lower → upper 两个方向计算完整混合牙弓的表面穿入代理，包括牙龈。它不改变语义、可见性恢复、真实遮挡或逐牙存在性。

令对颌隐式场为 φ、其模型坐标梯度为 g，内部点使用局部近似深度：

`d_proxy_mm = 50 * max(-φ, 0) / max(||g||, 1e-6)`

两方向分别取等面积确定性采样点，计算 `mean(max(d_proxy_mm - tolerance_mm, 0)^2 / scale_mm^2)` 后平均。总目标增加 `weight * L_collision`。

默认参数：每颌 512 个采样点、容许代理深度 0.2 mm、归一化尺度 1 mm、权重 0.1。它们是可调工程初值，尚未在真实牙模上标定。参考 [完整可选配置](../configs/collision_optional_v1.json)。

梯度连接源颌表面、目标颌隐式场及双方的 q/整体位姿；一次反传固定采样索引与梯度归一化分母。查询只在目标的已验证采样域内进行，域外点不外推神经 SDF。两颌表面仍须通过已有的正值域边界检查；内部场梯度退化时明确报错。

这是**有限采样的局部隐式场代理**，不是精确毫米距离或完整网格相交证明。可能漏掉小范围穿插，不可据此宣布咬合或临床验收。分离表面的损失和吸引梯度均为零。

### 启用

在现有 `fit-scene` 命令后增加：

```powershell
--collision-config configs/collision_optional_v1.json
```

不传该参数时完全关闭，无对颌场查询开销。也可显式配置 `enabled=false`。

防穿插项参与初始化提议、逐步回溯和 shape 重新提取网格后的完整目标判断。最终仍存在超过容许深度的采样点时返回 `COLLISION_GATE_NOT_MET`，不能报图像拟合收敛；如仍有待恢复的观测区域，优先返回 `VISIBILITY_RECOVERY_UNRESOLVED`。报告、逐步日志和 checkpoint 均保存扩展参数及诊断，续跑不能切换此选项。

拟合输出新增 `T_upper_from_lower = inv(T_world_from_upper) @ T_world_from_lower`，供读取相对位姿；未新增相对位姿优化变量，避免重复自由度。

## 3 验证与版本边界

运行 `scripts/check_dual_extension.py` 可复现原有 GPU/匹配/拟合回归和本次扩展测试，保存新的 `runs/dual_extension_<timestamp>/`。测试包含两个临时的单 epoch 小型合成训练任务，用于验证独立保存及续跑；不是正式 DMM 训练。

新增测试覆盖：跨颌患者泄漏拦截、已完成颌不重训、结果哈希校验、颌配置错配、默认关闭、分离表面无吸引、双向穿插位姿有限差分、mesh ownership 与字典顺序无关、实际 GPU 全目标回溯和可选配置续跑绑定。

本次实际 RTX 5050 CUDA 验证：**50 项测试通过，0 跳过、0 失败**。结果与日志保存于 [验证汇总](../runs/dual_extension_20261004T155156476344Z/validation.json)、[测试日志](../runs/dual_extension_20261004T155156476344Z/tests.log)。

固定的 5 步拟合诊断中，无穿插输入在关闭/启用扩展时的最终最差 IoU 均为 0.9669，状态均为预算耗尽或停滞。人为穿插输入的最大采样代理深度从 7.550 降至 4.580 mm，仍有穿插，明确返回 `COLLISION_GATE_NOT_MET`；不记作恢复成功。其 [逐步记录](../runs/dual_extension_20261004T155156476344Z/overlap_enabled/progress.jsonl) 与 [最终预览](../runs/dual_extension_20261004T155156476344Z/overlap_enabled/final_render/preview.png) 已保留并检查。这里使用未训练的八面体 DMM 夹具，不是牙齿解剖质量验证。

源码指纹随本次扩展变化，旧 bundle/checkpoint 的严格版本检查仍有效；没有放宽加载器来兼容不匹配源码。历史 GPU 验证和 sanity 结果保留为其原版本的证据。细网格整图 AA 有限差分限制仍见 [GPU 与分阶段拟合说明](staged_fitting_gpu_v1.md)。
