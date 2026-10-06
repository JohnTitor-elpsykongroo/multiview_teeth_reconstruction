# DMM 正式训练前的源码整合

日期：2026-10-06。目标为已知相机、静态双颌、逐牙语义掩码驱动的合成重建。完整方案同时更新在 `D:/WorkSpace/Dental/docs/multiview_teeth_reconstruction_semanticxy.md` 第六至十一章。

后续训练交接已修复本文记录的 reference 初始大梯度，增加跨系统哈希、运行时续训绑定和目标机预检。最新配置是 `configs/training_handoff_v1/`，详见 [RTX 5090 交接说明](TRAINING_HANDOFF_20261006.md)。下文保留上一轮实现与诊断记录，不把历史失败日志改写为通过。

状态：已实现训练与拟合所需的本轮接口；正式训练、配方校准和真实重建质量验收尚未完成。保留既有 DMM 组件网络，两颌独立训练，耦合 latent 不额外叠加逐牙 pose。历史 runs、数据集、既有权重不改写。

## 主要改动

| 范围 | 实现与边界 |
| --- | --- |
| 训练配方 | `training/recipe.py` 显式解析损失权重、分母、中心 reduction、latent 初始化、三组学习率、调度及可选梯度裁剪。默认 `effective_count_v1` 保留旧本地行为；另提供 `masked_mean_v1` 全点掩码均值候选，均不声称逐位等价官方训练。 |
| 诊断 | 每轮记录各损失原始值、加权值、有效点计数、实际参与病例数、三组梯度范数及学习率；牙龈可独立配置采样额度。 |
| 验证与选模 | `training/validation.py` 冻结 decoder，为原始 val 新建临时码，以实际训练原始存在牙代码均值初始化；拟合/评价点互斥，拟合中心只用拟合点。固定预算后评价毫米几何误差、语义/法线及拓扑诊断，保留所有失败；任一病例失败时不产生可比较 score。只在完整选定验证集合成功且改善时写 `best_val.pth`。 |
| 验证指标 | 主指标为逐牙等权 observed-to-generated mesh **采样点**平均距离，受网格分辨率和面积采样预算影响；不是精确点到三角形距离、Hausdorff 或临床误差。反向距离含未观察封口，仅供诊断。 |
| 训练子集 | `training_case_ids` 可做有限病例预实验；训练 embedding 表仍保留全清单行，但导出 prior 仅统计实际训练集合中原始存在牙代码。每组件至少 d+1 原始样本；镜像需包含原病例。 |
| 续跑与来源 | 全源码 fingerprint、完整配置签名、Adam/调度/RNG 状态及训练源码快照随运行保存；续跑仅允许扩展 epochs/改变保存频率。 |
| 模型兼容 | 新 bundle 使用 decoder 可执行依赖哈希及语义契约，训练全过程仍保留完整来源。拟合器代码变化不再单独使新 bundle 失效；旧 bundle 仍要求完整源码一致，不自动升级。导出需训练时源码版本。 |
| 像素接口 | 支持旧 `integer_centers` 与照片 `edge_origin_centers_at_half`。渲染、SemanticXY、恢复射线、候选初始化统一解释，K 不隐式修改，约定冲突拒绝加载，续跑绑定约定。 |
| 均值初始化 | 从训练均值表面估计软语义面积中心，观测掩码中心构造射线，使用 24 个旋转候选及可用的 Kabsch 候选，最后用共同深度遮挡下的全场景渲染排名。无 truth 输入；可见中心不是解剖同名点，欠约束会失败，成功也不表示姿态准确。 |
| 照片交接 | `prepare-scene` 将明确半像素的完整 candidate 校验并复制到新拟合目录，保留原候选与来源。只走显式引用白名单，不读取 truth/annotations；输入可消费不等于重建已通过。 |

## 配置及执行顺序

`configs/training_readiness/` 提供：

- `recipe.json`：masked_mean 候选，显式 latent std 0.002、30 epoch 学习率减半；数值待校准。
- `recipe_effective_baseline.json`：effective-count 对照。比较时固定其他因素，不按跨配方总 loss 绝对值选优。
- `validation.json`：固定预算验证选项。
- `upper_pilot.json` / `lower_pilot.json`：各 32 原始 train、4 原始 val、2 epoch；每组件原始支持最少分别为 17/18，满足 10 维统计最低数量。这些短程训练尚未执行。
- `upper_formal_candidate.json` / `lower_formal_candidate.json`：全数据训练候选草案，不代表正式配方已批准或已运行。

在项目根目录使用已配置环境，先校验输入与上下颌患者划分：

```powershell
. scripts/gpu_env.ps1
& $gpuPython third_party/DMM/dmm_cli.py validate-training --config configs/training_readiness/upper_pilot.json --other-config configs/training_readiness/lower_pilot.json
```

通过输入检查后，下一阶段再执行有限病例训练。下列命令是可执行说明，本轮没有执行：

```powershell
& $gpuPython third_party/DMM/dmm_cli.py train --config configs/training_readiness/upper_pilot.json --output runs/upper_pilot_v1 --device cuda
& $gpuPython third_party/DMM/dmm_cli.py train --config configs/training_readiness/lower_pilot.json --output runs/lower_pilot_v1 --device cuda
```

输出目录必须不存在。`best_train.pth` 保留作诊断；若全部验证病例失败，可能没有 `best_val.pth`，应修复原因而不是用失败平均值选模。续跑另建输出目录，历史最优分数继续保留，只有产生更优结果才在新目录写新 best_val；原目录检查点必须保留。

独立验证、模型导出和场景准备示例：

```powershell
& $gpuPython third_party/DMM/dmm_cli.py evaluate-checkpoint --config configs/training_readiness/upper_pilot.json --checkpoint runs/upper_pilot_v1/best_val.pth --output runs/upper_pilot_eval_v1 --device cuda
& $gpuPython third_party/DMM/dmm_cli.py export-bundle --checkpoint runs/upper_pilot_v1/best_val.pth --output runs/upper_pilot_bundle_v1 --model-id upper-pilot-v1
& $gpuPython third_party/DMM/dmm_cli.py prepare-scene <candidate_manifest.json> --output <fresh_scene_root>
```

验证输入必须匹配 checkpoint 的训练配方、实际训练集合、specs、清单和 canonical 参考；验证预算可显式修改。执行新 `fit-scene` 时可加 `--initialize-mean`，不能与 `--resume` 同用。prepare-scene 仍需要完整模型与初始化结构；均值初始化在载入后替换 q 和 pose。

## 验证记录与尚未关闭的门槛

本轮证据在 `runs/source_upgrade_20261006/`：修改前文件和哈希、完整回归日志、真实病例 CUDA 前后向报告及最终变更清单。最终计数与源码哈希见该目录 `completion_report.json`。

真实数据冒烟各使用 `00OMSZGW_upper`、`00OMSZGW_lower` 的 16 点/组件与 32 非表面点，完整当前 10 维网络执行一次 loss/backward；无 optimizer step、无模型产物。三组梯度均有限非零，但 reference 组初始梯度范数约 3e9，说明“有限”不足以证明训练稳定；短程实验需跟踪实际参数更新、梯度裁剪对照及几何改善。约 0.5 GB 峰值只适用于该小采样冒烟，不是正式训练显存估计。

单元测试中选模/续跑测试使用微型夹具且部分几何评分为 mock；另有真实网格提取、实际均值初始化渲染及 CUDA 梯度测试，仍不能替代真实牙列精度实验。

本轮曾在 GPU 拟合续跑测试中观测到约 0.0023 mm 的平移差，未通过原 2e-5 mm 轨迹比较门槛；另做两个不中断控制也出现差异，说明该门槛不能单独诊断检查点损坏。失败日志 `full_regression_2.log` 和 `resume_diagnostic/` 原样保留。现分别验证 GPU 参数/历史/游标精确恢复及参考后端 1e-10 的连续轨迹对照，并保存 `starting_parameters.json`。这是明确重放契约的调整，不是宣称 GPU 严格数值重放已通过，也没有放宽原门槛后重新宣布通过。

下一步先做有限病例过拟合、独立 val 拟合和资源测量，再冻结源码/配方进行正式训练。训练后需验收均值/随机码表面、逐牙误差、缺牙支持和统计 prior，再进入双颌合成恢复。已有细网格整图 AA 严格有限差分限制、位姿/latent 耦合和防穿插门槛仍保留，不以本轮通过的接口测试宣布解决。
