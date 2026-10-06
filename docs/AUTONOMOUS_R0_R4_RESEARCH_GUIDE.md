# 目标机 Codex 自主实验指南：DMM 先验训练 R0–R4

日期：2026-10-06。适用目标机：RTX 5090、128GB、Windows + WSL2，已有 Python 3.10 项目环境。环境已配置，本指南不要求重新安装。

## 1. 授权与最终任务

用户已明确将 R0–R4 的实验规划、执行、诊断、必要源码修改和阶段推进交给目标机 Codex 自主负责。**本指南取代此前“R1 后等待主开发端批准”“R2/R3 必须由另一对话发配置”的协作限制。** 可以自行生成 R2/R3 配置、设计对照、修复训练与评价代码、调整模型及配方、延长或提前停止实验、选择检查点，并在证据支持时进入 R4；不必逐轮请求本对话批准。

自主推进不意味着预设所有阶段必然成功。可以回退、重试、否定方案；必须保持可复现证据，不以放宽校验、忽略失败或不断增加训练时间换取“通过”。只有确实缺少无法获取的数据/权限、资源不可用、需要改变研究范围或用户必须决定的取舍时才询问用户。常规技术选择由你作出并记录理由。

最终研究目标是：**已知相机、静态上颌和下颌、逐牙语义掩码驱动的合成三维重建。** R0–R4 的直接交付物是两套训练完成、来源可追溯、具备独立几何评价的 DMM 候选先验，以及完整的实验结论和后续重建交接材料。

这里的“静态”是多视图共享同一病例的形态与两颌位姿，不表示两颌世界位姿已知。先验由三维扫描训练；之后冻结 decoder，使用掩码和相机拟合病例 latent 与每颌整体位姿。不要把这两个优化问题混成同一训练任务。

暂不扩大到 FLAME、人脸视频、动态下颌跟踪、分割网络训练、多卡系统或临床精度声明。可以为当前问题修复/改进模型，但若改变表示语义、单位、接口或 latent 含义，应建立新版本、迁移和验收；不能偷偷让旧 bundle 继续声称兼容。形状/排列解耦、额外逐牙刚体姿态属于新的表示研究，不作为 R4 的默认前提。

## 2. 开始前掌握事实，不重复安装或重跑

- 项目已在 `/mnt/e/.../dental/multiview_teeth_reconstruction`。使用实际工作目录，不猜测中间路径，不重新 clone、不擅自移动项目。
- Python 3.10 环境基线为 `training-r0-v3` / `2e5b15e`；本指南随 `training-autonomy-v1` 交接。模型和训练 JSON 未因本指南改变。实际运行可能已有后续本地提交，先检查，不能强制退回旧标签。
- 远程：[JohnTitor-elpsykongroo/multiview_teeth_reconstruction](https://github.com/JohnTitor-elpsykongroo/multiview_teeth_reconstruction)。修改后的 DMM 与 nvdiffrast 已作为普通源码纳入仓库，不需要下载官方版本覆盖它们。
- 先读适用的 `AGENTS.md`，检查 Git 状态、活动进程、tmux 会话、已有预检/训练结果。确认哪些任务已经成功、哪些仍在运行，只补缺失或失效部分。
- 数据为项目同级 `../data/Teeth3DS_DualArch_v1` 和 `../data/Teeth3DS`；后者是清单引用的原始资源，不能省略。数据集内的 `work/` 是预处理缓存，不是训练入口。
- 不改写冻结数据、已有运行、旧标签或其他人的未提交工作。环境若出现实际错误，可在明确诊断后修复；更换依赖需记录新环境并重做相应数值检查，不因“更新版本”本身而重装。

本开发机已知证据：Windows Python 3.10 + 同版 CPU PyTorch 的 26 项训练/交接测试通过；Linux cu130 wheel 与科学计算依赖解析通过。此前另有本机 GPU 小病例检查。**这些都不等于目标机 R0、全量训练或模型质量已经通过。** 目标机已有实际结果应优先于这些历史描述。

## 3. 必读材料与阅读产出

### 3.1 按此顺序阅读仓库文档

| 顺序 | 材料 | 必须弄清楚的问题 |
| --- | --- | --- |
| 1 | [主设计文档](multiview_teeth_reconstruction_semanticxy.md) | 收窄后的任务、训练与拟合的区别、最终输入输出；重点第六至十一章 |
| 2 | [双颌参数与数据契约](dual_arch_parameter_data_contract_v1.md)、[固定契约配置](../configs/contracts/dual_arch_static_semantic_v1.json) | 单位、FDI、presence、canonical、共享位姿、latent 语义 |
| 3 | [训练交接与数值修复](TRAINING_HANDOFF_20261006.md)、[训练整合说明](training_readiness_v1.md) | 初始化问题、损失分母、验证与选模、统计和兼容性 |
| 4 | [分轮计划](TRAINING_ITERATION_PLAN.md) | 现有入口、初始预算、反馈脚本；其旧人工审批流程由本指南替代 |
| 5 | [源码/ICCV 对照审查](DMM_source_and_ICCV2025_audit_20261006.md) | 哪些来自论文、哪些为项目改编、哪些结论曾经未完成 |
| 6 | [双颌接口](dual_arch_interfaces.md)、[可选扩展](dual_arch_optional_extension_v1.md) | train/evaluate/export 的实际契约，双颌调度不等于联合模型 |
| 7 | [渲染](differentiable_rendering_v1.md)、[SemanticXY](semanticxy_matching_v1.md)、[拟合诊断](staged_fitting_gpu_v1.md) | 先验最终如何使用，哪些梯度/优化问题仍未解决 |
| 8 | [oracle 观测](stage2_oracle_workflow.md)、[观测契约](tooth_observations_v1.md)、[阶段三融合](stage3_observation_fusion_v1.md) | 后续合成标签与真实推理边界、相机和像素约定 |

**历史快照不能直接当作当前实现。** 审查文档中“mlp 未改”“恒定学习率”“缺少 best_val”等描述属于早期状态；当前已增加 SIREN 输出初始化、分组学习率/调度和验证选模。反之，单元测试通过也不能覆盖文档中的全部质量问题。发现冲突时：用户最新目标优先；实现行为以当前源码和实测确定；数据接口的既定语义不能因实现错误而被静默改写。将差异写入阅读笔记。

数据端还要读 `../data/Teeth3DS_DualArch_v1/DATASET.json`、`TRAINING_DATA.txt`、两颌 manifest/canonical、相关 QC 报告。冻结目录中的旧 `train_config.json`/specs 不自动替代项目 `configs/training_handoff_v1/` 的更新配置。

### 3.2 必须阅读的两篇论文

**A. Zhang et al., An Implicit Parametric Morphable Dental Model, SIGGRAPH Asia / ACM TOG 2022。**

- [作者项目页](https://vcai.mpi-inf.mpg.de/projects/DMM/)、[arXiv 原文](https://arxiv.org/abs/2211.11402)、[作者 PDF](https://vcai.mpi-inf.mpg.de/projects/DMM/data/paper_lowres.pdf)、[官方源码](https://github.com/cong-yi/DMM)。
- 精读 §3、§4.1–4.3、§5.1–5.2，以及评价/消融和局限部分；公式以原文为准。
- 理解组件隐式场、模板与形变、超网络、修正场、混合权重；核对几何/法线/Eikonal、语义、中心、平滑、latent 等项的数学含义和实现归一化。
- 不把论文的 GPU 数量、epoch、loss 权重或归一化空间误差直接迁移成本项目的预期用时、毫米阈值或最佳配方。

**B. Zheng et al., Teeth Reconstruction and Performance Capture Using a Phone Camera, ICCV 2025。**

- [CVF 官方论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_ICCV_2025_paper.pdf)。
- 精读 §4.1（表示适配）、§4.2（语义、SemanticXY、正则、遮挡）、§5.1（优化流程）、§5.3–5.4（评价与消融）。
- 理解隐式场如何进入网格渲染再回传、软语义与几何 XY 为何提供约束。论文完整任务包含脸部与动态捕捉；本项目只借鉴静态牙齿重建所需部分。
- 不把本项目 partial OT、局部隐函数梯度桥或上下颌防穿插代理称为论文实现的逐项等价复现。

开发机原文位于 `D:\文献\参数牙\DMM.pdf` 和 `D:\文献\参数牙\Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_ICCV_2025_paper.pdf`，目标机不保证有此路径。PDF 不在 Git 中；可读取目标机已有文件或从上述原始来源获取，保存至忽略的 `runs/literature/` 并记录 URL/文件 SHA256。中文译本仅辅助理解，不替代原文。若网络暂不可用，仍可执行与论文阅读无关的 R0/R1；涉及论文特定公式的修改前必须核实原文，不伪称已读。

阅读产出：在实验记录中列出“论文规定 / 上游实现 / 本项目实现 / 实际待验证”四列，尤其核对 loss 分母、采样、坐标、先验正则和评价指标。不要全文复制论文进仓库。

### 3.3 对应源码入口

| 文件/目录，相对 `third_party/DMM/` | 关注点 |
| --- | --- |
| `networks/dmm_net.py`、`deform_net.py`、`mlp.py`、`meta_modules.py` | 组件网络、代码维度、查询梯度、初始化 |
| `networks/loss.py`、`train_dmm.py` | 上游旧路径对照；不要启动旧 `-e` 流程替代 manifest 入口 |
| `data/arch_dataset.py` | 资源校验、patient split、采样、存在牙、embedding row |
| `training/recipe.py`、`arch_training.py` | 损失定义、有效点、学习率、更新、检查点、续训与导出 |
| `training/validation.py` | 独立临时代码、点划分、几何门槛、失败保留 |
| `dmm/provenance.py`、`validation.py`、`bundle.py` | 源码/运行时绑定、单位、先验统计 |
| `dmm/__main__.py` | CLI 实际支持的参数，不能凭文档猜不存在的命令 |
| `dmm/scene.py`、`surface.py`、`rendering.py`、`semanticxy.py`、`fitting.py` | 训练后使用边界；无需为了先验训练先重做所有渲染研究 |

项目脚本需读 `training_target_preflight.py`、`training_smoke.py`、`verify_training_package.py`、`run_training_wsl.sh`、`collect_training_feedback.py`；训练回归以 `test_dual_arch`、`test_training_handoff`、`test_training_readiness`、`test_repository_handoff` 为起点。

## 4. 必须保持的实验含义

1. 上下颌独立 auto-decoder：同时学习网络与训练病例代码；验证/测试冻结网络，只拟合临时代码。不是监督照片网络，也不是把 val 加入训练 embedding 更新。
2. 当前每颌 14 颗非第三磨牙 + 牙龈，10 维组件码；模型单位固定 `1 = 50 mm`，canonical 为 LPS。不得通过每病例自由缩放掩盖单位错误。
3. latent 耦合形态与排列，不能说 latent-only 一定只改形状。下游额外逐牙 pose 容易带来不可辨识性，不默认添加。
4. 缺牙通过显式 presence 处理。牙龈是几何组件，牙齿图像语义里归背景；二者的标签语义不同。第三磨牙在 28 牙观测契约中是 ignore，不当作缺失普通牙或可靠背景。
5. 患者隔离跨上下颌和镜像保持一致；镜像是训练增广，不是新的独立患者。验证/测试只用原始病例，不能用 test 指标挑超参。
6. 当前 canonical 来自冻结完整原始 train，而小子集训练的临时代码均值来自实际选中原始 train。小子集实验不等于 canonical 也只用了这些病例，报告须如实说明。
7. 数据期望计数：上颌 train 1290（原始 645 + 镜像）、val 88、test 80；下颌 train 1276（原始 638 + 镜像）、val 86、test 78。现场以通过哈希校验的清单核实，不以文档数字覆盖异常。
8. 当前验证主指标是逐牙等权的 observed-to-generated **网格采样点**距离，单位 mm；不是精确点到三角形、Hausdorff 或临床精度。反向距离涉及生成封口/未观测区域，只作诊断。
9. 后续图像拟合的输入与 `truth/annotations` 隔离。三维先验监督合法使用训练扫描；oracle 合成观测必须显式标为 oracle，不能冒充 RGB 识别精度。

## 5. 自主执行循环与版本管理

每轮执行：**观察 → 假设 → 最小对照 → 运行 → 评价 → 决定继续/修改/回退。** 在跑之前写下此次要回答的问题、病例列表、唯一主要变化、评价预算、推进/停止规则和资源预算。配置选择与日志应足以让另一位工程师重跑。

### 5.1 计划、预算和可比较性

- 自行建立 `runs/autonomy_<UTC>/plan.md`、实验清单和 `status.json`，其中写 R0–R4 的当前阶段、活动 PID/tmux、配置 SHA、源码 commit、运行目录、下一动作。
- R0 实测每步、每 epoch、一次验证的时间和峰值显存后，估算 R2/R3/R4 耗时与磁盘需求。建议仅用一个 GPU 作业；初始诊断每个问题先安排 1 个基线加 1–3 个单因素候选，证据不足再扩展并记录原因，不默认大网格搜索。
- 不同 reduction/权重的总 loss 不能横向直接排名；保留统一的几何、法线、语义和失败率指标。同一候选比较使用相同病例、采样种子、验证 latent 优化预算与网格采样预算。
- 提高网格分辨率或 val fit_steps 后，应重新评价所有候选，不能只给新模型更多优化预算。统计平均值同时报告分母、逐牙/逐病例 P95、最差值与失败原因；不要把失败记作零或从总数中消失。
- 质量阈值由你在 R2 基线/重复性检查后提出并冻结，R3/R4 使用同一评价表。阈值必须有任务依据，避免臆造临床毫米门槛；如必须修订，另立协议版本并按新标准重评旧候选。

### 5.2 修改源码和配置可以自主完成

- 在独立 `training/5090-<唯一名称>` 分支开发；先保留当前来源，不能覆盖未知未提交工作。必要改动、固定病例列表和评价工具提交 Git，可推送专用分支同步；不强推共享 main，不移动旧标签，不上传数据/权重。
- 为每次真正启动的实验冻结一个干净 commit。配置放 `configs/autonomy/<实验编号>/` 并提交，输出放 `runs/`。JSON 引用相对其所在文件解析，换目录必须重建路径，资源不变时保持资源 SHA，不随意改哈希。
- DMM 模型源码、规范、损失、验证行为变化，先完成对应的接口/数值测试和小病例检查，才能再长训。新增能力先实现、测通再训练，不能先占用 GPU 把尚未完成的接口当作训练中待办。
- 当前 Git 预检绑定**整个项目 commit**，即使只提交文档也会使旧 PROOF 与新 commit 不符。运行过程的笔记先放 `runs/`，不要为了日常日志不断改变 commit；源码/受控配置提交后重新生成预检。
- 训练启动后保持该 checkout 不变。需要同时开发时使用独立 checkout/worktree，并正确处理其同级数据路径；不得移动数据来迎合目录。
- 同源码/配置兼容、同运行时的续训可使用 `--resume`；当前仅允许 epochs/保存频率等既定例外。改变模型、loss、采样、训练子集通常重新训练。若热启动有必要，自行实现明确的迁移入口、参数映射、优化器重置规则和测试，记录它是 warm-start，不能伪造 resume 签名。

## 6. R0–R4 流程和自主推进条件

| 阶段 | 初始方案，可按证据调整 | 阶段要回答的问题 | 推进依据 |
| --- | --- | --- | --- |
| R0 | 目标机预检 + 实际配置小步优化 | 输入和计算是否可靠？ | 预检通过；无 NaN/Inf；需要的实际配置采样检查通过 |
| R1 | 每颌 2 病例 1 epoch smoke；32 train / 4 val / 2 epoch pilot | 完整链路能否工作，早期数值是否合理？ | 优化、验证、保存与诊断正常；可自行进入 R2，不等另一对话 |
| R2 | 先保持 32/4，延长到约 20 epoch；必要时 1–8 病例记忆对照 | 有限病例能否学出有效几何？ | 原始场与梯度合理；学习/几何有可解释进展；持续缺面/缺牙先诊断 |
| R3 | 候选 128 train / 16 val，约 20 epoch | 扩展病例后是否泛化，选择哪种配方？ | 在冻结评价协议下选出稳定候选；有效几何、逐牙表现与资源可支撑 R4 |
| R4 | 全部 train + 原始 val；100 epoch 起始候选，按预设规则延长/停止 | 全量训练能否产出合格候选先验？ | 完成固定评价和模型选择、可追溯 bundle 与 final 报告；明确质量结论和不足 |

以上病例数和 epoch 是起点，不是不可修改的审批配额。R2/R3 配置尚未存在正是需要你实现的工作，**不要据此停在等待主开发端发文件的状态**。

### R0：已有环境上的可靠性检查

```bash
# 已位于实际项目根目录
bash scripts/run_training_wsl.sh preflight
PROOF="runs/target_preflight_本次实际时间戳/report.json"
```

只使用本次成功的 `TARGET_READY_FOR_BOUNDED_PILOT` 报告。环境或源码改变后重新检查；保留失败报告，先诊断根因。

当前预检固定读取 handoff pilot 配置。**它不自动验证你新建 R2/R3 的全部数值行为。** 对新增架构/采样/配方，必须另跑该配置的数据校验和 `training_smoke.py`，或自主扩展预检使其明确接受当前实验配置；不要把旧小采样的显存估计当成新预算的保证。

### R1：链路与短程证据

```bash
bash scripts/run_training_wsl.sh smoke upper "$PROOF"
bash scripts/run_training_wsl.sh smoke lower "$PROOF"
bash scripts/run_training_wsl.sh pilot upper "$PROOF"
bash scripts/run_training_wsl.sh pilot lower "$PROOF"
```

逐条检查退出码、`report.json`、`failure.json`、`progress.jsonl`、`validation_*.json` 和检查点。早期无零水平面可以记录为几何失败后完成短预算；非有限值、OOM、错标、错误单位、丢失保存状态不可当作正常继续。

R1 不是收敛实验。不能因为 2 epoch 几何差就否定模型，也不能因为 loss 下降就通过质量验收。完成结果分析后自行设计 R2，写明为何选择延长原配置或先做诊断。

### R2：先把有限病例学会

- 从 handoff pilot 构造明确版本的 upper/lower 配置，优先保持病例、配方和验证预算只延长训练；若要继续 R1 checkpoint，核对签名允许的变化，不更改其他字段。
- 小病例记忆实验只用于定位，区分固定点过拟合、重新采样表面泛化和未见病例泛化。训练病例应检查其已学习的代码；重新拟合一个 val code 不能替代训练代码检查。
- 按 FDI 检查语义权重、组件/混合场、零水平面与实际表面距离，保存正面、侧面、咬合面图。若尚无可用的训练病例评价/预览工具，可自主实现并写测试；不要假设 `evaluate-checkpoint` 能评价 train。
- 至少做一次重要改善的同种子对照；改善接近采样波动时重复采样或第二种子，不凭一次总 loss 认定修复有效。
- 10 维统计每组件需至少 11 个原始存在训练病例。1–8 病例模型不能直接作为最终 prior 导出；即使训练病例总数大于 11，也要逐组件检查支持数。

### R3：选配方与验证预算

- 从原始 train/val 分别选固定病例列表，记录选择方法、种子、缺牙/牙位覆盖。扩展集尽量包含此前病例以观察趋势，不能只挑容易病例。检查跨颌 patient split。
- 推荐将主要问题拆开：loss reduction、中心约束、normal_epsilon、学习率/调度、latent 正则、采样量、网络容量；先单因素筛选，再对必要组合做对照。适当增加网络容量是可自主决定的实现选择，须重新完成小样本检查。
- 以一致预算重新评价基线和候选，确认提升不是验证码优化更久、采样更多、分辨率更高或跳过难例造成。
- R4 前固定数据列表、配方、代码版本、验证预算、主/辅助指标、停止/延长规则。当前 best_val 规则要求选定 val 全部产生有效几何；R3 有持续无效病例时优先解决，不能通过删除病例或改变分母制造成功。

### R4：全量、选模与结束状态

- 从 R3 选定方案构造全量配置，保留全部 train 和镜像父子关系，使用全部原始 val。可以沿用候选 100 epoch / 每 5 epoch 验证，也可按实测时间和 R3 曲线合理调整并预先记录。
- 数据集变化后不能直接沿用子集训练的 resume。默认重新训练；经测试的 warm-start 必须显式标注。
- 估算完整验证耗时；val 的 decoder 冻结与临时代码优化不能为了速度偷偷省略。吞吐优化如 AMP、batching、缓存属于数值/数据流改动，先做等价性与梯度检查，不默认开启。
- checkpoint 选择以固定 val 协议为准。保留 best_train 作诊断、best_val 作候选、final 作最终训练状态；三者不能混称。
- 全量早期失败不必立即永久放弃；按预先定义的诊断窗口排查。若反复复现且受控候选均不能解决，停止无效长训，交付明确的“尚未达到验收”报告，而不是一直追加 epoch。
- 训练终止必须留出评价、导出、可视化和总结时间。不要以启动后台进程、到达 epoch 数或导出文件作为任务完成的全部依据。

## 7. R2/R3/自定义 R4 的可执行入口

以下是入口模板，**由你创建并提交真实配置后替换变量**，不是要求用户手填。`run_training_wsl.sh` 目前只有 preflight/smoke/pilot/formal，不存在 r2/r3 模式。

```bash
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/third_party/DMM${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

UPPER_CONFIG="configs/autonomy/你已创建的实验/upper.json"
LOWER_CONFIG="configs/autonomy/你已创建的实验/lower.json"
PROOF="runs/本次成功预检/report.json"
python scripts/verify_training_package.py --preflight "$PROOF"
python third_party/DMM/dmm_cli.py validate-training \
  --config "$UPPER_CONFIG" --other-config "$LOWER_CONFIG"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="runs/autonomy_$STAMP"
mkdir "$RUNROOT"

# 对新的实际采样配置：先读配置，把示例数值替换为该轮真实值；下颌同样检查。
python scripts/training_smoke.py --config "$UPPER_CONFIG" \
  --steps 2 --points 256 --offsurface 2048 --device cuda \
  --output "$RUNROOT/upper_sampling_smoke.json"

# 每一步成功后才执行下一步；持久 shell 脚本应使用 set -euo pipefail。
python third_party/DMM/dmm_cli.py train --config "$UPPER_CONFIG" \
  --output "$RUNROOT/upper" --device cuda
python third_party/DMM/dmm_cli.py train --config "$LOWER_CONFIG" \
  --output "$RUNROOT/lower" --device cuda
```

将控制台 stdout/stderr 保存，使用 tmux 或可靠持久会话，防止终端断开丢失长任务。每次输出目录必须新建；有旧目录不能覆盖。自定义入口不会自动为你运行预检，必须保留上面的前置校验。

若仍使用未改动的 handoff 全量候选，R4 可以直接调用 `bash scripts/run_training_wsl.sh formal upper "$PROOF"` 和 lower；若采用 R3 改进配置，必须明确传入新配置，不能误用旧 formal_candidate。

## 8. 调试顺序与判断方法

| 现象 | 先看什么 | 最小诊断与允许的修复 |
| --- | --- | --- |
| 路径/哈希不匹配 | JSON 所在目录、引用目标、实际资源 SHA、git diff | 修复搬运或配置引用；资源换版需新数据版本，绝不把错误文件的哈希写回冻结 manifest |
| 固定样本 loss 不降 | 各原始项/权重、有效点数、requires_grad、optimizer 参数组、实际参数更新 | 单病例固定点比较；查 detach、错误标签、缺失梯度、学习率和符号；再验证重新采样 |
| 初始法线梯度巨大 | reference 初值、空间梯度范数、normal_epsilon、裁剪前梯度 | 已知零输出 + 很小分母会放大梯度；当前 SIREN 输出初始化和 eps=0.001 是起点；做初始化/epsilon 单因素对照，不只加重裁剪 |
| NaN/Inf 或 clip 长期主导 | 第一个异常步、具体 loss 家族、梯度分位数、数据范围 | 保存可复现病例和点索引；小预算 anomaly 检查；查高阶导、归一化、SDF/normal 数据；修复后比对参数更新和几何 |
| 没有零水平面 | 场 min/max、正负比例、表面绝对 SDF、采样域、offsurface 项 | 区分早期训练不足和恒定场/符号/尺度错误；不要通过移动等值面假装生成成功 |
| 边界非正/表面截断 | canonical 位置、训练域覆盖、边界 SDF、原始扫描边界 | 查错位、尺度和域监督；域需变化则新配置/契约及全量复评，不能直接关闭边界门槛 |
| 混合场有表面但牙位缺失 | 每牙权重分布、presence、标签映射、中心项、组件/混合场差异 | 分别显示组件与融合表面；对照语义和中心项，确认不是 mesh face 标签错误 |
| train 好、val 差 | train 重建、临代码初始化/步数/梯度、val 难度与缺牙分布 | 先排除验证优化不足；统一预算后比较正则/容量/数据；不能在 val 上更新 decoder |
| 改 loss 后总 loss 更小但几何更差 | raw/weighted 各项、分母、统一几何与语义指标 | 重新按相同评价预算比较，避免跨 reduction 总 loss 误判 |
| OOM/过慢 | allocated/reserved、真实点数、验证分辨率、二阶图、I/O | 找主要瓶颈；分块查询、缓存或降低预算需记录并做对照；128GB RAM 不等于显存，不能跳过难例 |
| 续训或导出拒绝 | source fingerprint、runtime、config signature、训练子集 | 返回原实验源码导出；新模型做新实验/显式迁移，不改签名字段 |
| bundle 统计失败 | 每组件实际原始存在病例数、latent 方差、镜像和未训练行 | 统计只用实际训练原始存在代码；补足数据或查代码是否真的更新，不复制镜像凑数量 |
| 图像 IoU 好但位姿/牙形差 | latent/整体姿态补偿、逐牙几何、绝对位姿和可见性 | 这是后续拟合可辨识性问题；不能据 aggregate IoU 宣布先验或重建通过 |

调试输出需足够定位，但避免把所有点张量长期写盘。先保存失败病例 ID、种子、配置、关键点索引、首个异常项、参数/梯度统计；只有复现需要时保存有限张量。

## 9. 评价、导出与当前缺少的能力

### 9.1 验证可直接使用，训练/测试评价需检查或补齐

```bash
# 配置必须与 checkpoint 的训练子集、specs、配方等兼容；固定 decoder。
python third_party/DMM/dmm_cli.py evaluate-checkpoint \
  --config "$UPPER_CONFIG" --checkpoint "$CHECKPOINT" \
  --output "$NEW_EVALUATION_DIRECTORY" --device cuda

python third_party/DMM/dmm_cli.py export-bundle \
  --checkpoint "$CHECKPOINT" --output "$NEW_BUNDLE_DIRECTORY" \
  --model-id "upper_明确实验编号"
```

当前 `evaluate-checkpoint` 调用 `validate_arch`，只接受原始 **val**，并不提供 train/test 切换。若要完成训练病例记忆检查、最终独立 test 或自动几何图集，你可以自主实现这些工具。必须有显式 split/模式、测试隔离、冻结 decoder、互斥拟合/评价点、固定预算及失败保留测试；禁止把 test 行改成 val 来复用入口。

### 9.2 最终 test 的纪律

R2/R3 不读取 test 的几何表现。R4 在代码、配方、checkpoint 选择和评价协议冻结后，使用原始 test 上颌 80 / 下颌 78 做一次最终评估；先确认实现已测试。可对每个测试病例拟合新的临时代码，这是 auto-decoder 的推断步骤，不能更新网络，也不能根据 test 结果再挑候选。若 test 暴露严重问题，明确记为该冻结方案的测试结果；后续探索不可继续把同一测试集宣称为从未见过的独立留出集。

### 9.3 导出和交接

用训练对应源码导出，记录 checkpoint/数据/配置/源码 SHA、实际训练病例列表、每组件支持数和模型单位。检查 bundle 加载、均值及若干代码的场/网格、缺牙行为和有限梯度。当前状态 `ARCH_BUNDLE_EXPORTED_NOT_QUALITY_ACCEPTED` 只说明导出完成，研究评价另写结论，不随意把状态字符串改成通过。

R0–R4 不要求完成多视图拟合研究。给下一阶段保留均值模型、选中 checkpoint、bundle、代表病例图集与已知失败。已有细网格整图 AA 梯度、GPU 严格轨迹重放、pose/latent 补偿和防穿插问题仍需独立验证；它们不参与当前训练，不能一概阻断 R0，也不能因训练完成就宣称已解决。

## 10. 日志、反馈与完成标准

- 每个实验记录：假设、父实验、commit/tag、配置 SHA、数据/患者/牙位分布、种子、运行时、命令、开始/结束时间、峰值显存、病例和验证预算、结果、失败与下一决策。
- 汇总文件放 `runs/`，至少提供逐轮表、逐病例/逐牙表、损失/梯度曲线、固定视角几何图、资源耗时和决策记录。若使用新的指标，标明单位、采样预算、分母、适用表面范围。
- 使用 `scripts/collect_training_feedback.py --run ... --log ... --output ...` 打包。它只收集运行目录顶层诊断文件，不递归收集子目录或权重；新增工具若输出子目录，应另打包并给清单，不能以为已经自动收集。
- 源码/配置通过 Git 同步，模型/大报告另存并给 SHA。保留失败运行和完整检查点，不以清理磁盘为由删除唯一恢复状态。
- 自主长任务需要持续记录状态和下一步；没有真实 scheduler/后台执行能力时，不承诺会话结束后仍会自动醒来。运行进程放持久会话，交代恢复/查看方法。周期性向用户简短汇报有意义的进展，但不把每轮技术判断变成审批问题。

完成交付应回答：两个先验是否成功训练；按什么协议选模；val/test 各有多少成功和失败；哪些牙位/病例仍有缺陷；bundle 是否可加载；哪些结果可复现；后续图像重建还缺什么。若无法达到质量标准，交付失败机制、已尝试的受控对照与可复现材料，明确当前阶段；不能伪造“R0–R4 全部完成”。

## 11. 给目标机 Codex 的启动指令

> 阅读并按本指南执行。环境已配置，不重复安装。先核对实际 checkout 和已有结果，阅读必读材料，写出你自己的阶段计划、预算和评价表。然后自主完成缺失的 R0/R1，设计实现 R2/R3，依据证据修改、验证并推进 R4。你拥有当前任务范围内的实验与实现决策权，不需等待另一对话逐轮批准；保持数据隔离、版本可追溯和真实质量结论。只在确实需要用户输入或超出研究范围时询问。
