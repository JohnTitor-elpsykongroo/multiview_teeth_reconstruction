# 静态双颌重建参数与数据约定

版本：1.0.0。确定日期：2026-10-04。状态：DATA_SCENE_INTERFACES_IMPLEMENTED。

本约定确定已知相机、静态上下颌、逐牙语义掩码驱动的合成重建所使用的参数和数据含义。单颌训练及双颌加载、组装接口已直接实现在项目内 DMM 源码中，具体字段、入口和验证范围见 [接口使用说明](dual_arch_interfaces.md)。分阶段图像拟合器已实现；2026-10-06 新增训练验证、像素约定与初始化接口，见 [训练前实现说明](training_readiness_v1.md)。机器可读的固定选项见 [contract defaults](../configs/contracts/dual_arch_static_semantic_v1.json)，该 JSON 是约定配置，不是 JSON Schema，也不能直接作为训练配置。

用户已经认可的 synthetic sanity check 结果作为参考保留。新规范不追溯改变旧结果的状态，也不要求重新验收旧实验。外部 semanticxy 文档第六至十一章在 2026-10-06 同步更新为静态双颌方案与训练前计划。

## 1 已确定的边界

- 同一 scene 是同一个静态双颌状态的全部合格视角。形态、排列、两颌相对姿态均不随 view 改变。
- 上下颌使用两套单颌 DMM，各自具有权重、组件 latent 和 canonical 参考，代码实现可共享。场景渲染同时包含两颌。
- V1 使用 coupled_component_dmm：逐牙 latent 同时描述形态和排列，不称为纯 shape code；不额外开放逐牙刚体参数。
- 已知相机内外参、网络权重、牙位存在性和尺度固定。优化每副颌的整体 6DoF 位姿和有牙组件 latent。牙龈 latent 固定为对应训练均值。
- 双颌和牙龈的静态深度遮挡必须处理。首版不包含脸、嘴唇、舌头、动态关节和相机优化。
- 默认 28 颗恒牙，不包含智齿；允许具有明确标注的缺牙。不同静态张口状态属于不同 scene。

后续形态与排列解耦模型需要逐牙局部坐标、局部 pose 和共同刚体模式约束，另立主版本。V1 不提前为现有耦合 latent 增加一组重复 pose 自由度。两套单颌模型也不等于已经学得跨颌咬合先验。

## 2 参数及共享关系

| 参数 | 数据形式 | 是否拟合 | 共享范围 |
| --- | --- | --- | --- |
| upper/lower 的 T_world_from_arch | 4×4 刚体矩阵，平移为 mm | 各 6DoF | scene 内全部 view |
| z[arch, fdi] | 长度 d_i 的浮点向量 | 对存在牙位拟合 | scene 内全部 view |
| z[arch, gum] | 长度 d_g 的浮点向量 | 固定为训练均值 | scene 内全部 view |
| presence[arch, fdi] | Boolean | 固定且已知 | scene 内全部 view |
| K、T_camera_from_world | 每个 view 一套 | 固定 | 仅该 view |
| 两个 decoder 的网络权重 | 固定 checkpoint | 固定 | 整次运行 |
| model_unit_mm | 固定为 50 | 固定 | 新训练数据及模型 |

d_i、d_g 必须从 model bundle 读取，不能假设官方的 10 维或旧实验的 20 维永远不变。完整联合拟合维数为 12 加全部存在牙齿的 latent 维数之和。牙龈码不计入优化变量。

不再增加第三个 scene global pose，也不开放每病例、每颌或每颗牙的额外缩放。两颌相对变换只作派生量保存：T_upper_from_lower = inverse(T_world_from_upper) @ T_world_from_lower，不作为额外优化变量。位姿矩阵不得携带缩放或反射。

内部旋转增量使用弧度和稳定的 SO(3)/SE(3) 运算；输出统一保存矩阵。评估展示旋转误差时可用度，但字段必须以 _deg 明示。优化顺序由运行配置控制，不改变参数的含义。

## 3 坐标和尺度

所有点使用列向量，矩阵按“目标坐标系 from 来源坐标系”命名。JSON 数组按矩阵行序列化，末行为 [0,0,0,1]。

```text
x_arch_mm   = 50 * x_model
x_world_mm  = R_world_from_arch * x_arch_mm + t_world_from_arch_mm
x_camera_mm = R_camera_from_world * x_world_mm + t_camera_from_world_mm
[u,v,1]     ~ K * x_camera_mm
```

### 3.1 新训练数据的 canonical 约定

上下颌均使用右手系：+X 指患者左侧，+Y 指后方，+Z 指上方。两颌使用相同的解剖轴含义，下颌不得通过未记录的轴翻转适配上颌。

每颌固定一个训练集建立的 canonical 参考，原点为该参考 14 个牙位中心的等权平均。参考牙位中心和解剖方向锚点需左右对称化后冻结，记录来源、版本及 SHA-256；该参考用于坐标对齐，区别于 DMM 网络学得的 Ref-Net 表面。建立参考不得使用验证或测试病例。

新病例先根据有依据的源单位换算成 mm，再通过存在牙位及方向锚点做刚体对齐。保存 T_arch_mm_from_source_mm。禁止每个病例单独进行相似尺度归一化，牙列大小差异应保留在几何中。方向歧义、锚点不足或单位不明的病例标记为待处理，不进入正式数据版本。

mm 到模型坐标统一除以 50。50 mm 只是固定数值尺度，不是平均牙弓宽度的估计，也不是对旧 checkpoint 的单位解释。模型采样域在归一化后按数据覆盖确定，不要求所有几何都处于 [-1,1]^3。

源单位转换系数及依据必须保存在数据清单中。旧数据若采用每例独立缩放，只有恢复经核实的源单位后，才能制作新数据；不能用统一乘数把旧 DMM 距离直接声明为 mm。旧 checkpoint 不自动满足本规范的尺寸契约，转换或重新训练需要单独记录。

### 3.2 世界坐标和双颌关系

世界坐标由合成场景定义，相机和两颌共同使用它，单位为 mm。相机固定后不再重新选择世界坐标来减小评价误差。

单颌训练不要求上下颌配对。双颌场景可采用以下装配来源，必须声明 assembly_source：

| 值 | 含义与必需依据 |
| --- | --- |
| synthetic_defined | 生成器明确指定两个 GT 位姿；只代表人工定义的静态双颌关系 |
| registered_pair | 具有外部记录的跨颌配准变换及证据；扫描病例匹配本身不够 |

scene 的 upper/lower 各自记录 geometry_source。独立采样或不同患者组件的组合必须记录 pairing=synthetic_composition；同患者但无跨颌配准证据记录 pairing=same_patient_unregistered。二者均可用于 synthetic_defined，但不能声明为该患者真实咬合。生成真值中的这些来源信息不作为拟合器的几何输入。

## 4 组件身份和存在性

| 颌 | 牙位顺序 | 牙龈组件 |
| --- | --- | --- |
| upper | 11,12,13,14,15,16,17,21,22,23,24,25,26,27 | upper:gum |
| lower | 31,32,33,34,35,36,37,41,42,43,44,45,46,47 | lower:gum |

所有组件以 (arch, component_id) 标识。每个单颌 decoder 内部仍可用 0 表示牙龈，但跨颌场景禁止把两个内部 0 当作同一组件。牙位的颌归属必须与 FDI 一致。

presence 来自完整病例标注或合成生成配置，不能从当前 view、当前 batch 的随机表面采样或像素数推断。V1 接受 true/false，不接受 unknown。缺牙组件不参加解码合成、对应 latent 优化或统计 prior。训练表可保留该行占位，但该缺失项不能得到组件训练损失或进入统计。

每 view 的 visible_fdi 由有效掩码中出现的 FDI 导出；它不同于 presence。存在但不可见的牙仍参与场景几何和遮挡，不得因当前 view 没有它而删除。若所有视角均未见某存在牙，报告 unobserved，输出属于 prior 支持的补全，不能标为该牙个体形态已恢复。

V1 模型包含两副牙龈，牙龈码固定为各自训练均值；受控合成目标也采用这一已公开的模型配置。固定牙龈码不等于混合表面上的牙龈区域完全不随其他组件变化。

## 5 相机与图像约定

每 view 必需字段如下：

| 字段 | 类型及含义 |
| --- | --- |
| view_id | scene 内唯一字符串 |
| width, height | 正整数，指提供的拟合图像尺寸 |
| K | 3×3，像素单位；fx、fy 为正，最后一行为 [0,0,1] |
| T_camera_from_world | 4×4 世界到相机刚体变换，平移为 mm |
| distortion_model | V1 固定 none |
| A_fit_from_source_pixels | 3×3 像素变换，未裁剪时为单位阵 |
| K_source, source_width, source_height | 裁剪前相机内参及图像尺寸 |
| mask, valid_mask | 相对路径及 SHA-256 |

相机为右手系，+X 朝图像右、+Y 朝图像下、+Z 朝前；仅 Z>0 且在裁剪范围内的几何可见。像素数组为 [row=v, column=u]，深度采用相机 Z，单位 mm。2026-10-06 起显式支持 `pixel_convention`：旧清单未声明时为 `integer_centers`，左上中心 (0,0)、边界 -0.5；照片接口使用 `edge_origin_centers_at_half`，左上中心 (0.5,0.5)、边界 0。相机总表声明与单视图声明若冲突则拒绝。渲染、SemanticXY、射线及初始化均按约定解释 K，加载时不暗加减 0.5。

裁剪和缩放必须同步更新 K_fit = A_fit_from_source_pixels @ K_source。edge 约定直接使用 u'=sx*(u-x0)、v'=sy*(v-y0)。以下公式仅适用于旧 integer 约定：若裁剪左上角为整数 (x0,y0)，再以 sx、sy 按半像素中心规则缩放，则：

```text
u_fit = sx * (u_source - x0 + 0.5) - 0.5
v_fit = sy * (v_source - y0 + 0.5) - 0.5
```

V1 合成图像默认不裁剪，A 为单位阵。后续真实图像需先去畸变，不能把非线性去畸变并入该仿射矩阵。

SemanticXY 中屏幕坐标统一为 ((u-(W-1)/2)/D, (v-(H-1)/2)/D)，D=sqrt(W^2+H^2)，保持横纵比例。预测 XY 必须具有到几何的明确梯度；目标 XY 使用相同坐标约定。

视角数动态确定，消费 scene 中全部合格视角，不写死三视角或八视角。联合双颌拟合要求每颌至少在两个不同相机视角中具有可见牙齿；这只是输入下限，不保证全部参数可观测。覆盖率和观察方向另行报告。

重复 view_id 或相同 K、位姿、掩码与有效区的重复观测应拒绝，避免重复加权；不同相机出现同样的掩码不单独构成重复。损坏、尺寸不符、非法数值或标签冲突必须报错。全空但有效的观测保留为负向图像证据，不计入上述每颌可见视角下限。

## 6 逐牙语义和可见性

观测 mask 为单通道 uint8 PNG：FDI 表示对应牙齿，0 表示确认的非牙齿像素，255 表示忽略。valid_mask 为相同尺寸的 uint8 PNG，取值只能为 0/1；valid==0 当且仅当 label==255。未标注区域不得写成有效背景。

目标语义使用固定 28 通道，顺序为上表 upper 后接 lower；有缺牙或不可见牙也不重编号。模型输出对应 28 个牙齿通道，背景概率为 1 减牙齿概率和。牙龈是可渲染几何，在目标牙齿语义中为背景；两个牙龈的内部组件身份仍分别保留。

同一场景中所有存在牙齿与两个牙龈共同进行深度测试。V1 颌内采用 DMM 混合表面及其归一化组件权重生成软语义，两颌通过场景深度合成，不将两套单颌的全部组件跨颌重新归一化成一个 SDF。SDF 查询、表面提取和表面语义必须使用相同 presence。

逐牙独立硬标签网格可以用于调试，但需要另一个明确的 representation_profile，不能静默替换本规范的混合表面结果。历史 reference runs 不受这一新要求追溯约束。

有效背景支持惩罚多渲染出的牙齿；忽略区不支持这种惩罚。预测无可见像素而目标有牙齿时，必须保留其语义误差并报告 empty_prediction，不能因无法建立 OT 对应就跳过该目标或认定收敛。采用何种可见性恢复策略属于求解器实现。

静态开口不要求上下牙接触；如后续增加防穿插约束，其含义是排除不合理穿透，不能强制闭合咬合。混合 SDF 的数值不自动视为精确的毫米距离。

## 7 Latent 初值和统计 prior

每个存在组件保存训练均值 mu 和正则化协方差 Sigma_reg。默认使用原始训练病例中该牙确实存在的行计算统计，不重复计入镜像，不使用验证、测试或当前目标的拟合真值。每组件需至少 d_i+1 个有效样本，否则不得制作该版本的统计 bundle。

为使实现唯一，V1 固定 Sigma_shrink = 0.95*Sigma + 0.05*diag(diag(Sigma))；将其特征值下限设为 1e-6*trace(Sigma)/d_i，再重建 Sigma_reg。零总体方差、非有限值或分解失败使 bundle 校验失败。保存 mu、Sigma_reg、Cholesky 下三角 L、有效行数、病例清单摘要及对应 checkpoint hash。

拟合变量采用 z=mu+L*q，初值 q=0。优化器可另行使用数值步长控制，但不能把截断 q 或更换统计量作为未记录的实现细节。默认 prior 是逐存在牙 mean(q^2) 后再跨牙平均。该形式为二次正则化近似，不声称白化后的代码已经服从标准高斯，也没有表达完整的跨牙排列概率。

图像 loss 的像素/牙位/视角 reduction 和各项权重必须写入 resolved_config；view 使用等权平均，空牙位的处理必须显式记录。增加视角不能因为求和规模改变而意外改变 prior 的相对强度。SemanticXY 的软语义、坐标、采样、质量与未匹配规则已由 [匹配协议](semanticxy_matching_v1.md) 和 [算法配置](../configs/semanticxy_static_v1.json) 确定；这些算法选项不混入数据本身。完整拟合目标的 prior 权重已接入 FitConfig，数值仍需实验校准。

初始 G_up/G_low 来自配置、模型均值对掩码的粗配准或其他有记录的观测估计。正式 joint 模式不接收 GT 扰动初始化。若另做 oracle 诊断，应创建独立输入类型，明示其真值条件；不能以诊断包冒充普通 joint 输入。

## 8 新训练数据的最小清单

一个训练 manifest row 表示一个单颌扫描，不要求行内有另一副颌。必需字段如下：

| 字段 | 含义 |
| --- | --- |
| case_id, patient_id, arch, split | 扫描身份、患者分组、upper/lower、train/validation/test |
| embedding_row | 每颌按 case_id 字典序固定的连续行号 |
| source_geometry, source_annotation | 相对路径和 SHA-256 |
| source_unit, source_unit_to_mm, unit_evidence | 原单位、换算因子、单位依据 |
| canonical_reference_id, T_arch_mm_from_source_mm | 固定参考及源毫米坐标到 canonical 毫米坐标的刚体变换 |
| model_unit_mm, sampling_domain_model | 固定为 50；模型坐标中的实际采样域 |
| presence | 完整 14 牙位 Boolean 表 |
| samples, centers, surface_definition | 采样数据、质心、扫描/牙冠表面范围说明及 hash |
| augmentation_parent_id, augmentation_transform | 无增强则 null；否则明确来源与几何/FDI 变换 |

牙齿和牙龈采用扫描可支持的表面定义，不凭空把牙冠扫描当作完整牙根。不同裁切面、扫描基底和人工封口必须具有一致处理规则及单独标记。

模型输入以表面点、单位法线、标签和自由空间点为基本监督。表面 SDF 目标为 0；自由空间点用于 Eikonal/非零排斥，不能将未知距离列当作真实正负 SDF。组件采样根据完整 presence 分配配额。损失按有效病例、牙位和点数归一化，缺失组件不计入分母。

患者的上下颌、重复扫描和镜像必须在同一个 split。镜像在有记录的参考系中执行，交换左右 FDI、正确处理法线与面朝向，随后按同一 canonical 参考完成对齐；双颌配对增强需要共同的场景变换和双颌关系溯源。镜像产生的反射不允许伪装成刚体 pose。

## 9 模型和重建数据包

以下是新数据包的规范目录，现有旧运行目录无需迁移：

```text
scene_<id>/
  fit_input/
    manifest.json
    cameras.json
    masks/<view_id>.png
    valid/<view_id>.png
  model_bundle/
    upper/{model.json,weights.pth,latent_statistics.npz}
    lower/{model.json,weights.pth,latent_statistics.npz}
  initialization/
    parameters.json
  truth/
    manifest.json
    parameters.npz
    meshes/
    depth/
    heldout_views/
  outputs/<run_id>/
```

fit_input/manifest.json 必须包含 contract_id/version、scene_id、representation_profile、length_unit、两颌 presence、camera_file、view 列表、model_bundle 引用、initialization 引用及各文件 SHA-256。路径相对该文件解析，只允许引用本包的 fit_input、model_bundle 和 initialization 文件，不允许引用 truth、源训练样本或源扫描目录。

model.json 必须包含 arch、model_id、上游 commit、本地 patch/source hash、checkpoint hash、representation_profile、逐组件 ID 与 latent_dim、canonical_reference_id、model_unit_mm、采样域、统计文件 hash、固定牙龈码和训练数据/划分的摘要。weights 不包含病例训练 embedding 表；部署 bundle 仅提供统计量和固定 decoder。原始训练码表保留在训练实验中。

initialization/parameters.json 必须包含两个初始 T_world_from_arch、各存在牙的初始 q、初始化方法与来源，以及 model/statistics hash。默认 q=0；若由图像估计初始化，记录生成算法。两套 gum latent 从 model bundle 固定读取。

truth 保存生成用 latent、G_up/G_low、几何、深度、预留相机、装配来源与生成 seed。已知的拟合相机和 presence 是本任务明确允许的输入；GT 牙齿参数和几何不属于它。评估器可同时读取拟合输出和 truth；拟合器必须按允许清单加载输入，不能遍历 scene 根目录搜文件。

输出至少包含：最终参数与两颌 world-mm 网格、逐视角预测语义、目标/预测叠图、训练及统计来源 hash、可见性覆盖、优化日志、分项 objective、完成/失败原因。每次使用新 run_id；算法完成与事后几何评价为不同字段。

## 10 加载校验及结果解释

loader 与后续拟合器必须在优化前检查以下不变量：

1. 版本、profile、两颌标签与模型归属一致；各向量维数和文件 hash 一致。
2. 全部数字有限；刚体矩阵最后一行、R 正交性和 det(R)=+1 的误差均不大于 1e-5；单位、正尺度和 K 合法。
3. 图像尺寸、标签范围、valid/ignore 一致；缺牙 FDI 不得出现在有效目标中。
4. 视角 ID 无重复；每颌满足至少两个可见视角；全部有效视角被纳入或有明确的输入拒绝记录。
5. model bundle 的 prior 为正定、presence/statistics 有效；默认初始化不依赖 truth。
6. 采样域覆盖所需表面，越界、无等值面或退化几何必须报错，不静默截断。

几何误差报告 world-mm 表面距离，并另存 canonical 和整颌 pose 诊断；不在主指标计算前用 GT Procrustes 配准消除拟合误差。报告还须分开可见区域和未观察区域。V1 耦合 latent 仍可能与整颌 pose 互相补偿，因此 pose/code 精确恢复不作为唯一质量判据；高掩码 IoU 同样不能替代三维评价。数值通过门槛由新实验协议另行固定，不沿用旧 DMM 单位门槛。

## 11 下一步实现顺序

1. 已实现 manifest/model bundle loader、双颌 scene 组装、固定语义通道及跨颌深度合成接口。
2. 已修改官方 DMM 训练入口与模型查询：显式 presence、组件查询返回值、稳定小角度变换、device/采样域参数化、训练 canonical buffer 和版本元数据。新入口使用独立 manifest；原有 `-e` 入口保留旧数据协议。
3. 按本契约制作新的毫米尺度上下颌训练数据；确定下颌数据来源、单位证据及 canonical 参考后，才能形成完整 model bundle。
4. SemanticXY 匹配模块及人工张量测试已实现。继续接入联合深度可微渲染、软语义、XY 和隐式表面梯度；新能力使用独立运行验证，不重审已认可的旧 sanity check。所需源码修改与扩展完成后再进行正式 DMM 训练。
5. 形状/逐牙排列分解作为下一主版本，增加局部坐标和去除共同刚体模式的定义后再训练；不能将 V1 latent 改名为 shape latent 来迁移。

本文确定参数与数据语义，接口测试使用小型临时数据验证训练与组装链路。当前尚未提供合规的真实上下颌训练数据集、完成训练的新双颌权重或图像拟合结果；旧脚本未切换到该规范。

## 12 依据与决策来源

- [项目内官方 DMM](../third_party/DMM/README.md)，上游 commit 4081275ae13ad482b789fb57df4855049b1a46b8；[DMM 源码](../third_party/DMM/networks/dmm_net.py)和[组件网络](../third_party/DMM/networks/deform_net.py)用于确认耦合 latent、presence 和语义混合接口。
- [DMM 原论文](D:/文献/参数牙/DMM.pdf)，第 3–4 节与第 6 节，用于组件模型和下颌扩展依据；[官方项目页](https://vcai.mpi-inf.mpg.de/projects/DMM/)。
- [ICCV 2025 论文](D:/文献/参数牙/Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_ICCV_2025_paper.pdf)，第 4.1.1、4.2、5.1 节，用于表面梯度、软语义、SemanticXY 及跨视角共享参数依据。
- [现有 SemanticXY 方案](D:/WorkSpace/Dental/docs/multiview_teeth_reconstruction_semanticxy.md)提供任务背景；其中旧上颌阶段限定被本次用户明确指定的双颌目标取代。
- 毫米制、固定 50 mm 数值尺度、LPS 解剖轴、两独立位姿、固定牙龈码、prior 统计规则和包格式是本项目本次确定的工程选择，不是上述论文已经验证的最优配置。
