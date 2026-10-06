# Blender 多视图照片数据契约 v1.0.0

状态：阶段一与阶段二开发接口。schema_id = `dental_multiview_photo`。

项目：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator`。
数据根：`D:/WorkSpace/Dental/data/MultiviewDentalPhotoSynthetic`。
每次生成写入新的 `runs/<run_id>`，不覆盖历史数据。数据集索引为该运行的 `dataset.json`。

## 范围

同一个 scene 内上下颌、牙齿、牙龈几何在所有照片中保持不变，仅移动相机及相机灯。
相机内外参已知，针孔、无畸变、无景深、无运动模糊。视图数可变，消费者遍历 views。
首个 profile 为 `exposed_dental`: 扫描牙齿和保留的牙龈，无脸部遮挡；口腔软组织场景另作 profile。
来源为独立 Teeth3DS 扫描，保留原始 FDI。人工双颌装配明确标记 synthetic_articulation，非患者真实咬合。

新增 profile `static_oral_soft_tissue`：保留带材质的脸、嘴唇、脸颊、舌头及口腔后壁。
软组织仅在拍摄前拟合一次，所有视图共享固定几何。禁止通过近裁剪移除遮挡体或为每个视图移动颌骨。
运行配置：仿真项目 `configs/pilot_soft_tissue.json`。基础配置和历史运行保持独立。

## 文件布局

```text
dataset.json                       # 场景索引、患者、划分、QC，不是推理输入
cases/<scene_id>/
  input/
    manifest.json                  # 推理唯一入口；不引用 annotations/truth/build
    cameras.json
    rgb/<view_id>.png               # RGB uint8，sRGB/AgX 显示图
    valid/<view_id>.png             # L uint8，0/1；采集有效性，首版全1
  annotations/
    manifest.json                  # 训练/评价监督，显式选择才允许加载
    fdi/<view_id>.png               # L uint8 原始 FDI 可见区域，0=非牙，255=忽略
    tissue/<view_id>.png            # L uint8: 0=牙或空背景;100=扫描非牙组织;
                                   # 101=脸/唇/颊;102=舌;103=后壁;104=补建牙龈
    instances/<view_id>.json        # fdi, jaw, visible_pixels, bbox_xyxy_exclusive
  truth/
    manifest.json                  # evaluation_only=true；来源/位姿/单位证据/装配
    meshes/<fdi>.obj                # 三角面，已求值世界坐标，mm
    meshes/<jaw>_gingiva.obj
    geometry/<view_id>.npz          # depth_mm:相机Z深度;normal_camera;valid=几何命中
    occluders/*.obj                 # 新profile的已求值软组织世界mm网格，仅评价
    amodal/<view_id>/<fdi>.png       # 单牙独立投影 L uint8 0/1，仅评价
  qc/report.json
  qc/overview.jpg
```

input/manifest.json 的必需字段：
`schema_id`, `schema_version`, `scene_id`, `length_unit="mm"`,
`camera_file="cameras.json"`, `views=[{"view_id", "rgb", "valid_mask"}]`。
路径统一 `/` 分隔、相对所属 manifest 所在目录；不允许绝对路径、`..`、符号链接越界。
input 下不得放入真实网格、深度、牙齿存在性、真实牙弓位姿、GT 掩码或初始化。

cameras.json 包含 `schema_id`, `schema_version`, `scene_id`, `views`。
每个相机记录：`view_id`, `width`, `height`, `K` (3x3),
`T_camera_from_world` (4x4), `distortion_model="none"`, `distortion_coefficients=[]`,
`pixel_convention="edge_origin_centers_at_half"`, `length_unit="mm"`。
相机右手坐标：+X右、+Y下、+Z前；世界：+X患者左、+Y后、+Z上。
`X_camera_mm = R @ X_world_mm + t_mm`，`[u,v,1] ~ K @ X_camera_mm`。
图像左上边缘为(0,0)，像素(row,col)中心=(col+0.5,row+0.5)。禁止暗加/减0.5。
裁剪缩放后必须保存源K及 A，使 K_new=A@K_source；首版无裁剪缩放。

annotations/manifest.json 含相同 schema/scene 标识、`supervision_only=true`，
`views=[{"view_id","fdi","tissue","instances"}]`。
每视图 instances JSON 为列表，含该场景所有存在牙齿；不可见牙 bbox=null、visible_pixels=0。
这些存在性记录仅供监督/评价，禁止推理利用。bbox 为 [xmin,ymin,xmax,ymax]，右/下界不包含。
首版硬标签按像素中心的首个表面命中定义，RGB 边缘存在抗锯齿混色；标签禁止双线性缩放。
标签0包括可见牙龈/软组织及空背景。遮挡组织不等于牙齿缺失；不可见区域不生成虚构可见标签。
软组织profile继续使用相同的v1.0.0输入和监督格式；truth新增occluders/amodal引用，不进入input引用链。
组织101/102/103覆盖某颗牙的amodal投影，表示该像素首个命中为软组织且该牙投影被遮挡；
QC统计按视图取这些像素的并集，不能把像素数理解为独立牙齿数或临床遮挡率。

## 阶段二输出接口（由阶段二开发）

保存到独立 `predictions/<scene_id>/`，不得写回 annotations：

- `manifest.json`: schema_id=`dental_tooth_observations`, schema_version=`1.0.0`, scene_id,
  input_manifest（来源引用）, model/version/config 来源，以及动态 views。
- 每个 view 记录 view_id、labels、valid_mask、confidence、instances，以及
  `K_source`, `K`, `A_fit_from_source_pixels`, `T_camera_from_world`。
- labels: L uint8，0背景，FDI为原编号，255忽略。
- valid_mask: L uint8，0/1；必须与 labels!=255 一致。
- confidence: float32 HxW NPY，[0,1]，忽略区域为0。
- instances: 列表，FDI、可见bbox、置信度和编号不确定性；不得从 GT 补齐牙齿存在性。

面向DMM的有效区域需要区别组织遮挡与可信背景。阶段二可使用tissue监督训练遮挡识别，
推理时必须从RGB预测遮挡/不确定区域，再生成255/valid=0；不能读取GT tissue生成正式预测有效掩码。
现有FDI监督中软组织位置仍为0，保持“可见牙齿分割”定义。嘴唇遮住的牙不能作为缺牙监督。

DMM当前支持固定28牙：11..17,21..27,31..37,41..47。
数据格式保留第三磨牙18/28/38/48的真实编号；进入当前DMM的适配器将不支持的区域设255并记录，不能改成背景。
FDI掩码本身即可区分实例，无需把FDI映射成连续类别后写回数据；网络内部映射需显式保存。

## 划分、验证与使用权限

dataset.json 每个场景记录 scene_id、patient_id、split、input_manifest、annotations_manifest、
truth_manifest、qc_report、accepted_for_training。split按原始患者分配，所有视角/光照/装配变体继承同一划分。
已有DMM数据划分时必须继承并核对上下颌一致。历史反复检查的患者标记 development_reserved，不能宣称独立测试。
只有通过自动QC且完成必要视觉验收，才可将 accepted_for_training 标记true；初始示例默认为false。
阶段二开发可显式加载 review_required 的小样例作接口调试，不能自动收进正式训练集。

自动QC：文件/尺寸/模式/标签范围、实例像素和bbox、患者隔离、静态几何指纹、源文件SHA256、
相机与Blender投影一致(<0.002px)、标签射线与Blender场景独立检查、真值网格投影及深度一致性。
背景深度=0；geometry.valid不同于input.valid，禁止用前者充当推理有效掩码。
三维真值只由评价端读取。真值逐牙掩码进入重建仅限独立命名的oracle诊断路径。

## 阶段二对话交接提示

先阅读本契约和仿真项目README，按input/manifest.json实现动态多视图加载器，
显式区分inference与supervised模式；接入RGB逐牙分割/FDI编号基线，输出上述predictions接口。
不要等待大批渲染或DMM新权重。优先用首个pilot样例验证读写、几何变换、标签编号和遮挡语义。
不得读取truth改进预测或初始化，不得按图像随机划分患者，不得修改阶段一生成器及已生成数据。

## 多患者批量扩展（2026-10-06）

输入schema与半整数像素中心不变。新增批量入口为仿真项目 `batch_runner.py`，
配置 `configs/batch_pilot_v1.json`，完整规则见仿真项目 `docs/BATCH_PROTOCOL_V1.md`。

- `plan.json` 在渲染前冻结患者、train/val继承、候选选择seed/规则/清单hash、场景和质量门槛。
- 患者划分同时检查镜像/增广记录；01328DDN继续development_reserved，test不用于本轮调参。
- 每scene新增 `qc/coverage.json`、`coverage_matrix.csv`、`collision_report.json`、
  `scene_invariants.json`、`independent_audit.json`。均不加入input引用链。
- 可见面积、bbox、amodal面积和比例扩展到annotations实例记录，缺牙由原始扫描presence证明。
- 根 `pairs/<pair_id>.json` 记录严格配对控制；只有声明遮挡体的render可见性不同。
- `diagnostic_control=true`仅表示配对实验控制，不代表常规口内照片。
- dataset索引同时保留quality_status和failed_scenes；不能只按文件存在就批准训练。
- 修复批可通过 `reused_renders.json` 追溯已冻结的正常场景渲染，重新导出时保存新源码快照。

本轮开发门槛固定为可见像素≥32、可见/amodal图内比例≥0.02，并至少有一对合格相机
中心基线≥20mm、朝向差≥10度。64像素只是旧统计分档。上述门槛不代表临床或学习能力校准。
未达到门槛保留COVERAGE_INSUFFICIENT，源缺牙为SOURCE_MISSING。

几何检查覆盖牙—牙与牙—可见非牙网格，tol=0.02mm；区分横穿相交、源牙龈接口接触、其他接触歧义。
非闭合扫描的表面检查不提供体积无穿插证明；失败/歧义病例不自动移除或升级训练许可。
阶段二推理仍只能从RGB预测组织遮挡，采集valid保持原语义；Blender不为旧DMM的像素偏移暗改K。
