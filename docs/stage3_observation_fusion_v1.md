# 阶段三：观测融合与初始化基础 v1

2026-10-05。当前为无新 Blender 数据、无新 DMM 训练权重的接口开发。正式重建、均值模型对照片的位姿初始化和真实观测质量验证均未完成。

## 实现范围与入口

- `observation_fusion/contract.py`：独立校验阶段二 `dental_tooth_observations/1.0.0`，不依赖阶段二模块。
- `observation_fusion/geometry.py`：world-mm 投影、像素射线、最小二乘交会、独立同名三维点的刚体估计基础。
- `observation_fusion/pipeline.py`：28牙适配、覆盖与编号风险报告、存在性策略、DMM 候选组装。
- `scripts/stage3_fuse.py`：命令行入口。
- `tests/test_stage3_fusion.py`：独立合成协议与解析几何测试；不读取历史 runs。

输入仅为阶段二 manifest 引用的标签、valid、confidence、instances 与内嵌相机。`input_manifest` 只保留追溯意义，不会打开，即使其位置不存在也不影响推理。引用禁止绝对路径、反斜杠、父目录逃逸、Windows ADS、symlink 越界以及 annotations/truth。不会扫描数据根、读取 GT 或从旧运行寻找模型。

验证尺寸、PNG L/uint8、NPY float32、有限值、FDI、valid/ignore 严格等价、ignore confidence=0、32类区域概率、实例像素/bbox/均值置信度、quality、唯一视图、相机刚体、无畸变及 `K=A@K_source`。FDI 不确定实例应在阶段二已经成为 ignore，违反时拒绝输入。

全部 views 按原顺序处理；空背景视图保留负向证据，全 ignore 场景明确报告不足。重复 camera/K/mask/valid 观测拒绝；同相机不同图像可以保留，但不作为新增三角化相机。每颌至少两个不同可见相机只是 DMM 输入下限，不代表三维可观测。

## 使用

显式存在性策略 JSON 示例（由调用者在自己的新运行/配置目录保存）：

```json
{
  "kind": "assume_all_present",
  "origin": "外部实验方案明确假设28牙均存在；不是从照片或GT得出的存在性"
}
```

`explicit_external` 策略则额外提供 `presence.upper` / `presence.lower`，各含完整14个字符串FDI键与 Boolean 值，并注明 `origin`。已观察FDI与外部缺牙声明冲突时直接拒绝；不会删除观察来迁就策略。未知存在性没有隐式默认值。`assume_all_present` 只是显式实验假设，所有未观察牙仍标为 unobserved，不能宣称恢复了其个体形状。

```powershell
$dmmPython = 'D:/WorkSpace/Dental/teethDMM/.venv/Scripts/python.exe'
& $dmmPython -B scripts/stage3_fuse.py --observations <prediction_dir>/manifest.json --presence-policy <policy.json> --output <new_output_dir>
```

无模型即可运行。手工夹具默认拒绝，只有显式 `--allow-fixture` 可用于调试，输出持续标记 fixture。CLI 返回0表示诊断流程完成，**不表示初始化/拟合成功**；必须读取 `report.json` 的 `fit_ready` 与 `pending`。输出目录必须尚不存在，失败目录不能自动复用。

输出：

```text
report.json                    # 最后写入：诊断完成标记，当前 status=PENDING
adapted/manifest.json          # 独立 dental_fused_observations schema，含完整相机、实例与新quality
adapted/labels/0000.png
adapted/valid/0000.png
adapted/confidence/0000.npy
```

18/28/38/48 像素变为255，同时 valid=0、confidence=0；每视图逐第三磨牙记录排除像素数，不改阶段二源文件。适配后 manifest 内实例列表只含支持牙位，quality 重新计算；它不是阶段二 prediction manifest。输出逐资源 SHA256，报告保留原 manifest 与原资源 SHA256。像素 confidence 完整保留，但当前 DMM scene loader 不读取该权重；不能把“保存confidence”解释成“已接入加权拟合”。

## 几何与编号风险的含义

相机遵循 `X_camera=R@X_world+t`，mm，像素中心 `(col+0.5,row+0.5)`。直接用 K 与外参投影/反投影，绝不加减0.5修改 K。

逐牙报告所有可见视图、像素数、平均置信度、独立相机数、像素质心。每视图的区域概率中，本FDI与最强其他FDI差小于0.1时标记 `LOW_NUMBERING_MARGIN`。射线质心重投影 RMS 大于5px标记跨视图不一致；这两个值仅是未校准的诊断阈值，不是实验验收门槛，不触发重编号或删除视图。

交会少于两条射线、零基线、病态方向、交点在相机后方分别报告原因。`CENTROID_HEURISTIC` 仅表示线性计算非退化；不同视角的可见掩码质心不是同一解剖点，良好残差也不证明编号正确，更不能据此自动拟合牙弓。遮挡、可见表面变化和重复编号都可能影响这个指标。未实现独立实例的重复编号检测或编号纠正。

`estimate_rigid(source_mm,target_mm)` 提供无尺度 Kabsch 基础，拒绝少于3个点和共线点；只适用于外部明确同名三维点。没有自动把 mask 质心冒充模型中心，也没有把模型canonical坐标或单位矩阵当成估计成功的 world pose。

## DMM 候选组装与当前阻塞

可选 `--assembly-config <json>`：

```json
{
  "model_bundle": {
    "upper": "models/upper/model.json",
    "lower": "models/lower/model.json"
  },
  "initialization": {
    "method": "external_pose",
    "origin": "明确记录独立于truth的外部估计方法及来源",
    "T_world_from_arch": {
      "upper": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "lower": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  }
}
```

上述单位矩阵仅展示结构，不是可用初始化建议。CLI bundle 路径相对配置文件解析；Python API 使用调用者明确提供的路径。来源字符串是审计记录，不能由软件证明外部位姿没有使用GT，调用者必须保证来源独立。

两份模型包及外部位姿齐备、观察满足最低视图门槛时，复用实际 `dmm.bundle.load_bundle`、`dmm.scene.load_observations` 与 `ArchState` 验证模型、统计、来源绑定和参数维数。只复制模型JSON、权重和latent统计三类白名单资源，q 按模型维度置零，gum 继续由 DMM 固定训练均值。组装两颌独立位姿，不改变模型源码或指纹。

追加输出 `model_bundle/upper|lower/`、`initialization/parameters.json`、`fit_input/cameras.json`、`fit_input/observations/`、`fit_input/candidate_manifest.json`。DMM要求拟合资源位于 fit_input/model_bundle/initialization 允许目录，因此候选自包含复制适配观测，保留confidence sidecar。

**当前没有发布 `fit_input/manifest.json`，所有结果 fit_ready=false。** 原因：

1. 本轮固定的新观测契约是 edge origin、半像素中心；实际 `dmm/rendering.py` 的 CPU `_reference` 仍在整数网格采样，CUDA `clip_coordinates` 也含 `+.5*z` 的旧中心转换。旧双颌数据文档仍写整数中心。不能仅通过 loader 验证便认定渲染语义兼容。
2. 用户明确禁止相机暗加减0.5、禁止修改DMM源码。因此保存K不变，并通过候选文件名阻止现有 `load_scene` 的正式入口误用。后续需模型/渲染维护方明确版本兼容并验证投影；**不要手工重命名候选文件来跳过阻塞**。
3. 自动“模型均值 → 照片观测”的双颌位姿估计尚未实现，需合规模型包后继续开发和验证。目前只可提供诊断或明确标记 externally supplied 的独立外部位姿；没有伪造成功初始化。

无模型时报告 `MODEL_BUNDLES`，无独立位姿时报告 `INDEPENDENT_ARCH_POSES`，观测不足则报告 `INSUFFICIENT_OBSERVATIONS`。`MODEL_MEAN_TO_OBSERVATION_POSE_ESTIMATION` 记录自动初始化能力未完成，即使调用方提供位姿也仍保留该研发待办。

## 本轮验证与交接

```powershell
& $dmmPython -B -m unittest discover -s tests -p 'test_stage3*.py' -v
```

21项测试：20通过、1项Windows symlink权限限制跳过。覆盖7视图、第三磨牙同步ignore、外部存在性冲突、编号概率风险、空视图/all-ignore、路径/重复键、相机与实例契约、解析投影和射线交会、反射与共线退化、无模型pending、输出不可覆写。DMM候选组装测试使用临时未训练的小网络与显式测试统计，调用真实DMM加载/状态接口；不用于重建质量声明，也不运行旧sanity。

本轮开始的训练源码指纹：`3e8ef821758358cd054ffcb65f801829380e61ba49f408c61dffa5d5d15b4af4`。组装测试核对前后指纹一致。未修改阶段二契约/模块、根README、DMM源码或历史runs。

后续需：阶段二真实prediction输出、合规上下颌训练bundle、半像素渲染接口解决与独立测试、均值模型初始化算法及视觉检查。当前测试不证明分割编号精度、真实双颌位姿精度、三维形态质量或临床适用性。

## 2026-10-06 DMM 维护方后续集成

上述内容保留阶段三原交付的权限与状态。本次用户另行授权 DMM 源码修改后，渲染、SemanticXY 和射线新增显式 `edge_origin_centers_at_half` 支持，K 保持不变；新增 `fit-scene --initialize-mean` 候选初始化及 `prepare-scene` 入口，详见 [实现与验证说明](training_readiness_v1.md)。阶段三生产器和历史 candidate 仍保留原状态，不能直接改名跳过校验。

已有完整候选可执行 `dmm_cli.py prepare-scene <candidate_manifest.json> --output <fresh_root>`，由 DMM 验证 bundle、观测与半像素约定后，仅复制显式白名单引用，生成新 `fit_input/manifest.json`。`fit_ready=true` 表示输入接口可用，不表示分割、初始化或重建质量通过。缺少合规模型包或完整初值结构的候选仍不可消费；自动均值初始化的真实牙列恢复效果尚待验证。

## 2026-10-06 使用 Blender 现成标签

用户将当前观测来源改为Blender标签，阶段三加载器新增 `--allow-oracle` / `allow_oracle=True`，支持 `dental_tooth_observations/1.1.0` 的oracle来源。原prediction/fixture入口保留；oracle不会被自动视为prediction或fixture。
打包和实际80张图联调见[当前oracle流程](stage2_oracle_workflow.md)。阶段三仅读取已打包观测，不追溯访问annotations/truth。
外层pending与候选状态改为待DMM prepare-scene验证及均值初始化实际运行，移除固定的旧像素约定不支持结论；没有改动DMM源码或重命名旧候选绕过校验。
