# 阶段二 → 阶段三：逐牙观测契约 v1.0.0

## 2026-10-06 用户授权的 oracle v1.1 扩展（当前主线）

直接使用Blender annotations中的现成FDI/tissue，按[oracle流程](stage2_oracle_workflow.md)操作。本扩展覆盖下文“未来oracle另行设计”的旧安排；RGB预测入口仍禁止读取监督。
主体像素/实例/相机格式保持不变，schema_version=1.1.0、observation_kind=oracle。增加oracle来源对象，包括annotations_manifest、annotations_manifest_sha256、confidence_semantics=unit_annotation_weight_not_probability、uses_3d_truth=false、presence_source=not_inferred、ignored_tissue_ids=[101,102,103]。
model.backend=blender_annotation_oracle、version=1.0.0、checkpoint_sha256=null；这一路没有识别模型。每视图增加annotation_sha256.fdi/tissue。
阶段三显式allow_oracle才接收，适配输出保持oracle来源；不访问来源引用路径。不读取3D truth、真实位姿、不可见牙清单。有效标注confidence=1仅为单位权重。
v1.0 prediction/fixture继续兼容。以下为历史v1.0定义：

状态：无数据初步开发接口。与 `blender_multiview_dataset_v1.md` 配套。
阶段二拥有 `tooth_observation/`、`scripts/stage2_observe.py`、`tests/test_stage2*.py`。
阶段三独立拥有 `observation_fusion/`、`scripts/stage3_*.py`、`tests/test_stage3*.py` 和其说明文档；不得修改阶段二或 DMM 训练源码。

## 边界

阶段二推理只读取阶段一 `input/manifest.json` 引用链中的 RGB、采集有效掩码及相机。
阶段三只消费本契约的 prediction manifest；来源引用仅用于追溯，不自动加载原始数据。
禁止读取 annotations、truth、真实存在性、真实牙弓位姿来改进预测/初始化。
监督训练和评价必须显式选择另外的入口。无权重时不得伪造成功预测；测试夹具必须标记 fixture。

## 布局与 manifest

每次调用指定一个尚不存在的新输出目录（推荐 `runs/<run_id>/predictions/<scene_id>/`）。
最终 `manifest.json` 最后写入，存在才表示完整输出，失败目录不自动复用。

```text
manifest.json
labels/0000.png
valid/0000.png
confidence/0000.npy
instances/0000.json
review/0000.png
```

manifest 必需字段：

```json
{
  "schema_id": "dental_tooth_observations",
  "schema_version": "1.0.0",
  "scene_id": "example",
  "length_unit": "mm",
  "pixel_convention": "edge_origin_centers_at_half",
  "world_axes": "X_patient_left_Y_posterior_Z_superior",
  "observation_kind": "prediction",
  "status": "OBSERVATIONS_READY_REVIEW_REQUIRED",
  "input_manifest": "D:/data/cases/example/input/manifest.json",
  "input_manifest_sha256": "sha256",
  "model": {"backend": "torchscript_semantic", "version": "user_version", "checkpoint_sha256": "sha256", "class_ids": [0,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]},
  "config": {"confidence_threshold": 0.6, "margin_threshold": 0.1},
  "views": [{
    "view_id": "front", "width": 640, "height": 480,
    "source_width": 640, "source_height": 480,
    "labels": "labels/0000.png", "valid_mask": "valid/0000.png",
    "confidence": "confidence/0000.npy", "instances": "instances/0000.json",
    "K_source": [[500,0,320],[0,500,240],[0,0,1]],
    "K": [[500,0,320],[0,500,240],[0,0,1]],
    "A_fit_from_source_pixels": [[1,0,0],[0,1,0],[0,0,1]],
    "T_camera_from_world": [[1,0,0,0],[0,1,0,0],[0,0,1,100],[0,0,0,1]],
    "distortion_model": "none", "distortion_coefficients": [],
    "quality": {"valid_pixels": 307200, "ignored_pixels": 0, "observed_fdi": [11]}
  }]
}
```

示例像素数量仅演示结构。实际实例和 quality 必须从最终标签重算。
`observation_kind` 只允许 `prediction` 或 `fixture`；阶段三正式入口默认拒绝 fixture，调试需显式开关。
未来 oracle 单独契约/入口，不能伪装 prediction。
资源路径相对 manifest 目录，禁止绝对路径、反斜杠、`..`、符号链接越界。
仅 `input_manifest` 来源定位字符串允许绝对路径；消费者不读取它。完整 provenance 可扩展记录输入各资源 SHA256。

## 像素和实例

- labels：PNG L uint8，0=确定非牙，11..18/21..28/31..38/41..48=FDI，255=忽略。
- valid_mask：PNG L uint8 0/1，严格等于 `labels != 255`。
- confidence：NPY float32 H×W，[0,1]，忽略位置严格为0；含背景置信度，分数未经校准。
- 网络内部类别映射必须保存在 model.class_ids，禁止把连续类别写成 FDI。
- 不确定编号、低置信度、采集无效区域设255；不能变成背景。
- 每颗牙按 FDI 聚合为一个观测实例，可含多个可见碎片。首版为语义基线，不声称解决重复编号实例。
- 第三磨牙保持原FDI；阶段三适配当前28牙DMM时设255、同步valid/confidence并记录排除像素数。

每个 instances 文件为列表（可为空）：

```json
[{"fdi":11,"jaw":"upper","visible_pixels":100,
  "bbox_xyxy_exclusive":[10,20,20,30],"confidence":0.8,
  "fdi_uncertain":false,"fdi_probabilities":{"11":0.8,"12":0.2}}]
```

confidence 是该最终FDI区域的像素置信度均值；fdi_probabilities 是该区域类别概率的平均值在32牙类别上重新归一化，包含全部32键。
它是语义网络的区域摘要，不是独立实例编号头的校准概率。无观测牙不补记录，不推断缺牙。
阶段三使用可见掩码即可重建轮廓，无需依赖 review 图片。空视图仍保留；全ignore场景应报告观测不足。

## 相机和几何

相机 +X右、+Y下、+Z前，世界 +X患者左、+Y后、+Z上；平移单位mm。
像素(row,col)中心=(col+0.5,row+0.5)，禁止额外平移0.5。
K = A_fit_from_source_pixels @ K_source。首版输出恢复至原始图像分辨率，A=I；网络内部缩放不改变输出K。
后续实际裁剪缩放必须同步元数据。阶段三遍历全部 views，不能硬编码视图数或读取GT补视图。

## 阶段三职责

1. 严格检查上述格式、尺寸、编号、实例一致性、相机刚体与 K/A 关系。
2. 保留置信度、统计多视图编号冲突/覆盖；不把未观察到等同于缺牙。
3. 显式接收外部配置的模型包和存在性策略；未知存在性不得借用GT。记录策略假设。
4. 估计上下颌初始刚体位姿。可先实现无模型接口、投影几何、约束诊断；需要均值模型而尚无权重时明确 pending，不生成假的正式拟合输入。
5. 生成兼容现有 `third_party/DMM/dmm/scene.py` 的 fit_input、独立 initialization 和报告。保持训练源码指纹不变。
6. 射线交会的掩码质心仅是粗启发，不声称同一解剖点对应。退化/不足应给可解释状态。

## 无数据开发验收

手工几何/概率夹具只验证协议和数值路径：动态视图、忽略传播、FDI映射、路径边界、相机变换、失败状态。
真实照片质量、分割/编号精度、完整初始化成功率需后续独立数据验证，不由夹具测试替代。

## 2026-10-05 实际 Blender pilot 接入：兼容扩展

新增RGB遮挡分支，观测schema仍为1.0.0；下列可选字段不改变既有必需字段及其含义：

- `model.occlusion_handling`: `rgb_learned_soft_tissue` 或 `explicit_exposed_only_legacy`；fixture可单独标记。
- `model.occlusion_tissue_ids`: 新模型为[101,102,103]。网络checkpoint格式升级为 `stage2_occlusion_baseline_v2`，与观测schema版本独立。
- `config.occlusion_threshold`: 默认0.3，只是未经数据调优的开发阈值。
- 每视图 `occlusion_probability`: 可选相对路径，NPY float32 H×W，[0,1]，记录从RGB预测的软组织概率，含所有像素。
- 每视图 `quality.predicted_soft_tissue_ignored_pixels`: 采集有效且遮挡概率达到阈值的像素数。

达到遮挡阈值的像素一律labels=255、valid=0、confidence=0。阶段三只消费这些最终结果即可；不需要读取新增概率图。
这是“该射线命中可能遮挡牙齿的软组织”的保守区域，不是某颗牙的精确amodal遮挡图；牙龈100/104仍按原约定处理。
GT tissue位置FDI仍为0，保持可见牙齿语义；训练单独监督遮挡头。推理不允许使用GT tissue或amodal生成有效区域。
旧无遮挡头checkpoint必须显式选择exposed-only兼容模式。全ignore输出代表观测不足，不能作为拟合成功。
