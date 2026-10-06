# 阶段一 Blender 仿真交接

数据契约：[blender_multiview_dataset_v1.md](blender_multiview_dataset_v1.md)，版本1.0.0。

- 新项目：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator`
- 项目交付索引：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator/DELIVERY.json`
- 阶段二提示：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator/docs/STAGE2_HANDOFF.md`
- Blender可打开场景：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator/scenes/pilot_static.blend`
- 新数据根：`D:/WorkSpace/Dental/data/MultiviewDentalPhotoSynthetic`
- 完成的运行：`runs/20261005T021823Z_dd9cac`
- 场景：`cases/01328DDN_static_51005`

在该场景下，阶段二推理读取 `input/manifest.json`，监督读取 `annotations/manifest.json`；
`truth`仅供独立评价。参考加载器在新项目 `loader.py`。

当前交付：1个开发病例、7个静态双颌视图、640×480、28牙。
实际标签/相机/深度/网格已导出并通过自动QC；助手已查看7视图叠加图。
可用于阶段二接口开发，accepted_for_training=false，仍保留人工验收。
基础profile为暴露牙列与牙龈，无嘴唇/脸颊遮挡；人工双颌装配不是患者真实咬合。
没有修改原Blender项目、原始扫描或DMM实现。

## 新增口唇遮挡场景

- profile：`static_oral_soft_tissue`
- 配置：新项目 `configs/pilot_soft_tissue.json`
- 新运行：`runs/20261005T120954Z_ff0e1e`
- 新场景：`cases/01328DDN_static_oral_soft_tissue_51006`
- Blender场景：新项目 `scenes/pilot_soft_tissue.blend`
- 最新交接索引：新项目 `DELIVERY.json`；原基础交付保存在 `DELIVERY_EXPOSED.json`

保留连接的脸、嘴唇、脸颊、舌头、口腔后壁；拍摄前一次拟合并固定几何。
七视图每张可见17–24牙，合并覆盖28牙，脸部/牙冠表面相交检查为0。
448个自由点重投影最大误差0.00008153px，深度到表面最大距离0.00007879mm，
牙齿与软组织深度标签核对错误均为0，278个产物哈希一致。
已检查RGB及标签叠加图；保留人工验收标记，仍为开发样例。
input/annotations格式不变，truth新增occluders和amodal；阶段二不得用GT组织标签生成正式推理valid。

## 最新：多患者消费者修正交付（2026-10-06）

最新运行：`D:/WorkSpace/Dental/data/MultiviewDentalPhotoSynthetic/runs/batch_20261006T061820Z_7e5c22`。
新项目 `DELIVERY.json` / `DELIVERY_BATCH_20261006.json` 指向该批。
详细交接：`D:/WorkSpace/BlenderProjects/MultiviewDentalPhotoSimulator/docs/BATCH_DELIVERY_20261006.md`。

- 2位新患者，继承train/val：013FHA7K（真实缺45牙）、FJS5HCDU（目标28牙完整）；未选择test。
- 每患者中等遮挡、牵拉、牵拉的严格无遮挡控制，共6场景、66张640×480照片。
- 两组配对不变量通过：同一保存场景，仅切换声明遮挡体render可见性；相机、灯光、材质绑定及牙列几何一致。
- 17/27在牵拉场景中合格视角分别由前7方向的2→5、1→4；中等遮挡两场景仍覆盖不足。
- 2472项产物哈希、4111个独立对应FDI深度点通过；最大点到面距离0.00005329mm，重投影0.00010408px。
- 几何检查已实现；val的24/25牙对存在5组三角面横穿候选，其他接触歧义保留复核。
- 7项规则/契约测试、Blender原生射线回归、2项真实样例副本变异测试通过，完成后的恢复跳过已完成场景。
- 全部66张RGB与6个逐视图FDI概览已检查。旧两组357项产物哈希复核未变。

状态 `COMPLETE_WITH_QC_GAPS`，所有accepted_for_training=false；未启动任何训练。
原生参考射线render可见性、闲置材质清理的两次修复及失败批均保存在repair_history.json。
最终批哈希复用已验证渲染并重新导出/审计，重复导出不计作新增照片或患者。
覆盖矩阵、逐病例碰撞范围报告、配对哈希、来源、预览、恢复进度与许可状态均从dataset.json与delivery_report.json追溯。
