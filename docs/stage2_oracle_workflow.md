# 当前阶段二：直接使用 Blender 标签

2026-10-06，按用户最新要求：暂停逐牙识别、分割模型开发和训练。当前主线使用第一阶段现有可见FDI和tissue标注，后续再接 `D:/WorkSpace/Dental/SegmentAnyTooth`。

## 当前完成的流程

`dataset.json → input相机/RGB + annotations中的FDI/tissue → oracle_observations → 阶段三适配/覆盖诊断`。

入口：`scripts/stage2_oracle.py`，实现：`tooth_observation/oracle.py`。不用PyTorch、分割权重、SegmentAnyTooth或DMM权重即可打包与诊断。

- FDI原编号保留；实例由可见标签重算，不读取标注里的不可见牙清单。
- tissue101/102/103、采集无效和原始ignore转换为255；valid/confidence同步。牙龈100/104保持背景约定。
- confidence=1表示保留标注的单位权重，0表示忽略，不是网络概率；逐牙one-hot摘要也只是已知标签。
- 保留原分辨率、K、外参、A=I及半像素中心，无额外0.5偏移。
- 不读取truth下的网格、深度、amodal、真实位姿；没有用三维真值做初始化。
- 第三磨牙在阶段二保留，在阶段三转换ignore并记录排除像素。
- 每次写新目录，不修改源数据、QC标志或历史结果。

## 明确的oracle来源

观测schema升级为 `dental_tooth_observations/1.1.0`，`observation_kind=oracle`。
manifest包含annotation来源、哈希、逐视图FDI/tissue哈希；backend为blender_annotation_oracle且checkpoint为null。
旧prediction/fixture v1.0仍可读取。阶段三默认拒绝oracle，必须指定 `--allow-oracle`；适配产物和报告一直保留oracle标志。
不能将该结果用于声称RGB分割精度。现有预测评价命令默认也不接受oracle，避免将GT与自身比较获得虚假分数。

## 运行

```powershell
$stagePython = 'D:/WorkSpace/conda/envs/dmm/python.exe'

# 只打包，不作存在性假设
& $stagePython -X utf8 scripts/stage2_oracle.py '<run>/dataset.json' --output 'runs/oracle_<new_id>'

# 打包并执行阶段三诊断；此配置仅为28牙均存在的开发假设
& $stagePython -X utf8 scripts/stage2_oracle.py '<run>/dataset.json' --output 'runs/oracle_fusion_<new_id>' --presence-policy configs/oracle_presence_assumption.json

# 也可单独交给阶段三，使用自己明确给定的存在性策略
& $stagePython -X utf8 scripts/stage3_fuse.py --observations '<oracle>/manifest.json' --output '<fresh_fusion_dir>' --presence-policy '<policy.json>' --allow-oracle
```

默认打包入口不自动推断缺牙。示例policy是诊断假设，不是来源于真实存在性；正式拟合前需明确存在性配置。
例如013FHA7K的45未观察到，报告保留unobserved_assumed_present，不自动认定它存在或缺失。

## 已运行的真实Blender数据

本轮最终产物在 `runs/stage2_oracle_20261006_ready/`：

| 场景组 | 场景数 | 视图数 | 报告 |
|---|---:|---:|---|
| 原裸露pilot | 1 | 7 | [exposed](../runs/stage2_oracle_20261006_ready/exposed/report.json) |
| 原软组织pilot | 1 | 7 | [soft_tissue](../runs/stage2_oracle_20261006_ready/soft_tissue/report.json) |
| 2026-10-06新增批次 | 6 | 66 | [batch](../runs/stage2_oracle_20261006_ready/batch/report.json) |

共8场景、80张图已转换并由阶段三实际读取。每场景含oracle_observations和fusion两个独立目录，标签/置信度/实例/叠加图/相机及来源可追溯。
本轮早期 `runs/stage2_oracle_20261006/` 留存为初次联调记录，其中旧阶段三pending名称已被ready目录的新报告替代。

新增批次仍为COMPLETE_WITH_QC_GAPS：013FHA7K有接触复核项，FJS5HCDU有GEOMETRY_FAILED；全部accepted_for_training=false。打包只保留和传递这些状态，不构成新场景几何验收。原2个pilot同样保持development_reserved。

## 后续接入与当前剩余条件

阶段三原来固定报告RENDERER_PIXEL_CONVENTION已经过时：当前维护后的DMM已提供显式半像素支持及prepare-scene。本轮只更新外层报告为待DMM准备校验和均值初始化实际运行，不改DMM源码或训练指纹。
拿到合规模型包、明确存在性策略及初始参数后，阶段三可生成candidate，再由DMM `prepare-scene`正式验证；`fit-scene --initialize-mean`按其接口运行。当前没有模型包，因此没有生成可直接拟合的完整manifest或伪造三维重建。

SegmentAnyTooth的本地源码入口接收图像和upper/lower/left/right/front视图类型。将来接入时需明确11视图到该接口的路由、其FDI映射以及没有遮挡置信度时的有效区域处理。当前不运行、不修改该模型；届时替换观测来源为prediction，继续使用同一逐牙观测主体格式。

验证：阶段二30项测试通过；阶段三21项测试中20通过，1项Windows符号链接权限测试跳过。oracle测试检查标签精确保留、遮挡忽略、禁止读取3D truth/不可见实例清单、来源不混淆、源文件不变和第三磨牙衔接。
