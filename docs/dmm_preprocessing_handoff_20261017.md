# DMM 数据预处理新一轮交接

## 已执行的用户决定

2026-10-17，用户明确要求将以下七份扫描直接从训练输入排除，原始文件留档，其余数据留待新一轮研究预处理；新对话使用 GPT-6 Sol。

- NQJYZS60_lower
- THGBYHX3_lower
- 5J9O0LG3_lower
- X3EW8O0A_lower
- 00OMSZGW_lower
- 6L4N5MA2_lower
- Q5GSR6N2_upper

排除单位为具体颌扫描，不连带排除配对另一颌。派生增强样本同样排除。原始 OBJ、JSON 留在原路径，已核验七份源数据哈希与审计一致；没有删除、搬移或修改原始数据。

## 新版输入入口

根目录：`D:/WorkSpace/Dental/multiview_teeth_reconstruction/runs/dmm_training_inputs_filtered_20261017_01`

- `exclusion_policy.json`：用户批准的排除规则，下一轮必须执行。
- `excluded_sources_archive_index.json`：被排除源文件的原路径、哈希、诊断指标，原位留档。
- `preprocessing_candidates.json`：1793 份有标注、未被此次排除的源扫描；仅为候选，不代表全部通过训练准入。保留历史排除状态、患者划分及路径。
- `missing_annotation_holdout.json`：100 份缺标注扫描，暂不进入有监督预处理候选池。
- `upper/manifest.json`、`lower/manifest.json`：实际过滤后的旧采样数据清单，采用绝对路径，重新连续编号 embedding_row。上颌去除2条（1原例及增强），下颌去除12条（6原例及增强）。原始7份对应含增强14条。
- 每颌 `r4_training_case_ids_filtered.json`：过滤后旧训练 ID，含增强上颌1288、下颌1264；原始训练扫描共1283减7为1276。
- 每颌 `legacy_filtered_pilot.json`：基于历史 pilot 配置重新指向过滤清单的副本，仅供接线参考，尚未执行完整训练加载/优化检查，其他历史引用仍须核验。不要直接开跑。
- `summary.json`、`delivery_manifest.json`：计数及文件哈希。

历史数据目录、历史训练配置、已训练权重和部署包均未改动。排除在本次新版本清单中生效；任何后续训练必须使用新清单或按排除策略重新构建，不能重新 glob 历史样本绕过排除。已有旧模型不会因为清单变化自动改变，也不能直接用新 embedding 索引恢复旧 checkpoint。

## 任务主线与下一轮范围

目标：改善 DMM 训练数据质量，解释和改善新版初始模板劣于旧版的问题。用户要求抓主要矛盾，不再钻研少数异常扫描的复杂封底修复。

下一轮先研究并规划：尺度、固定参考牙列对齐、牙号、原表面法线、表面采样、空间采样。每阶段需要可视化验收；先做小规模验证，不覆盖现有数据，不直接全量预处理或训练。

用户认可代理网格的唯一目的：为随机空间点提供稳定的内外分类，重点保护牙齿及附近牙龈。训练表面、法线应来自原始观测面；人工封底不是训练目标表面。远端裁切边缘不是主要目标。保持训练/验证/测试按患者隔离，不能因重建清单改变划分；此次未撤销此前缺牙、FDI等独立准入条件。

先查阅：

- `docs/dmm_preprocessing_replan_20261016.md`
- `D:/WorkSpace/Dental/teeth3DS_to_DMM`：旧版预处理实现，实际读取核实，不凭记忆。
- `D:/WorkSpace/Dental/teeth-reconstruction-research/library/papers/A04_An_Implicit_Parametric_Morphable_Dental_Model_2211.11402v1_main.pdf`
- `D:/WorkSpace/Dental/teeth-reconstruction-research/library/papers/A07_Teeth_Reconstruction_and_Performance_Capture_Using_a_Phone_Camera_conference_open_access_main.pdf`
- `runs/dmm_preprocess12_20261016T1815_r1`、`scripts/align_preprocess12_fixed.py`：12例诊断及固定参考对齐工作，使用前核验实际产物状态。

## 全量筛查证据与限制

`runs/dmm_raw_risk_census_20261017_01/index.html`：1900份源扫描，1800份可做FDI分区筛查，100份缺标注。原始网格MeshLab自交候选面积，牙体加距牙顶点2mm内牙龈面质心为关键区。基准00OMSZGW为0.4414%，严格达到基准5份，接近或超过0.4%为7份，一半基准14份。用户此次批准的是7份清单，不是14份。数值风险不等于已验证空间点分类错误，也不证明其余数据都健康。

## 已收口的失败路线

00OMSZGW 曾试过平面侧壁、Poisson、Meshmixer Make Solid、高分辨率调参及分区混合修复。最终 `runs/dmm_hybrid_proxy_00OMSZGW_20261017_01/index.html` 未通过质量验收，不作为训练输入。不要重复此路线。封闭不等于保真。早期47对相邻面候选属于共享边误报，不能说47处真实重叠。分区牙体接口曾被当成洞而错误封闭，MeshFix也损伤后牙。当前明确改用少量异常扫描排除策略。

本轮只完成排除清单与交接，没有执行新预处理或训练。
