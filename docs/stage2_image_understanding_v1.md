# 阶段二：RGB → 逐牙观测初步实现

**2026-10-06用户调整：本页分割训练/预测路线暂缓。当前运行入口见[直接使用Blender标签](stage2_oracle_workflow.md)，后续再接SegmentAnyTooth。本页保留历史实现说明。**

日期：2026-10-05。状态：两组 Blender pilot 已完成开发输入验收，25项测试通过；模型增加RGB软组织遮挡分支。尚无可用分割权重，未验证牙齿识别精度。详见 [Blender验收与本轮改进](stage2_blender_acceptance_20261005.md)。

## 入口与边界

- 上游：[Blender 照片契约](blender_multiview_dataset_v1.md)。推理唯一入口是 case 的 `input/manifest.json`。
- 下游：[逐牙观测契约](tooth_observations_v1.md)。输出供 `observation_fusion` 阶段三消费。
- 本实现位于 `tooth_observation/`，入口 `scripts/stage2_observe.py`，不修改 `third_party/DMM`。
- 阶段三已在当前项目独立对话“阶段三：逐牙观测融合与初始化初步开发”启动，模型 GPT-6 Astra、推理 medium；thread ID `01a109de-1467-7cd0-ad71-7aa40dc3c420`。

## 实际实现

1. 惰性逐视图读取 RGB、有效掩码，相机契约/刚体/像素中心/尺寸校验；动态视图数量。
2. 路径真实解析后检查根目录包含关系，禁止绝对引用、父目录跳转和符号链接越界；禁止输入目录内写预测。
3. 小型卷积编码器、上下文分支、跳接和33类语义输出：背景＋完整32牙FDI；v2增加单独的软组织遮挡头，tissue101/102/103为正类。
4. RGB按sRGB uint8/255处理，不线性化、不镜像。网络内部保持宽高比缩小，双线性插值 logits 恢复原分辨率后 softmax，输出相机和分辨率保持不变。
5. 最大类别概率与第一/第二概率差联合筛选；低置信度、编号歧义、采集无效区域、RGB预测遮挡概率达到阈值的区域输出255，confidence为0。置信度未经校准，遮挡阈值默认0.3尚未调优。
6. 同FDI可见碎片聚合成一个实例，记录像素数、右下排他bbox、平均置信度和32牙概率摘要；不推断真实存在性。
7. 第三磨牙18/28/38/48原样保留，由阶段三适配DMM时忽略。
8. 新目录写标签、有效掩码、float32置信度、实例和叠加预览；最后写manifest。失败输出没有完整manifest且不会自动覆盖重跑。
9. 显式监督训练入口按患者划分，用逐类平衡像素交叉熵，验证集macro tooth IoU选best，保存各epoch权重、日志及输入SHA256。

这是便于换模型的工程基线。小网络没有专门的实例分离/全局牙弓编号头；重复FDI的错误不会被自动修正；不宣称可解决真实口内照片的编号和遮挡。后续用同一输出契约接入更强实例分割与编号网络。
本轮只读取了旧 TeethPhotoSeg 的说明与接口作参考，没有导入其权重或混用其训练/评价入口。

## 环境

Python 3.10+，NumPy、Pillow；模型训练/推理另需PyTorch（支持 `weights_only=True`、`nearest-exact`）。
本机验证使用 `D:/WorkSpace/conda/envs/dmm/python.exe`：Python3.12、NumPy2.5.2、Pillow12.3、PyTorch2.11.0+cu128，测试在CPU上执行。

以下命令在项目根目录执行，路径占位符需替换；没有数据时不会自动生成训练样本或下载权重。

```powershell
# 只核验某个场景的推理输入，不加载标注
& 'D:/WorkSpace/conda/envs/dmm/python.exe' scripts/stage2_observe.py validate-input '<case>/input/manifest.json'

# 数据到位且完成验收后，先核验训练/验证划分
& 'D:/WorkSpace/conda/envs/dmm/python.exe' scripts/stage2_observe.py training-preflight '<run>/dataset.json'

# 后续显式启动分割基线训练；output必须不存在
& 'D:/WorkSpace/conda/envs/dmm/python.exe' scripts/stage2_observe.py train '<run>/dataset.json' --output 'runs/stage2_train_<new_id>' --device cuda --epochs 10 --long-edge 512

# 已训练权重推理；output必须在源case之外且不存在
& 'D:/WorkSpace/conda/envs/dmm/python.exe' scripts/stage2_observe.py infer '<case>/input/manifest.json' --checkpoint 'runs/stage2_train_<new_id>/best.pth' --output 'runs/stage2_predict_<new_id>/predictions/<scene_id>' --device cuda

# 人工夹具和跨阶段接口测试
& 'D:/WorkSpace/conda/envs/dmm/python.exe' -m unittest discover -s tests -p 'test_stage2*.py' -v
```

训练入口必须同时有非空 train/val，且所选场景 `accepted_for_training=true`。患者不能跨任何划分；若有dmm_split必须一致。
`test` 和 `development_reserved` 不加载其照片和标注；历史开发样本不能用于独立泛化结论。未验收的pilot可用validate-input和显式supervision函数调试，训练入口不放宽门槛。
监督加载显式访问annotations的FDI和tissue文件；不读取其不可见牙存在性列表、depth或truth。FDI标签保持“可见牙齿”定义，遮挡头独立监督；v2训练要求train/val均包含有效软组织正、负像素，不能只用裸露牙列声称已学会遮挡。
首版训练不支持续跑/分布式/学习率调参，也不包含正式测试集评测；中断后保留已保存epoch，下一次使用新输出目录。

## 模型扩展接口

`observe(input_manifest, output, predictor, ...)` 的predictor只接收RGB uint8 H×W×3。v2返回字典：`fdi_probabilities`为float32 33×H×W归一化概率，`occlusion_probability`为float32 H×W的RGB遮挡概率。
属性 `kind` 为 prediction或fixture；`metadata` 包含backend、version、checkpoint_sha256和固定顺序class_ids。
生产CLI默认支持原生 `stage2_occlusion_baseline_v2` checkpoint；旧 `stage2_semantic_baseline_v1` 仅在显式 `--allow-no-occlusion-head` 的裸露牙列兼容模式加载，其后端仍返回原数组。
生产predictor必须声明occlusion_handling；推理没有annotations/tissue参数。未训练fixture可为0个优化step，但正式supervised_training checkpoint必须有正数training_steps。
自定义后端可以在Python中实现同一接口；固定类别順序由`tooth_observation.CLASS_IDS`定义。
人工规则夹具位于测试文件，hash指向夹具生成器源码，默认不能进入正式预测/阶段三。

## 验证证据与剩余工作

25项测试通过，包括推理监督隔离、FDI/ignore/置信度一致性、第三磨牙保留、动态视图、K原样传递、实例bbox/面积、路径拒绝、重复视图、相机刚体、失败不落完成manifest、患者隔离、未验收数据拒绝、RGB模型反向更新及保存/加载、遮挡忽略传播、旧权重兼容限制以及全ignore不能抬高评价分数。
跨阶段测试实际调用阶段二导出后，由阶段三 `load_observations` 读取，再运行第三磨牙适配，验证valid/confidence同步。
训练测试仅对临时人工色块运行1个epoch、3个优化step，临时权重随测试清理；其IoU不是牙齿照片识别指标，没有正式分割训练发生。

接下来的顺序：

1. 已读取两组pilot、核验相机/尺寸并完成照片/标签视觉检查；保留development_reserved标记。
2. 收集经过验收、患者分离的train/val，明确FDI覆盖；训练并检查分割/编号失败类型。
3. 用固定病例和相机比较预测观测与oracle观测，解释上游错误对重建的影响。
4. 再接更强模型、多视图编号约束和置信度校准，不以接口测试替代上述评价。

阶段三已发现现有DMM参考渲染器与新照片的像素中心约定存在待核对差异；本模块始终保留半像素中心和原K，不在图像入口暗加/减0.5。正式拟合前由阶段三记录并解决兼容问题。

## 新增验收、数值联调及评价命令

```powershell
# 只读验收；可显式读truth抽查深度点到对应牙网格距离，不修改源QC
& 'D:/WorkSpace/conda/envs/dmm/python.exe' -X utf8 scripts/stage2_audit_dataset.py '<run>/dataset.json' --output 'runs/stage2_audit_<new_id>' --geometry

# 仅development_reserved数据：随机权重前向/反向、导出和阶段三读取；0次优化更新
& 'D:/WorkSpace/conda/envs/dmm/python.exe' -X utf8 scripts/stage2_development_smoke.py '<run>/dataset.json' --output 'runs/stage2_smoke_<new_id>'

# 显式预测评价：不读取三维truth；fixture需额外--allow-fixture
& 'D:/WorkSpace/conda/envs/dmm/python.exe' -X utf8 scripts/stage2_evaluate.py '<prediction>/manifest.json' --input-manifest '<case>/input/manifest.json' --annotations-manifest '<case>/annotations/manifest.json' --output 'runs/stage2_eval_<new_id>'
```

audit的geometry模式另需trimesh；只对对应FDI网格作稀疏最近表面检查，不替代完整射线遮挡/碰撞证明。
评价逐FDI IoU、牙齿二值IoU、编号错误像素、有效区域覆盖、真实牙齿被忽略比例和软组织被忽略比例；预测ignore仍计真实牙齿漏检，只有GT ignore/采集无效像素被排除。
当前评价只接受原分辨率、A=I的预测，并核对输入SHA256与相机；单场景报告不宣称独立泛化评测。
