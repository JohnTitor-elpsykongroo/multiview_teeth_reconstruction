# An Implicit Parametric Morphable Dental Model

### [Project Page](https://vcai.mpi-inf.mpg.de/projects/DMM/) | [Paper](https://arxiv.org/abs/2211.11402)

This is a PyTorch implementation of the Siggraph Asia 2022 Paper "An Implicit Parametric Morphable Dental Model".

## Installation

```
git clone https://github.com/cong-yi/DMM.git
cd DMM
conda create -n dmm python=3.9
conda activate dmm
pip install -r requirements.txt
```

## Data Layout

Our training data follows the dataset structure used in [DeepSDF](https://github.com/facebookresearch/DeepSDF) but adapt it for dental data with semantic labels and tooth centroids. The structure is as follows:

```
<data_source_name>/
    avg_centroids.txt
    SdfSamples/
        <instance_name>.npz
        <instance_name>.pkl
```

where `<instance_name>.npz` and `<instance_name>.pkl` are samples and centroids respectively.

Subsets of the unified data source can be reference using split files, which are stored in a simple JSON format. For examples, see `examples/splits/`.

## Training the network
```
python train_dmm.py -e ./examples/upper_dmm
```

## Pre-trained model
You can download the pre-trained models from [drive](https://drive.google.com/file/d/1nNyTdI59nUhmuoaT0YYG5CP1QsERJSUH/view?usp=share_link) and put it under the subfolder `examples/upper_dmm`.

## Evaluate the network
To generate the reference shapes:
```
python generate_meanshapes.py -e ./examples/upper_dmm
```

To reconstruct dental scans:
```
python reconstruct.py -e ./examples/upper_dmm -d ./test_data --iters 300 --lr 1e-3 -s examples/splits/test_split.json
```

Due to the protocol governing the usage of our clinical data, distribution to the public is not allowed. However, I have converted a publicly available [3D dental model](https://dentistry.co.uk/2022/03/11/3shape-model-maker-create-dental-models-from-3shape-trios-scans-in-minutes/) into SDF sampling data, complete with teeth numbering, for the purpose of conducting a simple test (put the [data](https://drive.google.com/file/d/18nNzqGEpRB5N6vki6jrDL6XEAf20nAqt/view?usp=sharing) under the subfolder `test_data`). This test data example, together with the reference shapes, serves as the reference for data alignment.

## Citation
If you find DMM useful for your research, please cite our
[paper](https://dl.acm.org/doi/10.1145/3550454.3555469):
```
@article{zhang2022dmm,
author = {Zhang, Congyi and Elgharib, Mohamed and Fox, Gereon and Gu, Min and Theobalt, Christian and Wang, Wenping},
title = {An Implicit Parametric Morphable Dental Model},
year = {2022},
issue_date = {December 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3550454.3555469},
doi = {10.1145/3550454.3555469},
journal = {ACM Trans. Graph.},
month = {nov},
articleno = {217},
numpages = {13},
}
```

## Acknowledgement
This code repo is heavily based on [DeepSDF](https://github.com/facebookresearch/DeepSDF), [SIREN](https://github.com/vsitzmann/siren) and [DIF-Net](https://github.com/microsoft/DIF-Net). And users can train their own network on the data from the [MICCAI Challenge](https://3dteethseg.grand-challenge.org/). Thanks for these great projects.

## 当前固定参考相似尺度数据集（2026-10-18，WSL）

以下从 `multiview_teeth_reconstruction` 仓库根目录运行。源码为本目录的
`data/fixed_similarity_dataset.py`、`training/fixed_similarity.py`；不修改同级独立的 `Dental/DMM`。
配置在 `configs/dmm_latest_v2/{upper,lower}.json`，相对指向同级
`Teeth3DS_DualArch_Preprocess/output/latest`，无需复制或转换61GB数据。

```bash
cd /mnt/d/WorkSpace/Dental/multiview_teeth_reconstruction
bash scripts/run_dmm_latest_wsl.sh validate upper
bash scripts/run_dmm_latest_wsl.sh validate lower

# 全部已导出条目（包含REVIEW）的逐文件SHA、数组、法线、牙号与PKL核验
bash scripts/run_dmm_latest_wsl.sh validate upper --cohort all_exported --verify-arrays
bash scripts/run_dmm_latest_wsl.sh validate lower --cohort all_exported --verify-arrays
```

入口复用 `.venv/bin/python`（Linux Python 3.10）；可通过 `DMM_PYTHON` 指定已有Linux环境。
已有环境直接使用，不运行会拒绝现有环境的旧初始化脚本。
GPU依赖沿用项目的 PyTorch 2.10.0+cu130、torchvision 0.25.0+cu130 与
`configs/training_handoff_v1/requirements.txt`。这与Windows虚拟环境是两个独立环境。

### 训练及恢复

下面为用户主动启动训练的命令，本次开发只运行独立小样本验证。8GB设备先运行一个牙颌；不要同时运行上下颌。

```bash
# 默认：计算PASS的训练条目；样本4096、batch=1、按需读取、内存缓存1例
bash scripts/run_dmm_latest_wsl.sh train upper --output runs/upper_latest_training_01
bash scripts/run_dmm_latest_wsl.sh train lower --output runs/lower_latest_training_01

# 明确选择全部导出训练条目（包含REVIEW）；不更改排除清单
bash scripts/run_dmm_latest_wsl.sh train upper --cohort all_exported --output runs/upper_latest_all_exported_01

# 有界优化器检查，可指定原例及镜像；保存中途位置，未完成epoch也能恢复
bash scripts/run_dmm_latest_wsl.sh train upper --output runs/upper_latest_check_01 \
  --case-id 01328DDN_upper --case-id 01328DDN_upper__mirror \
  --samples-per-scene 512 --max-steps 1 --epochs 1
bash scripts/run_dmm_latest_wsl.sh train upper --output runs/upper_latest_check_resumed_01 \
  --case-id 01328DDN_upper --case-id 01328DDN_upper__mirror \
  --samples-per-scene 512 --epochs 1 --resume runs/upper_latest_check_01/latest.pth
```

也支持 `.venv/bin/python third_party/DMM/train_dmm.py --config configs/dmm_latest_v2/upper.json --output <新目录> --device cuda`，根据配置schema自动进入新版入口。
所有输出目录必须是新目录，已有模型和运行结果不会被覆盖。配置中的100个epoch只是启动预算，不代表已运行或已验证足够训练。

- 默认计算PASS队列：上颌train872、val70、test52；下颌train330、val13、test19。
- `all_exported`：上颌train1340、val89、test84；下颌train1320、val86、test82。
- 只有train原例及其镜像建立embedding并进入优化器；不从原始文件夹glob、不恢复被排除7扫描或其增强，不把val/test混入训练。
- NPZ `surf/normals/pos/neg` 与同名PKL直接读取。表面/空间各占50%，空间pos/neg各取一半，保持当前 `DMM.forward` 的损失和权重；没有添加近表面采样、SDF符号监督、裁剪或重归一化。
- 保留固定参考与逐病例正逆相似矩阵，使用现有 `avg_centroids.txt` 作中心目标。模型坐标没有统一50mm比例，也不强制参考质心零均值/左右对称。REVIEW中的域外牙龈表面保持原值。
- 默认每100步覆盖一次 `latest.pth`；每个完整epoch也保存latest。每10个完整epoch保存周期checkpoint，同时保留best_train及本次终点final；未完成epoch的final会记录next_epoch/next_batch，不能冒称完整epoch。
- 同一状态的best/latest/final优先通过同盘硬链接共享存储；后续写入采用原子替换，不会改写已有best/周期状态。不支持硬链接时回退为普通复制。
- resume检查源码、输入hash、case-to-embedding对应、坐标契约和运行时；旧50mm/旧embedding checkpoint不能接入。源码变化后重新开始有界验证。
- 每个run保留小体积 `source_snapshot/`，用于恢复包含未提交改动的训练代码；不会复制数据集或虚拟环境。
- `best_train.pth`只按完整epoch平均训练损失选取。此入口尚未接入held-out latent fitting或best_val选模，不把训练loss当质量验收；val/test在这里只作输入核验，未参与模型更新。
- 新checkpoint保存为独立family，不能导出成现行固定50mm场景bundle。后续部署需使用逐病例逆变换或另外明确物理尺度的接口。

### WSL实测入口

```bash
.venv/bin/python -B scripts/check_dmm_latest_wsl.py --output runs/dmm_latest_wsl_check_20261018_01
```

该检查在真实WSL/CUDA上对上下颌各选一对原例/镜像，执行512点真实优化器更新、保存、恢复及第二步更新，另外测试配置中的4096点预算；不启动全量训练。结果见输出目录的 `WSL_VALIDATION.json`、各run的 `report.json` 和 `data_binding.json`。
