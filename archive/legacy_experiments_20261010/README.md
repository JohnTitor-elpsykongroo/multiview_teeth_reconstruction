# 早期实验脚本归档（2026-10-10）

当前入口保留在项目 `scripts/`：训练交接、CUDA 构建与回归核查、阶段二观测和阶段三接入。旧的 CPU 单颌前向、Pose-only、Shape-only、Joint、扰动批次及其评估/绘图工具，已经被当前双颌 DMM CLI 和训练流程替代，收进本目录 `sources.zip`。暂停的 RGB 开发 smoke、三份 AA 临时探针也一并归档。

归档保留原始相对路径和原始字节，不修改脚本中的路径与哈希规则。`manifest.json` 记录每份源码的大小、SHA256、清理前 Git 提交、当前 DMM 指纹，以及 smoke 检查点去重证据。历史实验的完整说明见 [HISTORICAL_README.md](HISTORICAL_README.md)。

## 恢复

在项目根目录执行以下命令，恢复归档文件到原路径。命令没有 `-Force`；如出现同名文件冲突，应先检查，避免覆盖后续工作。

```powershell
Expand-Archive -LiteralPath archive/legacy_experiments_20261010/sources.zip -DestinationPath .
```

压缩包不包含当前 README、模型实现、数据或实验结果。恢复后，旧脚本仍要求历史说明中列出的冻结权重、数据和运行目录；归档不把历史合成诊断升级为正式训练或真实照片验证。

## 清理边界

原始数据、已安装 Python/CUDA、DMM/nvdiffrast 源码、上游 Git 元数据、历史失败与成功报告保留。每颌 native smoke 的三个检查点逐 ZIP 成员比较 SHA256 后，确认未压缩内容完全相同，仅保留 `final.pth`；被删除的容器哈希和保留文件哈希记录在 manifest 中。其余删除项是可再生字节码、编译对象、已经验证安装的 CUDA 下载副本和 PDF 临时渲染。

详细结果见 [清理记录](../../docs/CLEANUP_20261010.md)，逐文件清单见本机 `runs/cleanup_20261010/inventory.json`。
