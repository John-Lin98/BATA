# 数据与特征身份

本仓库不再分发原始 assay 数据、第三方模型权重或 embedding 缓存。`configs/DATA_IDENTITY.json` 记录冻结运行使用的文件 SHA、候选数量、特征形状和有序候选身份摘要，不包含 fitness 或序列。

初始化 manifests 中的索引只在候选行顺序完全一致时有效。不能简单重新按序列或 fitness 排序后继续使用这些索引。冻结候选范围为 GB1 149361、PABP 37708、TrpB 159129、HIS7 496137、GRB2 63366；这不是对原始数据集全量大小的通用声明。

## 已有冻结特征的输入适配

```bash
python scripts/prepare_features.py --benchmark GB1 --domain /path/to/public_domain.json --tokens /path/to/tokens.npz --dca /path/to/dca_features.npz --plm /path/to/embeddings.npy --output data/gb1_cli
```

此命令读取并核验 manifest 中的 SHA、行索引、有序候选身份及形状，产生 `dca.npy`、`tokens.npy` 和不含路径的身份回执。PLM 文件保留原位，不复制。输入不包括 fitness 文件。输出目录必须不存在。

GB1 或 one-hot baseline 可以向主 CLI 传入 `--tokens data/gb1_cli/tokens.npy`，替代 `--task-features`；one-hot 按原 float64 规则分块生成，不必在磁盘保存全量稠密 one-hot 缓存。PLM 方法仍使用 `--task-features /path/to/embeddings.npy`。

这一步仅解决已核验冻结特征的格式适配，不代表已完成从上游原始数据开始的重建。

## 精确复现与重新提取特征

固定已核验缓存、软件版本和随机数规则，是重放冻结轨迹的前提。特征提取记录包含经误差检查的混合精度矩阵计算；即便使用同一 PLM 名称，换成另一数值路径也不能预先保证 feature SHA 或候选顺序逐位相同。重新生成特征的验证与 frozen-run exact parity 应分别报告。

当前数值核心与测试在 Python 3.9 环境运行，依赖版本见 `requirements.txt`。这份文件不包含后续特征提取及全部 baseline 所需依赖。

## 尚待完成

各上游下载来源、固定 revision、再分发条款和原始数据预处理/特征重建脚本仍须补齐。当前不能仅凭哈希清单声称外部用户已能从零复现。不得用未经核对的下载文件替代上述冻结输入。
