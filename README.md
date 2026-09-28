# Feedback-Calibrated Protein Optimization with Batch-Aligned Tail Arbitration

BATA 用已测样本的五折折外排序证据，校准 prior-informed 与 task-adaptive 两个预测器，再选择下一批蛋白变体。本仓库提供论文代码、冻结结果汇总、清单和复现说明。主方法为 original M5；Fine-only M20 和 G20 仅作机制分析。

BATA 是根据每轮 assay feedback 重新拟合的 sequential optimizer，不是通用 pretrained checkpoint。相关 reference states 仍为私有资产，并非公开下载；见[范围说明](docs/REFERENCE_STATES.md)。

## Quick Start / Reproduce results

安装核心和测试所需依赖，并核验已发布结果：

```bash
python -m pip install -r requirements.txt -r requirements-baselines.txt
python -m unittest discover -s tests
python scripts/verify_results.py
```

核验范围、配对统计口径与未覆盖项见[结果核验说明](docs/RESULT_VERIFICATION.md)。

神经基线和特征重建另需对应 requirements 文件，见[基线说明](docs/BASELINES.md)
和[数据来源](docs/DATA_SOURCES.md)。数据、权重和缓存由读者独立获取，不包含在仓库中。

单轮 CLI、固定初始化闭环和汇总命令见[复现说明](docs/REPRODUCIBILITY.md)。
当前已接入原 M5、四种 calibration objective、matched-expert controls、Fine-only/G20，
以及 ALDE/RF/EVOLVEpro650/REAP100 适配入口。主表、敏感性、校准、消融和 transfer
的冻结初始化清单已提供。代表性 frozen run 已通过 exact parity；这不等于所有任务、
方法和起点都在打包阶段重新执行。合成测试和结果核验也不代替独立科学审查。

## Results snapshot

已整理主 35-init、70-seed 敏感性、matched-24 calibration 和 Fine-only/G20 汇总，见 [结果说明](docs/RESULTS.md)。
HIS7/GRB2 transfer、PABP低预算边界、容量分析与负结果汇总见[附录结果](docs/APPENDIX_RESULTS.md)。
三张 TikZ 汇总图及数据映射见[图表说明](figures/README.md)；arXiv source package 包含最终论文图。保留正负结果，不混合 cohort。

## Citation

引用元数据见 [`CITATION.cff`](CITATION.cff)。作者为 Zefeng Lin、Xianyong Fang、Tianfan Fu、Xiaohua Xu。
