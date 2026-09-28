# 运行入口与协议

当前入口读取两份按同一候选顺序排列的二维 NPY 特征，以及仅包含已测样本的 `index,fitness` CSV。`--task-features` 在 GB1 上应为 one-hot，其余已接入任务为论文冻结的 PLM 表示。不要混用行顺序、模型 revision 或 pooling。

```bash
python -m bata --benchmark GB1 --dca-features data/dca.npy --task-features data/onehot.npy --observed data/observed.csv --algorithm-seed 31 --round 1 --output outputs/batch1.json
```

输出父目录须事先存在；已存在文件不会覆盖。第一轮提供 96 个已测样本，第二至四轮分别提供 192、288、384 个。每次输出完整 96 条候选后，外部实验系统才可返回该批结果。CLI 没有完整 fitness landscape 的输入参数，不代替实验系统检查传入数据是否确实已经测量。

输出中的 `selected_scores` 是预测的融合排序分数，不是 measured fitness。每轮拟合所用已测索引排序、fit seed、Thompson RNG 和原数值运算保持冻结实现规则。

## Matched calibration

`manifests/main35.json`、`sensitivity70.json` 和 `matched_objective24.json` 分别提供各 run 的初始 96 条索引及算法随机种子。每个 benchmark/group 的初始索引已跨方法核对一致；三个 cohort 不混合。索引只对指定数据行顺序有效，数据准备文档和行身份验证仍须补齐后才能据此完整复现。Manifest 不包含 fitness 或原始序列。

默认 `--objective T-DCG-96` 为 BATA 原主方法。论文中的机制对照分别指定 `G-Rank`、`T-Uniform-96` 或 `T-DCG-192`。对照必须共用 benchmark、初始 96 条身份、algorithm seed、特征、selector 和 480-query 预算，不根据中途结果更换配置。

## Expert-sufficiency variants

论文的 matched-expert 对照可用 `--method matched_fine`（原 M5/任务 recipe）、
`evolution_only`、`equal_rank` 和 `global_mse` 单轮运行。
`matched_fine` 不等于 `fine_m20`：前者保持原任务专家容量和 selector。
`equal_rank` 固定两专家候选百分位排序各占一半；`evolution_only` 使用进化专家排序。
`global_mse` 原实现采用交叉拟合残差去偏和 Ledoit-Wolf 协方差权重，融合的是
原始预测值，不是 percentile rank；其 fold seed 常量为9917，而 BATA 为22022。
这些差异按论文冻结对照保留，不能把它描述成仅更换一个损失的完全匹配实验。
该组控制已接入 CLI 和 `--cohort ablation24`，方法名使用清单里的 `BATA`、
`Matched-fine`、`Evolution-only`、`Equal-rank`、`Global-MSE`。清单机械抽取360个
arm-run，含三个任务各24组共享起点；无标签。GB1 group0 的五种方法均已完成
四轮冻结重放：选点、成员、selected scores、历史及Final/Query-AUC精确一致，
见 `manifests/GB1_CONTROLS_PARITY.json`。这不覆盖其他任务或其余起点。

`--method fine_m20` 使用冻结的 20-member task-only 实现；`--method g20` 使用固定 0.20 阈值。`g30`、`g40` 仅供重放论文开发阶段阈值对照，不能作为 confirmation 的替代方法。GRB2 的 BATA 映射到 PABP 冻结 recipe；Fine-only M20 在各任务都使用 TS96，包括原 BATA 使用 greedy 的任务。不要把这种强单专家对照误称为只删除一个模块的完全 matched 消融。

这些入口的验证仍需完整 frozen-run replay；合成单轮 parity 不替代该验收。

## 单个冻结闭环复现

HIS7/GRB2 外部任务使用 `--cohort transfer24`；240个run对应两任务、五方法、
各24组起点。`group` 按论文冻结汇总编号，不从原服务器文件名推算 seed。
这些结果不与主任务 cohort 合并。PLM任务专家使用 `--task-features`，RF/ALDE
使用 `--tokens`；表示身份仍须按数据文档核验。

`scripts/reproduce_run.py` 只接受已发布主表、敏感性、matched calibration、
matched-expert、transfer 和 Fine-only/G20 清单中的具体 benchmark/group/method，
不生成新起点或扫描超参数。例如：

```bash
python scripts/reproduce_run.py --cohort main35 --benchmark GB1 --group 0 --method BATA --assay-csv data/GB1.csv --dca-features data/dca.npy --tokens data/tokens.npy --output outputs/main35_gb1_bata_0
```

此命令会真正执行四轮拟合，并非 smoke；请按资源条件显式运行。完整 assay CSV
仅由 evaluator 使用，先核验冻结 SHA；selector 子进程只接收当时已经揭示的
96/192/288/384 条反馈。每批 96 条身份全部固定并验证无重复后才揭示反馈。
这是程序数据流隔离，不是对同一 OS 用户的安全沙箱。输出目录不可覆盖；中断
运行保留原文件，不自动重启或续跑。生成的逐轮反馈不得提交到论文仓库。

指标原样采用冻结实现：五个批次结束时的历史最好值，Final 为最后一点，
Query-AUC = `(h0 + 2*(h1+h2+h3) + h4)/8`，对应查询 96–480 的归一化梯形面积。
runner 记录输入特征 SHA，但不因此认定任意输入特征符合论文；必须先按数据文档
核验表示和行身份。新入口已对预先固定的 main35 GB1 group0 BATA 完成四轮重放：
选点、成员、selected scores、融合权重、完整历史及 Final/Query-AUC 与冻结记录精确一致。
证据见 `manifests/GB1_CLI_PARITY.json`。这不代表其他任务或 baseline 均已验收。

```bash
python scripts/aggregate_replays.py --metrics outputs/main35_gb1_bata_0/metrics.json --output outputs/summary
```

汇总脚本拒绝跨 cohort 混合、重复 run、非完整480查询及不一致指标。单run的样本
标准差留空而不是伪造为0；只汇总提供的文件，不保证35/70个起点已全部覆盖。

## 本地检查

```bash
python -m unittest discover -s tests
python scripts/verify_results.py
```

合成 smoke 测试不构成真实数据结果；结果 SHA 与汇总复核也不等于独立科学审查。
五任务的数据准备、正式初始化 manifests、baseline/variant 入口均已发布；精确
frozen-run parity 只覆盖文档明确列出的代表性运行，不能外推成每个任务和每个起点
都已重新执行验收。第三方资产仍须由读者按固定 revision/SHA 独立获取。
