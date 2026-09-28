# 结果范围与复核

`main35_per_run.csv` 保留三任务 × 五方法 × 35 初始化的全部 525 条记录；`main35_summary.csv` 的均值与样本标准差由这些记录机械计算。

`sensitivity70_per_run.csv` 保留 GB1、TrpB 的五方法 × 70 初始化，共 700 条记录。论文的敏感性综合排名沿用 main35 的 PABP 列；不能把两个 cohort 的样本合并后计算一个均值。

`calibration/` 保留四种 calibration objective、三任务、24 matched groups 的 288 条记录及 paired effects、权重和诊断摘要。包括非显著和负向结果，不能仅按有利效应筛选。

运行 `python scripts/verify_results.py` 可检查发布文件 SHA 并重新计算主表与敏感性表的均值/标准差。来源 SHA 记于 provenance JSON；它是文件一致性证据，不等于独立科学审查通过。

Fine-only 和 G20 的正式汇总已放入 `variants/`，适用范围与 cohort 注意事项见 `EXPERT_SUFFICIENCY.md`。其他附录边界结果仍在整理，当前目录不是完整论文结果 release。
