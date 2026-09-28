# Expert-sufficiency 结果范围

Fine-only M20 与 G20 是机制分析，不替换 BATA original M5。`results/variants/` 保留正式汇总中的全部数值，包括不利结果。

- `fine_only_m20.json`：GB1/PABP 各 35、HIS7/GRB2 各 24 的 118 场，以及另行复用的 TrpB 70-seed 汇总。
- `gate_development.json`：阈值筛选的开发结果，不能称为独立确认。
- `g20_confirmation.json`：五任务、三方法、各 10 个独立初始化，共 150 场，阈值不在该 cohort 上调节。

## 来源中的 cohort 注意事项

正式 Fine-only 汇总中的 TrpB 行把 Fine-only 的 70-seed 均值与 BATA 主表的 35-init 均值并列。其 `Delta_vs_BATA_Final` 不能解释为匹配 70-seed 的配对效应。这里原样保留数值，并显式说明差异；不据此重新计算置信区间，也不静默修改论文数字。需要同 cohort 的 BATA 均值时应独立查看 `sensitivity70_summary.csv`，完整配对分析仍须对应初始化身份。

G20 fresh10 的负结果同样保留：跨任务整体排名未超过 BATA。不要只引用它在 TrpB 上的改善。
# Frozen run entry points

The replay runner additionally accepts `fine_m20_formal118`, `gate_development5`,
and `g20_confirmation10` as separate cohorts. Method names in these manifests are
`Fine-only-M20`, `G20`, `G30`, `G40`, and `BATA` where applicable. Development
thresholds must not be pooled with fresh confirmation results. The 118-run
formal cohort covers GB1/PABP/HIS7/GRB2; `fine_m20_trpb70` contains the separately
released FINAL70 TrpB cohort: 70 Fine-only and 70 matched REAP runs, formed from
24 old + 35 fresh + 11 new initializations. It is not the earlier `robust70` cohort.
Development contains 100 variant entries plus 25 explicitly marked reused BATA
controls, not 125 new runs. These per-run CSVs preserve the actual frozen metrics;
they are not substitutes for the full source summaries already included here.

`manifests/FINAL70_ADAPTER_PARITY.json` records synthetic comparisons against the
separate frozen TrpB capacity implementation used by FINAL70. Round 1 and round 4
checks cross both 8192-row and 32768-row prediction block boundaries; the full
20-member predictions and ordered selections are exact. Wrapper metadata names
differ. This is adapter validation, not a claim that all 70 frozen trajectories
have been rerun with the publication code.
