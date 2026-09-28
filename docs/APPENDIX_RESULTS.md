# Auxiliary evidence and boundaries

All files in `results/appendix/` are mechanical projections of frozen results.
`PROVENANCE.json` binds source and output SHA256. Internal paths and execution
bookkeeping are excluded; numerical values, unfavorable outcomes and cohort
identities are preserved.

| Evidence | Files | Scope |
|---|---|---|
| Expert/fusion controls | `ablation_per_run.csv`, `ablation24.json` | 24 matched initializations per task; five arms, 360 runs |
| Transfer boundaries | `transfer_per_run.csv`, `transfer24.json` | HIS7/GRB2, 24 initializations each, five methods; descriptive transfer |
| Ensemble/acquisition capacity | `trpb_capacity_development_per_run.csv`, `trpb_capacity_development24.json`, `Fig_trpb_finalization.csv` | Separate TrpB development cohort; not main35 or sensitivity70 |
| Representation capacity | `representation_capacity24.json`, `Fig_capacity.csv` | Matched 24-run 650M versus 15B adaptations |
| Low-budget boundary | `pabp_low_budget48.json` | Separate 48-query protocol; author-released FolDE trajectories, not a new local rerun |
| Engineering speed | `speed_comparison.csv` | Warm-cache timing and engineering-only attempts; not performance-selection evidence |
| Curves and weights | `curves_per_run.csv`, `curves_summary.csv`, `weights_summary.csv` | Main/transfer R1 curves; weights retain explicit R1/D1 cohort labels |

Historical arm identifiers in these summaries identify only paper-used controls:
`c22_bata` = original BATA, `c0_fine_only` = matched assay-only expert,
`c22_equal_rank` = equal rank fusion, `c21_global_mse` = Global-MSE calibration,
`evolution_only` = prior-informed expert only. These are not imported historical
project frameworks.

The main table does not imply universal dominance: ALDE wins GB1, Global-MSE
beats BATA on PABP in the auxiliary 24-run control, FolDE wins the separate
48-query setting, and RF/REAP100 lead the HIS7/GRB2 descriptive transfer tables.
These boundaries remain in the publication assets. Do not rank methods across
different budgets, representations, initialization sets or fitness scales.

Main-method and expert-sufficiency development/confirmation cohorts must remain
separate. Fine-only M20 and G20 do not replace the frozen BATA M5 method.
