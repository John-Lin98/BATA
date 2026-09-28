# PABP low-budget boundary

This appendix protocol is **3 batches × 16 = 48 queries**, in a fixed 18,854-row
half-world sampled from the 37,708-row PABP domain. It is not the main 480-query
protocol and does not replace original BATA M5. Negative results remain in
`results/appendix/pabp_low_budget48.json`.

`manifests/pabp_low_budget48.json` contains 10 frozen ordered worlds and 20
run entries (random-init and zero-shot-init for each world), without sequences
or fitness. Initial 16 indices are copied from each run's first sealed batch;
replaying them avoids changing the historical initialization. The zero-shot
scoring model is not silently replaced with a different checkpoint.

```bash
python scripts/select_low_budget.py --sim 0 --arm random_init --round 1 --output outputs/batch1.json
python scripts/select_low_budget.py --sim 0 --arm random_init --round 2 --observed data/observed16.csv --dca-features data/pabp_dca.npy --task-features data/pabp_plm.npy --output outputs/batch2.json
```

Reveal all 16 outcomes only after fixing a batch, then supply `index,fitness`
feedback for round 2 (16 observations) or round 3 (32 observations). Candidate
order follows the frozen half-world order. The selector uses five-fold OOF
calibration with tail size 16, the original two experts, and greedy rank fusion.
No labels are accepted in round 1. No full oracle state is distributed.

The adapter's two numerical functions were extracted from the frozen runner.
For preselected sim0, both arms' rounds 2/3 reproduce exact frozen indices,
scores and arbitration weights; see `manifests/LOW_BUDGET_PARITY.json`. This
checks the numeric adapter, not every CLI trajectory or every simulation.

After all three batches have been fixed, run the separate post-hoc evaluator:

```bash
python scripts/evaluate_low_budget.py --assay-csv data/PABP.csv --batches outputs/batch1.json outputs/batch2.json outputs/batch3.json --output outputs/endpoint.json
```

Only this evaluator receives the full-world labels. Percentiles use average
ties within the frozen half-world: Top10 counts percentiles >=0.90 and Top1
success means at least one percentile >=0.99 (not necessarily the single best
sequence). Regret is `(world_max - observed_best)/(world_max - world_min)` with
the original machine-epsilon denominator floor. Recomputing all 20 frozen BATA
endpoints reproduced the published two-arm means and Final sample SD to 1e-14.
FolDE numbers are from the authors' released trajectories, **not a local rerun**.
