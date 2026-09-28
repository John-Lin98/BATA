# Frozen comparison recipes

The local adaptations below preserve the frozen comparison recipes. Naming an
adaptation does not imply that every setting of the upstream publication is
reproduced. They use the same measured observations and candidate order as BATA.

| CLI method | Task features | Fit | Acquisition |
|---|---|---|---|
| `rf_onehot` | One-hot | 200-tree random forest; squared-error criterion; round-derived seed | Greedy 96 |
| `evolvepro650` | Frozen ESM2-650M mean features | 100-tree random forest; Friedman-MSE criterion; seed 1 | Greedy 96 |
| `alde` | One-hot | Five DNNs, 30/30 hidden units, CPU float64 fit, 300-update cap, original training-loss stopping | TS 96 |
| `reap100` | Frozen ESM2-650M mean features | 100 RankReg MLPs, original 90/10 splits and stopping, CUDA | UCB 96; coefficient .5 in rounds 1–3, 0 in round 4 |

Both forests use bootstrap samples, all features per split, and one CPU job.
Ties are resolved by ascending candidate index. The EVOLVEpro-labelled row is
the paper's local adaptation, not a copy of the upstream repository.

Use `python -m bata --method rf_onehot` or `--method evolvepro650`, with the
remaining arguments described in `REPRODUCIBILITY.md`. `--task-features` must
contain the representation in this table (not necessarily BATA's task-specific
representation). The current shared CLI also requires a same-order DCA file;
these two baselines do not use DCA in fitting or selection.

ALDE's used `DNN_FF` class is extracted from the MIT-licensed upstream code
(copyright 2025 Jason Yang), with the full notice in `licenses/ALDE-MIT.txt`.
The local fit adapter preserves five independent initializations, fixed member
splits, training-label max scaling and training-loss early stopping; held-out
labels are not used for ALDE stopping. Source: https://github.com/jsunn-y/ALDE.

Install `requirements-baselines.txt` for these optional neural baselines.
For REAP, obtain https://github.com/zyan-y/REAP separately, review its AGPL-3.0
license, and check out `96c69e94b3b39c33ffb70ac394bd06022b4f035f`.
Run `python scripts/verify_reap_dependency.py /path/to/REAP` before installing
that checkout with `python -m pip install -e /path/to/REAP --no-deps`.
The manifest records all 12 frozen source-file hashes. No REAP source is
vendored here. Optional upstream package dependencies must also be installed.
This external-dependency arrangement does not waive upstream license obligations.

The ALDE extraction passed exact original-vs-extracted weight/score comparison
on synthetic data; this is not a new benchmark result or a full frozen-run
baseline replay. REAP training has not been rerun as part of packaging.
The staging repository remains incomplete pending full asset and release QA.
No raw upstream datasets, model weights, or repositories are bundled.
