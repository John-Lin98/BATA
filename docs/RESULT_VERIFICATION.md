# Frozen result verification

Run `python scripts/verify_results.py` from the repository root after installing
the pinned requirements. This command reads published summaries; it does not
train a model, acquire labels, or modify results.

The checks include:

- SHA-256 against result provenance manifests.
- Main and sensitivity means and sample standard deviations, kept in separate cohorts.
- Appendix ablation, transfer and candidate-capacity summaries.
- Five-point best-so-far curves, endpoint values and the frozen Query-AUC formula.
- Fine-only formal results, independent TrpB70, gate development and G20 confirmation.
- Calibration summary means, sample SD, quantiles, average-tie ranks and mean ranks.
- All 18 published B96-versus-alternative paired contrasts, including confidence intervals.
- Shared initialization identities, query counts and algorithm seeds for calibration pairs.
- Warm-cache speed medians, exact old/optimized M5 metrics, and the incomplete-M100 boundary.

## Calibration statistical definition

For each of GB1, PABP and TrpB, each objective has the same 24 initialization
groups. Subtract the alternative's value from T-DCG-96 within each group. The
interval is the 2.5th and 97.5th percentile of 20,000 resampled mean differences.
The RNG is `default_rng(SeedSequence([20260916, benchmark_index, metric_index]))`,
with benchmark order GB1/PABP/TrpB and metric order Final/Query_AUC. The same draw
matrix is used across alternatives within that benchmark and metric, matching
the frozen analysis. Intervals are descriptive and not multiplicity-adjusted.

## Boundaries

These checks establish agreement with the published frozen aggregates, not
independent validation of scientific generalization. They do not reconstruct
every per-round diagnostic or verify a new training run. Metadata hashes are
checked against the included provenance, which remains tied to the frozen release
through its recorded source hashes.

The Fine-only TrpB70 versus BATA35 comparison is deliberately checked as a
cross-cohort mean difference, not a 70-pair effect. Main M5, Fine-only M20 and G20
remain distinct methods and cohorts. No winning-seed filtering is applied.

Tests also exercise failure cases: an altered result, duplicate group or mismatched
calibration seed must be rejected. Numerical checks use absolute tolerance 1e-14;
file hashes are exact.
