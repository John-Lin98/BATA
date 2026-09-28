# BATA reference states

The associated reference states remain private and are not a public download.
BATA is a sequential optimizer, not a static pretrained network.

A local deterministic export of the prespecified **main35 / GB1 / group 0**
trajectory contains eight Ridge ensemble states: prior and task experts for each
of four rounds, five members per expert. All four rounds passed exact frozen-run
selection and score comparisons, including full candidate rescoring after NPZ
reload. No new scientific queries were made, and this trajectory was not chosen
for its performance.

The files are feature- and trajectory-specific reproducibility fixtures, not a
transferable protein model. Candidate-pool ranks and the round's arbitration
weight are required in addition to expert predictions. The main method remains
original M5; mechanism-analysis variants must be labeled separately.

NPZ files and internal export logs are not mirrored into GitHub. The private
reference states are outside the public release boundary. Local parity does not
establish a public hosted model release.

## Rebuild the prespecified states

After preparing the hash-verified GB1 DCA features and tokens and reproducing
`main35 / GB1 / group 0 / BATA`, use each replay-produced `observed_1.csv` through
`observed_4.csv`. The exact reference rebuild additionally needs the private
`EXPORT_RECEIPT.json`, which is not included in this public repository.

```bash
python scripts/export_reference.py \
  --dca-features prepared/GB1/dca.npy --tokens prepared/GB1/tokens.npy \
  --observed replay/observed_1.csv --round 1 \
  --reference-receipt reference/EXPORT_RECEIPT.json \
  --output rebuilt/round1
```

Repeat with rounds 2–4 and their corresponding observed files and new output
directories. The entry point fixes the cohort and initialization itself; there
is no seed-selection option. It refuses a mismatched kernel/manifest, calibration
weight or NPZ byte hash. Both experts must match before any output is written.
It records a rebuild receipt, not an upload receipt or new performance result.

Only already-revealed observations enter fitting. This script validates state
bytes and round calibration, not candidate selection afresh; selection parity
is covered by the original reference export receipt. It is scoped to the GB1
reference fixture, not a general checkpoint exporter for all BATA recipes.
