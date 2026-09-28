# Public release boundary

`PAPER_ARTIFACT_INDEX.csv` inventories regular, non-ignored publication files
with relative paths and SHA256. Regenerate it with
`python scripts/build_artifact_index.py` after changes. The index excludes itself
to avoid a recursive hash; it is versioned by Git. Classification does not imply
that every scientific or licensing gate has passed.

`EXCLUDE_INDEX.csv` lists excluded asset classes and the reader-facing replacement.
Excluding raw logs does not exclude unfavorable results: paper-used negative
results remain in the appendix summaries and corresponding protocol code.

Engineering tests, mechanical result verification, and representative frozen-run
parity establish their stated scopes; they do not rerun every experimental
campaign. Private reference states and internal receipts are outside this
repository. BATA-owned code is Apache-2.0. Included ALDE-derived code retains
its MIT notice; REAP remains an external AGPL-3.0 dependency; raw datasets,
MSAs, pretrained weights and caches are not redistributed.

## Reusable history-to-publication workflow

1. Freeze paper scope and identify the authoritative result releases.
2. Map every paper table/figure to summaries and the minimum necessary code.
3. Build a new repository with an explicit allowlist; retain the history repo unchanged.
4. Replace private paths with inputs, preserving arithmetic, seeds and protocols.
5. Bind data/models by revision and hashes; distribute only assets with appropriate rights.
6. Verify representative frozen trajectories and regenerate result summaries mechanically.
7. Scan, stage exact paths, commit a clean publication history, and read back the public repository.
8. Preserve the original private history separately with its provenance.
