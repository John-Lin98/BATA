# Full-context ESM2 feature reconstruction

`scripts/extract_embeddings.py` implements the frozen full-sequence layer-33
mean representation with configurable input/output paths. It reads ordered
mutation tokens and a reference context only, never assay labels. Model weights
are not bundled or automatically downloaded.

Obtain the original ESM2-650M checkpoint independently. The command checks its
SHA256 before loading. Context JSON contains exactly `sequence` (reference
protein) and `positions_1based` (ordered mutation positions). Mutation tokens
are canonical alphabet `ACDEFGHIKLMNPQRSTVWY`, codes 0–19, in frozen candidate
row order. Reference acquisition and alignment mappings are separate assets.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_embeddings.py --context inputs/context.json --tokens inputs/tokens.npy --checkpoint /path/to/esm2_t33_650M_UR50D.pt --numeric-mode fp32 --start 0 --end 29873 --batch 64 --output data/gb1_part0
```

Use `configs/PLM_RECIPE.json` for the exact benchmark mode, positions, batch
sizes, partition boundaries and target feature/slice SHA. GB1 used FP32 with
TF32 disabled. PABP, TrpB, HIS7 and GRB2 used three BF16 products for linear
layers with FP32 accumulation; attention, normalization and pooling remained
FP32. This is not full-model BF16 inference. The original acceleration kernel
is extracted without numerical changes in `bata/embedding_kernel.py`.

In `3xbf16` mode, eight fixed evenly spaced shard rows are compared with FP32
before extraction; a maximum absolute difference above 1e-4 aborts. This is a
numerical qualification, **not** a proof of frozen-cache bitwise equality.
Changing GPU/runtime, batch boundaries or model packaging may change arrays.
Always check reconstructed slice array hashes and frozen-run parity separately.

Install optional dependencies from `requirements-features.txt`. Only source,
CLI parsing and synthetic token/mean-pooling tests were exercised during this
publication packaging step; the expensive full feature rebuild was not run.
Use a new output directory per shard. No checkpoint training is performed.

After all shards are independently rebuilt, merge only verified exact matches:

```bash
python scripts/merge_embeddings.py --benchmark GB1 --shards data/gb1_part0 data/gb1_part1 data/gb1_part2 data/gb1_part3 data/gb1_part4 --output data/gb1_merged
```

The merger rejects missing/mismatched frozen partitions, modes, checkpoints,
contexts, token identities and array hashes before accepting output. It also
checks the complete NPY file SHA. A numerically close but nonidentical rebuild
is not silently substituted into the frozen reproduction protocol.

`PLM_RECIPE.json` contains only reproducibility settings and hashes; process IDs,
server paths, GPU assignments, queue state and internal receipts are excluded.
