# Prior-informed DCA feature construction

The shared arithmetic in `bata/dca_features.py` is extracted from the frozen
PABP/HIS7/GRB2 builders. Their three numerical function ASTs were identical;
the publication version only derives the region width from input and removes
server-specific import paths. GB1/TrpB used these same arithmetic functions.

Rebuilding requires independently obtained aligned region sequences and frozen
candidate order. Regions (1-based inclusive) are GB1 39–54, TrpB 183–228,
PABP 126–200, HIS7 6–211, GRB2 159–214. GB1/TrpB embed their four mutated sites
in the complete reference region before calculating features. Do not replace
the full region with the four mutation columns. MSA preparation must retain
the original sequence multiplicities and frozen filtering/alignment policy.

`build_dca.py` accepts ordered two-dimensional integer NPY arrays, alphabet
`-ACDEFGHIKLMNPQRSTVWY`. Gap is code 0; candidate regions have no gaps.
These are **not** the 20-letter task tokens used by `--tokens`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_dca.py --region-msa inputs/msa_codes.npy --candidate-regions inputs/candidate_region_codes.npy --output data/rebuilt_dca
```

This invokes GPU redundancy counting and CPU mean-field fitting, not an assay
optimization experiment. Redundancy theta is 0.2 and pseudocount is 0.5. Preserve
the frozen source arithmetic, including its local-field sign convention.
The output stores per-position float32 energy decompositions and the fitted
DCA parameters; do not commit these caches to the publication repository.

Install py-mfdca independently from https://github.com/utdal/py-mfdca, review its
upstream terms, and use a revision whose `dca/dca_functions.py` SHA256 equals
`d6b518bacf77ba8aedf5d6f30822c8128aaa726f55f597365a8bb370a71ffcb1`.
The frozen environment used numba 0.58.1 / llvmlite 0.41.1. No upstream
py-mfdca source is redistributed in this repository. The CLI rejects a different
arithmetic source before fitting.

For PABP/HIS7/GRB2, `configs/MSA_RECIPE.json` binds each original full-length
ProteinGym alignment filename and SHA to its frozen encoded region array.
Obtain the alignment independently; no raw MSA is redistributed here.

```bash
python scripts/prepare_msa.py --benchmark PABP --alignment inputs/PABP_YEAST_full_11-26-2021_b07.a2m --output inputs/pabp_msa
```

The script changes dots to gaps and uppercases letters, slices the fixed region,
rejects regions with over 50% gaps or noncanonical characters, and retains row
order and duplicate sequences. It rejects unexpected source or output hashes.
All three complete reconstructed arrays matched frozen inputs exactly:
PABP 7,852 × 75; HIS7 39,863 × 206; GRB2 23,970 × 56.
This is an input-reconstruction check, not a new scientific result or a complete
DCA refit validation.

GB1/TrpB are also supported by `prepare_msa.py`, with a separate frozen rule:
remove lowercase insertion characters and dots before slicing, reject over-50%
gap/noncanonical regions, group identical regions in first-occurrence order,
then expand each group by its original multiplicity. The complete arrays match
the frozen fitter inputs exactly: GB1 205 × 16; TrpB 160,618 × 46.
`MSA_RECIPE.json` records the SGPO upstream URL at commit
`73953f887a195e64474d15a0c893086b72493a33` and required content SHA. The HF tree's
LFS hashes at this revision equal the frozen source hashes for both alignments.
The SHA is mandatory rather than advisory. The upstream HF repository has no
license metadata or license file; **no redistribution permission is inferred**.
Obtain these alignments separately under applicable upstream terms; the BATA
release contains neither raw alignments nor their sequence-encoded copies.
TrpB's alignment reference has 389 residues, whereas the assay parent has 397;
the frozen builder checked equality of their 183–228 design regions. These are
not interchangeable full-length contexts. Upstream acquisition/license review
is still required before declaring the entire from-download pipeline complete.
Compare arrays and frozen-run selection parity separately; compressed-container
hashes can differ even for equal arrays.
