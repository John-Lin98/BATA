# External data and software boundaries

This is an asset inventory, not legal advice. A code repository's license does
not automatically license a separately hosted dataset or model weight file.
No third-party raw datasets, MSAs, pretrained weights, or embedding caches are
included in this release. Users obtain them from the original providers and
verify the content identities in `configs/`.

| Asset | Evidence / terms observed | Release treatment |
|---|---|---|
| ALDE TrpB assay | Zenodo 12196802 metadata: CC-BY-4.0; member SHA verified | Source attribution and extraction script only |
| ProteinGym v1.3 assay tables | Zenodo 15293562 metadata: MIT; three member SHAs verified | Source attribution and extraction script only |
| SGPO GB1/TrpB alignments | HF `jsunn-y/SGPO` commit `73953f887a195e64474d15a0c893086b72493a33`; no card license or license file; LFS SHAs match frozen inputs | Pinned download URLs and processing code only; no inferred permission to redistribute |
| ProteinGym PABP/HIS7/GRB2 alignments | Frozen names and source SHAs verified; distribution terms/source archive still to be documented separately | No raw or encoded MSA redistributed |
| GB1 ERA assay/reference | Fixed upstream commit and exact source hashes; see `DATA_SOURCES.md` | URLs only; no raw sequence/assay file redistributed |
| TrpB PDB 8VHH reference | Entity 1, version 1.0; remote FASTA equals frozen file | Download helper and attribution only |
| ALDE neural network implementation | MIT notice retained in `licenses/ALDE-MIT.txt` | Necessary attributed implementation included |
| REAP implementation | External pinned checkout; AGPL-3.0; see `BASELINES.md` | Local adaptation wrapper only; upstream not vendored; external installation does not waive its terms |
| py-mfdca | External arithmetic source bound by SHA; see `DCA_REBUILD.md` | Not vendored |
| ESM2 650M checkpoint | External checkpoint bound by SHA; see `PLM_REBUILD.md` | No weights or feature cache included |

This public code release does not redistribute assets whose terms are unresolved.
In particular, a model-card license must not be inferred from this inventory;
the private reference states require a separate provenance and redistribution review.
