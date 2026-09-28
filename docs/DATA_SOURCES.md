# Upstream data acquisition

Raw assay CSVs, sequence collections, alignments and pretrained model weights
are not redistributed by this repository. Obtain them from their respective
providers and review those providers' terms. A code license is not automatically
a dataset redistribution license.

## PABP, HIS7 and GRB2

Use the [ProteinGym v1.3 release](https://zenodo.org/records/15293562), not an
unversioned current release. The official archive and metadata names are
`DMS_ProteinGym_substitutions.zip` and `DMS_substitutions.csv`.
Their download URLs and exact SHA256 are in `configs/PROTEINGYM_SOURCE.json`.
The locally frozen archive/metadata also match the MD5 checksums displayed in
that release. All three extracted CSV member SHA256 and reference-sequence
identities match the frozen BATA inputs.
The release API identifies its license as `mit-license`; no raw data is bundled
here regardless. Retain the upstream notices and assay citations when acquiring
or using the data.

```bash
python scripts/unpack_proteingym.py --archive downloads/DMS_ProteinGym_substitutions.zip --metadata downloads/DMS_substitutions.csv --output data/proteingym
python scripts/prepare_sequences.py --benchmark PABP --assay-csv data/proteingym/PABP.csv --reference-fasta data/proteingym/PABP.fasta --output data/pabp_sequences
```

Repeat the second command with HIS7 or GRB2 and new output directories.
No candidate filtering, score thresholding or row sorting is performed.
`prepare_sequences.py` parses identity/sequence columns only, rejects mutations
outside the frozen design region, and checks the candidate-order digest before
writing task tokens, DCA candidate-region codes and PLM reference context.
The copied assay CSV contains labels; keep it with the external assay simulator,
not as optimizer input. Only currently revealed rows may reach `python -m bata`.

## GB1 and TrpB

GB1 uses `pera/data/GB1/scale2max/GB1.csv` and `pera/data/GB1/GB1.fasta` from
the [official ERA repository](https://github.com/rotskoff-group/era-directed-evolution),
at commit `1a90a860d793a72187f585e1c45fb0941cb7d609`. Both files were fetched
read-only from this revision and checked against the frozen local files:
CSV SHA256 `0555c9300cd73bf63b654b486d2cfba63fa4a8f55a4c3bc29563ee6ee031db5a`,
FASTA SHA256 `f1acaa18f06a15a27d30cc57ebd2f2d40158d4a8d4e7bf6b60d742a178fc99ce`.
TrpB uses `data/TrpB/fitness.csv` inside `data.zip` from
[ALDE data v1](https://zenodo.org/records/12196802), DOI 10.5281/zenodo.12196802,
credited to Jason Yang. The upstream metadata declares CC-BY-4.0. The remote
ZIP member was read with HTTP ranges and its CRC and SHA checked against the
frozen table: **byte-identical, with no further normalization or filtering**.
The full 5.9 GB archive was not downloaded or hash-verified; its publisher MD5
and the verified member SHA are distinguished in `configs/TRPB_SOURCE.json`.
No raw assay table or accompanying model-feature tensors are redistributed.

```bash
python scripts/unpack_trpb.py --archive inputs/alde_data.zip --output inputs/trpb_fitness.csv
```

TrpB's parent reference is the 397-residue entity 1 from
[PDB 8VHH version 1.0](https://www.rcsb.org/structure/8VHH), DOI
10.2210/pdb8VHH/pdb. The downloaded FASTA was byte-identical to the frozen
reference, including its terminal His tag. Do not trim it, replace it with
the wild-type UniProt sequence, or use the shorter MSA reference as PLM context.
`REFERENCE_SOURCES.json` binds the exact download and SHA for both GB1 and TrpB.

```bash
python scripts/download_reference.py --benchmark TrpB --output inputs/8VHH.fasta
```

Source-byte SHA and reference-sequence SHA are also recorded in
`DATA_IDENTITY.json` and `PLM_RECIPE.json`. The sequence-preparation command
accepts files only when their frozen identities match.

MSA acquisition/mapping is separate from assay sequence preparation and is not
yet fully packaged. Do not infer that matching an assay archive reconstructs
the prior-informed DCA representation. See `DCA_REBUILD.md` and `PLM_REBUILD.md`.
