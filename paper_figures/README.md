# Final paper figure reproduction

This directory contains the frozen summary CSVs and plotting scripts for the
six main figures (`Figure1`–`Figure6`) and seven appendix figures in the current
paper. It contains no raw sequences, MSA, model weights, or new experiments.

Run from this directory with Python, NumPy, pandas, Matplotlib, and Times New
Roman fonts available:

```bash
python reproduce.py
```

The five scripts write PDFs and PNGs to `generated/` and numerical/plotting audit
records to `qa/`. Both directories are ignored by Git. Each script checks
cohort sizes and figure-specific arithmetic against the frozen CSV input.
The final Figure 5 uses `draw_figure5_layout.py`; `reproduce.py` copies its
output to `Figure5.pdf` after the earlier 2.5-inch draft is generated.
The short `runtime_protocol.txt` preserves the two M100 stopping-condition
phrases checked by the appendix plotting script; these match the current paper.
The generated PDFs should be compared with the 13 PDFs in the arXiv source
package. Environment-dependent PDF metadata and fonts can change byte hashes;
verify the visual output and numerical audit before using figures elsewhere.
