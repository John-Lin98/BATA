# Paper figure sources

These three TikZ summary sources visualize selected paper results using the
listed numerical files. They are not the final figure files in the arXiv paper.
From this directory, compile `pdflatex figures.tex` twice (TikZ and standard
LaTeX packages required).

| Figure | Full-precision numerical source |
|---|---|
| 1: method diagram | No numerical result; see `bata/core.py` and `bata/calibration.py` |
| 2: main results | `results/main35_summary.csv` and `main35_per_run.csv` |
| 3a: prior-weight trajectories | `results/appendix/weights_summary.csv`, R1 BATA rows for the three main tasks |
| 3b: matched objective effects | `results/calibration/B96_paired_effects.csv` and `mean_rank.csv` |
| 3c: expert sufficiency | `results/variants/fine_only_m20.json` and `g20_confirmation.json` |

Figure 3c reproduces the paper's descriptive comparison. In particular, the
TrpB Fine-only M20 value uses 70 runs but its BATA reference uses main35; it is
**not a paired 70-run effect**. See `docs/EXPERT_SUFFICIENCY.md`. Do not interpret
that panel as a matched single-component ablation or silently pool cohorts.
The G20 confirmation rank comes from its separate disjoint confirmation set.

The CSV/JSON files, not rounded TikZ coordinates, are the precision source.
