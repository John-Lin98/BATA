"""Reproduce five BATA figures from released CSVs, at a true 5.5-inch width.

Compatible with Matplotlib 3.3.4 / NumPy 1.20.1. No bootstrap or experimental
results are recomputed. Data identities and plotted values are audited locally.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import numpy.typing as npt
# Pillow's annotations expect newer NumPy typing; no array behavior is changed.
if not hasattr(npt, "NDArray"):
    class _ArrayHint:
        def __class_getitem__(cls, item):
            return np.ndarray
    npt.NDArray = _ArrayHint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA, OUTPUT, QA = ROOT / "data", ROOT / "generated", ROOT / "qa"
OUTPUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
assert "times" in font_manager.findfont(
    font_manager.FontProperties(family="Times New Roman"),
    fallback_to_default=False,
).lower()
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 8.1, "axes.linewidth": 0.55,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "axes.unicode_minus": True,
})
TASKS = ["GB1", "PABP", "TrpB"]
ARMS = ["bata", "alde", "evolvepro650", "reap100", "rf_onehot"]
LABEL = {"bata": "BATA", "alde": "ALDE", "evolvepro650": "EVOLVEpro-650M",
         "reap100": "REAP100-650M", "rf_onehot": "RF"}
SHORT = dict(LABEL, evolvepro650="EVOLVEpro", reap100="REAP100")
STYLE = {
    "bata": ("#176B86", "o", "-"), "alde": ("#B3863C", "s", "--"),
    "evolvepro650": ("#8E7BAA", "^", "-."), "reap100": ("#4C7861", "D", ":"),
    "rf_onehot": ("#777777", "x", (0, (5, 2, 1, 2))),
}
TASK_COLOR = {"GB1": "#3979A5", "PABP": "#BA7C3E", "TrpB": "#786398"}
AUDIT = {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
         "font": "Times New Roman", "sources": {}, "figures": []}


def read(filename):
    path = DATA / filename
    AUDIT["sources"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def axis_style(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.2, width=0.55, pad=2.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#DCE1E4", linewidth=0.45, alpha=0.85)


def limited_ticks(ax, axis="x", nbins=3):
    limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
    low, high = sorted(limits)
    ticks = MaxNLocator(nbins=nbins).tick_values(low, high)
    ticks = [v for v in ticks if low - 1e-12 <= v <= high + 1e-12]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)


def method_legend(fig, y=0.015):
    handles = [Line2D([0], [0], color=STYLE[a][0], marker=STYLE[a][1],
                      linestyle=STYLE[a][2], markersize=3.5,
                      linewidth=1.4 if a == "bata" else 1.05, label=LABEL[a])
               for a in ARMS]
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                      ncol=5, frameon=False, handlelength=1.55, handletextpad=0.35,
                      columnspacing=0.85, borderaxespad=0, fontsize=8.1)


def save(fig, name, numerical_checks):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts, outside, small = [], [], []
    for obj in fig.findobj(match=Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        bbox = obj.get_window_extent(renderer)
        texts.append({"text": obj.get_text(), "font_pt": obj.get_fontsize(),
                      "bbox_px": [float(x) for x in bbox.bounds]})
        if (bbox.x0 < -0.5 or bbox.y0 < -0.5 or
                bbox.x1 > fig.bbox.width + 0.5 or bbox.y1 > fig.bbox.height + 0.5):
            outside.append(obj.get_text())
        if obj.get_fontsize() < 8:
            small.append(obj.get_text())
    assert not outside, (name, "text outside canvas", outside)
    assert not small, (name, "font below 8 pt", small)
    assert abs(fig.get_size_inches()[0] - 5.5) < 1e-12
    fig.savefig(OUTPUT / (name + ".pdf"), facecolor="white",
                metadata={"Title": name, "Author": "", "Subject": "",
                          "Creator": "Matplotlib", "Keywords": ""})
    fig.savefig(OUTPUT / (name + ".png"), dpi=300, facecolor="white")
    AUDIT["figures"].append({"name": name, "size_inches": list(fig.get_size_inches()),
                             "text_outside_canvas": outside, "fonts_below_8pt": small,
                             "numerical_checks": numerical_checks, "text": texts})
    plt.close(fig)


def figure2():
    rows = read("Figure2_trajectories.csv")
    fig = plt.figure(figsize=(5.5, 2.30))
    checks = []
    for k, task in enumerate(TASKS):
        ax = fig.add_axes([0.09 + k * 0.315, 0.285, 0.27, 0.57])
        axis_style(ax)
        upper = 3.0 if task == "PABP" else 1.04
        for arm in ARMS[1:] + ARMS[:1]:
            records = sorted([r for r in rows if r["benchmark"] == task and r["arm"] == arm],
                             key=lambda r: int(r["queries"]))
            assert len(records) == 5 and {r["n"] for r in records} == {"35"}
            assert {r["cohort"] for r in records} == {"original_main35"}
            x = np.array([int(r["queries"]) for r in records])
            mean = np.array([float(r["mean"]) for r in records])
            low = np.array([float(r["ci95_low"]) for r in records])
            high = np.array([float(r["ci95_high"]) for r in records])
            assert list(x) == [96, 192, 288, 384, 480]
            assert np.all((0 <= low) & (low <= mean) & (mean <= high) & (high <= upper))
            color, marker, linestyle = STYLE[arm]
            ax.fill_between(x, low, high, color=color, alpha=0.14 if arm == "bata" else 0.085,
                            linewidth=0, zorder=1)
            ax.plot(x, mean, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.5 if arm == "bata" else 1.05,
                    markersize=3.6 if arm == "bata" else 3.2,
                    markeredgewidth=0.65, zorder=4 if arm == "bata" else 3)
            checks.append({"task": task, "arm": arm, "n": 35, "queries": x.tolist(),
                           "mean": mean.tolist(), "CI_low": low.tolist(), "CI_high": high.tolist()})
        ax.set_xlim(81, 495)
        ax.set_xticks([96, 192, 288, 384, 480])
        ax.set_ylim(0, upper)
        ax.set_yticks([0, 1, 2, 3] if task == "PABP" else [0, 0.5, 1.0])
        ax.set_title("({}) {}".format(chr(97 + k), task), loc="left", pad=6)
        if k == 0:
            ax.set_ylabel("Best observed fitness", labelpad=4)
    fig.text(0.54, 0.145, "Measurements", ha="center", fontsize=8.5)
    method_legend(fig, 0.025)
    save(fig, "Figure2", checks)


def figure3():
    ranks = read("Figure3_rank_stability.csv")
    main = read("Main35_summary.csv")
    sensitivity = read("Figure3_sensitivity70_summary.csv")
    fig = plt.figure(figsize=(5.5, 2.50))
    rankax = fig.add_axes([0.09, 0.30, 0.23, 0.51])
    axis_style(rankax)
    checks = []
    for arm in ARMS[1:] + ARMS[:1]:
        row = next(r for r in ranks if r["arm"] == arm and r["metric"] == "Final")
        values = [float(row["main35_mean_rank"]), float(row["sensitivity_setting_mean_rank"])]
        offset = (ARMS.index(arm) - 2) * 0.014
        color, marker, linestyle = STYLE[arm]
        rankax.plot([offset, 1 + offset], values, color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.5 if arm == "bata" else 1.05,
                    markersize=3.8, markeredgewidth=0.65,
                    zorder=4 if arm == "bata" else 3)
        checks.append({"kind": "mean_task_rank", "arm": arm, "values": values})
    rankax.set_xlim(-0.18, 1.18)
    rankax.set_ylim(4.65, 1.3)
    rankax.set_yticks([2, 3, 4])
    rankax.set_xticks([0, 1])
    rankax.set_xticklabels(["Main-35", "Sensitivity"])
    rankax.set_ylabel("Mean Final@480 rank", labelpad=3)
    fig.text(0.09, 0.925, "(a) Mean task rank", fontsize=9.5, fontweight="bold")

    for k, task in enumerate(["GB1", "TrpB"]):
        ax = fig.add_axes([0.51 if k == 0 else 0.78, 0.30,
                           0.205 if k == 0 else 0.20, 0.51])
        axis_style(ax, "x")
        records = [r for r in main + sensitivity if r["benchmark"] == task]
        low = min(float(r["Final_ci95_low"]) for r in records)
        high = max(float(r["Final_ci95_high"]) for r in records)
        pad = (high - low) * 0.07
        ax.set_xlim(low - pad, high + pad)
        for j, arm in enumerate(ARMS):
            one = next(r for r in main if r["benchmark"] == task and r["arm"] == arm)
            two = next(r for r in sensitivity if r["benchmark"] == task and r["arm"] == arm)
            assert (one["n"], two["n"]) == ("35", "70")
            y = 4 - j
            color, marker, _ = STYLE[arm]
            ax.plot([float(one["Final_mean"]), float(two["Final_mean"])], [y + 0.16, y - 0.16],
                    color="#B3BBC0", linewidth=0.65, zorder=1)
            for row, dy, filled in [(one, 0.16, False), (two, -0.16, True)]:
                mean, lo, hi = [float(row[c]) for c in ["Final_mean", "Final_ci95_low", "Final_ci95_high"]]
                assert low - pad <= lo <= mean <= hi <= high + pad
                ax.errorbar(mean, y + dy, xerr=[[mean - lo], [hi - mean]], fmt=marker,
                            color=color, markerfacecolor=color if filled else "white",
                            markersize=3.6, markeredgewidth=0.75, capsize=1.8,
                            elinewidth=0.85, zorder=3)
                checks.append({"kind": "cohort_mean", "task": task, "arm": arm,
                               "cohort": row["cohort"], "n": int(row["n"]), "mean": mean,
                               "CI_low": lo, "CI_high": hi})
        ax.set_ylim(-0.55, 4.55)
        limited_ticks(ax, "x", 3)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if task == "TrpB" else "%.1f"))
        ax.set_yticks(range(5))
        ax.set_yticklabels([SHORT[a] for a in ARMS[::-1]] if k == 0 else [])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xlabel("Final@480", labelpad=4)
        fig.text(0.51 if k == 0 else 0.78, 0.925,
                 "({}) {}".format(chr(98 + k), task), fontsize=9.5, fontweight="bold")
    cohort_handles = [Line2D([0], [0], linestyle="none", marker="o", color="#56616A",
                             markerfacecolor="white", markersize=3.6, label="Main-35 (upper)"),
                      Line2D([0], [0], linestyle="none", marker="o", color="#56616A",
                             markersize=3.6, label="Sensitivity-70 (lower)")]
    fig.legend(handles=cohort_handles, loc="center", bbox_to_anchor=(0.74, 0.86),
               ncol=2, frameon=False, fontsize=8.0, handletextpad=0.25,
               handlelength=0.65, columnspacing=0.65, borderaxespad=0)
    method_legend(fig, 0.035)
    save(fig, "Figure3", checks)


def figure4():
    summary = read("Figure4_weight_summary.csv")
    pairs = read("Figure4_paired_shifts.csv")
    shifts = read("Figure4_shift_summary.csv")
    fig = plt.figure(figsize=(5.5, 2.45))
    ax = fig.add_axes([0.10, 0.275, 0.31, 0.55])
    axis_style(ax)
    checks = []
    for k, task in enumerate(TASKS):
        rows = sorted([r for r in summary if r["benchmark"] == task], key=lambda r: int(r["round"]))
        x = [int(r["round"]) for r in rows]
        mean = np.array([float(r["mean"]) for r in rows])
        low = np.array([float(r["ci95_low"]) for r in rows])
        high = np.array([float(r["ci95_high"]) for r in rows])
        assert x == [1, 2, 3, 4] and all(r["n"] == "35" for r in rows)
        assert np.all((0 <= low) & (low <= mean) & (mean <= high) & (high <= 1))
        ax.fill_between(x, low, high, color=TASK_COLOR[task], alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=TASK_COLOR[task], marker=["o", "s", "^"][k],
                linestyle=["-", "--", "-."][k], linewidth=1.35, markersize=3.8, label=task)
        checks.append({"kind": "weight_trajectory", "task": task, "n": 35,
                       "rounds": x, "mean": mean.tolist(), "CI_low": low.tolist(), "CI_high": high.tolist()})
    ax.set_xlim(0.83, 4.17)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Prior-informed weight", labelpad=4)
    ax.set_xlabel("Decision round", labelpad=4)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0, handlelength=1.7,
              handletextpad=0.4, labelspacing=0.15, borderpad=0.1)
    fig.text(0.10, 0.925, "(a) Mean weight", fontsize=9.5, fontweight="bold")

    paired = fig.add_axes([0.51, 0.275, 0.47, 0.55])
    axis_style(paired)
    for k, task in enumerate(TASKS):
        center, color = k * 2.0, TASK_COLOR[task]
        x = [center - 0.34, center + 0.34]
        rows = [r for r in pairs if r["benchmark"] == task]
        assert len(rows) == 35 and {int(r["group"]) for r in rows} == set(range(35))
        for row in rows:
            values = [float(row["round1"]), float(row["round4"])]
            assert all(0 <= value <= 1 for value in values)
            paired.plot(x, values, color=color, alpha=0.24, linewidth=0.5,
                        marker="o", markersize=1.5, markeredgewidth=0, zorder=1)
        aggregate = next(r for r in shifts if r["benchmark"] == task)
        means = [float(aggregate["round1_mean"]), float(aggregate["round4_mean"])]
        medians = [float(aggregate["round1_median"]), float(aggregate["round4_median"])]
        actual = np.array([[float(r["round1"]), float(r["round4"])] for r in rows])
        assert np.allclose(actual.mean(axis=0), means, rtol=0, atol=1e-15)
        assert np.allclose(np.median(actual, axis=0), medians, rtol=0, atol=1e-15)
        paired.plot(x, means, color=color, linewidth=1.65, marker="s", markersize=4.2,
                    markeredgewidth=0.65, zorder=4)
        paired.plot(x, medians, color="#273E4A", linewidth=1.05, linestyle="--",
                    marker="D", markersize=3.3, markerfacecolor="white", markeredgewidth=0.65, zorder=5)
        paired.text(center, 1.035, task, ha="center", va="bottom", color=color,
                    fontsize=8.5, fontweight="bold", clip_on=False)
        checks.append({"kind": "paired_weights", "task": task, "n": 35,
                       "group_ids": [int(r["group"]) for r in rows], "values": actual.tolist(),
                       "means": means, "medians": medians})
    paired.set_ylim(0, 1)
    paired.set_xlim(-0.68, 4.68)
    paired.set_yticks([0, 0.5, 1])
    paired.set_xticks([v for k in range(3) for v in [2 * k - 0.34, 2 * k + 0.34]])
    paired.set_xticklabels(["1", "4"] * 3)
    paired.set_xlabel("Decision round", labelpad=4)
    fig.text(0.51, 0.925, "(b) Campaign shifts", fontsize=9.5, fontweight="bold")
    handles = [Line2D([0], [0], color="#7D909D", linewidth=0.65, marker="o", markersize=2, label="Campaign"),
               Line2D([0], [0], color="#176B86", linewidth=1.65, marker="s", markersize=4, label="Mean"),
               Line2D([0], [0], color="#273E4A", linewidth=1, linestyle="--", marker="D",
                      markerfacecolor="white", markersize=3, label="Median")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.745, 0.025),
               ncol=3, frameon=False, fontsize=8.0, handlelength=1.3, handletextpad=0.35,
               columnspacing=0.8, borderaxespad=0)
    save(fig, "Figure4", checks)


def appendix_ecdf():
    rows = read("Main35_per_run.csv")
    assert len(rows) == 525 and {r["cohort"] for r in rows} == {"original_main35"}
    assert {r["status"] for r in rows} == {"PASS"}
    fig = plt.figure(figsize=(5.5, 3.75))
    checks = []
    for j, metric in enumerate(["Final", "Query_AUC"]):
        for k, task in enumerate(TASKS):
            ax = fig.add_axes([0.10 + 0.312 * k, 0.615 if j == 0 else 0.185, 0.268, 0.29])
            axis_style(ax)
            all_values = [float(r[metric]) for r in rows if r["benchmark"] == task]
            low, high = min(all_values), max(all_values)
            pad = (high - low) * 0.05
            xmin, xmax = low - pad, high + pad
            for arm in ARMS[1:] + ARMS[:1]:
                records = [r for r in rows if r["benchmark"] == task and r["arm"] == arm]
                assert len(records) == 35 and {int(r["group"]) for r in records} == set(range(35))
                values = np.sort([float(r[metric]) for r in records])
                proportions = np.arange(1, 36) / 35.0
                color, marker, linestyle = STYLE[arm]
                # Zero/one extensions complete the ECDF support; no sample is added.
                ax.step(np.r_[xmin, values, xmax], np.r_[0, proportions, 1], where="post",
                        color=color, linestyle=linestyle,
                        linewidth=1.35 if arm == "bata" else 1.0, zorder=4 if arm == "bata" else 3)
                ax.plot(values[::7], proportions[::7], linestyle="none", marker=marker,
                        color=color, markersize=2.8, markeredgewidth=0.65, zorder=5)
                checks.append({"task": task, "metric": metric, "arm": arm,
                               "n": 35, "sorted_values": values.tolist()})
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 1.025)
            ax.set_yticks([0, 0.5, 1])
            if k != 0:
                ax.set_yticklabels([])
            limited_ticks(ax, "x", 3)
            ax.set_xlabel("Final@480" if metric == "Final" else "Query-AUC", labelpad=3)
            ax.set_title("({}) {}".format(chr(97 + j * 3 + k), task), loc="left", pad=5)
    fig.text(0.015, 0.56, "Cumulative proportion", rotation=90, va="center", fontsize=8.5)
    method_legend(fig, 0.02)
    save(fig, "Appendix_ECDF", checks)


def appendix_weights():
    rows = read("Figure4_weights_per_run.csv")
    assert len(rows) == 420 and {r["cohort"] for r in rows} == {"original_main35"}
    fig = plt.figure(figsize=(5.5, 3.20))
    checks = []
    for k, task in enumerate(TASKS):
        records = [r for r in rows if r["benchmark"] == task]
        groups = sorted({int(r["group"]) for r in records})
        assert groups == list(range(35)) and len(records) == 140
        values = np.array([[float(next(r["prior_weight"] for r in records
                                      if int(r["group"]) == group and int(r["round"]) == rd))
                            for rd in range(1, 5)] for group in groups])
        assert values.shape == (35, 4) and np.all((0 <= values) & (values <= 1))
        ax = fig.add_axes([0.10 + k * 0.308, 0.245, 0.258, 0.66])
        mesh = ax.pcolormesh(np.arange(5), np.arange(36), values, cmap="Blues",
                             vmin=0, vmax=1, shading="flat", rasterized=False,
                             edgecolors="white", linewidth=0.12)
        ax.set_xlim(0, 4)
        ax.set_ylim(35, 0)
        ax.set_xticks(np.arange(4) + 0.5)
        ax.set_xticklabels(["1", "2", "3", "4"])
        ids = [0, 5, 10, 15, 20, 25, 30, 34]
        ax.set_yticks(np.array(ids) + 0.5)
        ax.set_yticklabels([str(groups[i]) for i in ids] if k == 0 else [])
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#6A7880")
        ax.set_title("({}) {}".format(chr(97 + k), task), loc="left", pad=6,
                     color=TASK_COLOR[task])
        if k == 0:
            ax.set_ylabel("Campaign index", labelpad=4)
        checks.append({"task": task, "n": 35, "group_ids": groups, "rounds": [1, 2, 3, 4],
                       "weights": values.tolist(), "colormap_limits": [0, 1]})
    fig.text(0.55, 0.173, "Decision round", ha="center", fontsize=8.5)
    colorax = fig.add_axes([0.365, 0.095, 0.36, 0.029])
    bar = fig.colorbar(mesh, cax=colorax, orientation="horizontal", ticks=[0, 0.5, 1])
    bar.ax.tick_params(length=2, width=0.5, pad=2, labelsize=8)
    bar.outline.set_linewidth(0.5)
    fig.text(0.545, 0.012, "Prior-informed predictor weight", ha="center", fontsize=8.5)
    save(fig, "Appendix_Weights", checks)


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    appendix_ecdf()
    appendix_weights()
    (QA / "figures_2_4_data_audit.json").write_text(json.dumps(AUDIT, indent=2), encoding="utf-8")
    print("Generated:", ", ".join(item["name"] for item in AUDIT["figures"]))
    print("All figures: width=5.5 inches; font>=8 pt; CI/data containment checks passed.")
