"""Reproduce G20, engineering runtime, and sensitivity-table appendix assets.

Uses the CSVs and fixed protocols supplied with the manuscript. All plotted
uncertainties are preserved; no confidence interval is inferred for G20 rounds.
"""
import json
import numpy as np
import draw_figures_2_4 as common
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt = common.plt
ROOT = common.ROOT
TASK5 = ["GB1", "PABP", "TrpB", "HIS7", "GRB2"]
TASK_COLOR = dict(common.TASK_COLOR, HIS7="#4C7861", GRB2="#777777")


def appendix_g20():
    ranks = common.read("Appendix_G20_summary.csv")
    switches = common.read("Appendix_G20_switching.csv")
    assert len(ranks) == 15 and len(switches) == 20
    assert {r["cohort"] for r in ranks + switches} == {"G20_fresh_confirmation10"}
    assert {r["n"] for r in ranks} == {"10"}
    arms = ["bata", "fine_m20", "g20"]
    values = np.array([[float(next(r["Final_task_rank"] for r in ranks
                                   if r["benchmark"] == task and r["arm"] == arm))
                        for arm in arms] for task in TASK5])
    assert np.all(np.sort(values, axis=1) == [1, 2, 3])
    means = values.mean(axis=0)
    assert np.allclose(means, [1.4, 2.4, 2.2], rtol=0, atol=1e-14)
    fig = plt.figure(figsize=(5.5, 2.50))
    matrix = fig.add_axes([0.12, 0.245, 0.335, 0.58])
    matrix.pcolormesh(np.arange(4), np.arange(6), values, cmap="Blues_r",
                      vmin=-0.5, vmax=3.6, edgecolors="white", linewidth=0.8,
                      shading="flat", rasterized=False)
    matrix.add_patch(Rectangle((0, 5), 3, 1, facecolor="#F1F3F4", edgecolor="none"))
    for i in range(5):
        for j in range(3):
            matrix.text(j + 0.5, i + 0.5, "{:.0f}".format(values[i, j]),
                        ha="center", va="center", fontsize=9)
    for j in range(3):
        matrix.text(j + 0.5, 5.5, "{:.2f}".format(means[j]), ha="center",
                    va="center", fontsize=8.5, fontweight="bold")
    matrix.set_xlim(0, 3)
    matrix.set_ylim(6, 0)
    matrix.set_xticks(np.arange(3) + 0.5)
    matrix.set_xticklabels(["BATA", "Fine-only\nM20", "G20"])
    matrix.set_yticks(np.arange(6) + 0.5)
    matrix.set_yticklabels(TASK5 + ["Mean"])
    matrix.tick_params(length=0, pad=3)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    fig.text(0.025, 0.925, "(a) Final@480 rank", fontsize=9.5, fontweight="bold")

    ax = fig.add_axes([0.62, 0.245, 0.355, 0.58])
    common.axis_style(ax)
    checks = [{"kind": "rank_matrix", "tasks": TASK5, "arms": arms,
               "n_per_task": 10, "ranks": values.tolist(), "mean_ranks": means.tolist(),
               "tie_rule": sorted({r["rank_tie_policy"] for r in ranks})}]
    handles = []
    for k, task in enumerate(TASK5):
        rows = sorted([r for r in switches if r["benchmark"] == task], key=lambda r: int(r["round"]))
        x = [int(r["round"]) for r in rows]
        counts = [int(r["fine_branch_count"]) for r in rows]
        y = [float(r["fine_branch_fraction"]) for r in rows]
        assert x == [1, 2, 3, 4] and all(r["n_campaigns"] == "10" for r in rows)
        assert np.array_equal(np.array(counts) / 10.0, np.array(y))
        assert all("rounds not independent" in r["uncertainty"] for r in rows)
        marker = ["o", "s", "^", "D", "x"][k]
        style = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))][k]
        line, = ax.plot(x, y, color=TASK_COLOR[task], marker=marker,
                        linestyle=style, linewidth=1.25, markersize=3.5,
                        markeredgewidth=0.65, label=task, clip_on=False)
        handles.append(line)
        checks.append({"kind": "switching", "task": task, "n": 10, "rounds": x,
                       "counts": counts, "fractions": y, "CI": None})
    ax.set_xlim(0.86, 4.14)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Decision round", labelpad=4)
    ax.set_ylabel("Fraction using\nFine-only M20", labelpad=4)
    fig.text(0.62, 0.925, "(b) G20 switching", fontsize=9.5, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.035),
               ncol=5, frameon=False, fontsize=8.1, handlelength=1.7,
               handletextpad=0.35, columnspacing=1.3, borderaxespad=0)
    common.save(fig, "Appendix_G20", checks)


def appendix_runtime():
    summaries = common.read("Appendix_runtime_summary.csv")
    records = common.read("Appendix_runtime_capacity_records.csv")
    # The two-round stopping condition is part of the existing protocol prose.
    protocol = (ROOT / "runtime_protocol.txt").read_text(encoding="utf-8")
    assert "reached their declared time limits after two rounds" in protocol
    assert "three M100 attempts stopped after two rounds at their time limits" in protocol
    assert len(records) == 15
    variants = ["old_m5", "fast_m5", "reap10", "reap100"]
    labels = {"old_m5": "Original M5", "fast_m5": "Optimized M5", "reap10": "REAP10", "reap100": "REAP100"}
    fig = plt.figure(figsize=(5.5, 2.40))
    ax = fig.add_axes([0.19, 0.245, 0.355, 0.57])
    common.axis_style(ax, "x")
    checks = []
    for k, variant in enumerate(variants):
        rows = sorted([r for r in records if r["variant"] == variant and r["status"] == "completed"],
                      key=lambda r: int(r["group"]))
        assert len(rows) == 3
        values = np.array([float(r["campaign_wall_s"]) for r in rows])
        aggregate = next(r for r in summaries if r["variant"] == variant)
        median = float(aggregate["median_s"])
        assert np.isclose(np.median(values), median, rtol=0, atol=1e-10)
        assert np.isclose(values.min(), float(aggregate["min_s"]), rtol=0, atol=1e-10)
        assert np.isclose(values.max(), float(aggregate["max_s"]), rtol=0, atol=1e-10)
        y = 3 - k
        color = "#176B86" if variant == "fast_m5" else "#77858D"
        ax.plot(values, [y - 0.095, y, y + 0.095], linestyle="none", marker="o",
                markerfacecolor="white", color=color, markersize=3.7,
                markeredgewidth=0.85, zorder=3)
        ax.plot(median, y, linestyle="none", marker="|", color=color,
                markersize=12, markeredgewidth=1.5, zorder=4)
        ax.text(values.max() + 25, y, "{:.1f}".format(median), va="center",
                fontsize=8.1, color=color)
        checks.append({"kind": "completed_timing", "variant": variant, "n": 3,
                       "all_seconds": values.tolist(), "median_seconds": median,
                       "reported_median": "{:.1f}".format(median), "CI": None})
    ax.set_xlim(0, 1430)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks([0, 400, 800, 1200])
    ax.set_yticks(range(4))
    ax.set_yticklabels([labels[v] for v in variants[::-1]])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Complete-campaign wall time (s)", labelpad=4)
    fig.text(0.025, 0.925, "(a) Completed timings", fontsize=9.5, fontweight="bold")

    stopped = sorted([r for r in records if r["status"] != "completed"], key=lambda r: int(r["group"]))
    assert len(stopped) == 3 and {r["variant"] for r in stopped} == {"bata_m100_ts"}
    assert all(r["Final"] == "" and r["Query_AUC"] == "" and r["campaign_wall_s"] == "" for r in stopped)
    tableax = fig.add_axes([0.625, 0.395, 0.355, 0.425])
    tableax.axis("off")
    cells = [[str(int(r["group"]) + 1), "{:.0f}".format(float(r["gate_limit_s"])),
              "{:.0f}".format(float(r["observed_elapsed_s"]))] for r in stopped]
    table = tableax.table(cellText=cells, colLabels=["Run", "Limit (s)", "Stopped (s)"],
                          cellLoc="center", colWidths=[0.23, 0.365, 0.405], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DEE2")
        cell.set_linewidth(0.4)
        cell.set_facecolor("#F0F3F5" if row == 0 else "white")
        if row == 0:
            cell.get_text().set_fontweight("bold")
    fig.text(0.625, 0.925, "(b) Partial M100 timings", fontsize=9.5, fontweight="bold")
    fig.text(0.8025, 0.29, "3/3 stopped after two rounds", ha="center", fontsize=8.5, color="#76563D")
    handles = [Line2D([0], [0], color="#77858D", marker="o", markerfacecolor="white",
                      linestyle="none", markersize=3.5, label="Runs (n = 3)"),
               Line2D([0], [0], color="#77858D", marker="|", linestyle="none",
                      markersize=9, markeredgewidth=1.5, label="Median")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.345, 0.035),
               ncol=2, frameon=False, fontsize=8.1, handletextpad=0.3,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0)
    checks.append({"kind": "incomplete_M100", "n": 3, "completed_rounds": 2,
                   "elapsed_seconds": [float(r["observed_elapsed_s"]) for r in stopped],
                   "limit_seconds": [float(r["gate_limit_s"]) for r in stopped],
                   "displayed_cells": cells, "final_fitness": None, "CI": None,
                   "protocol_source": "baseline appendix ADD-P5 and CAP-FA-RUNTIME"})
    common.save(fig, "Appendix_Runtime", checks)


def appendix_sensitivity():
    rows = common.read("Appendix_sensitivity_setting.csv")
    ranks = common.read("Figure3_rank_stability.csv")
    assert len(rows) == 15
    fig = plt.figure(figsize=(5.5, 2.70))
    checks = []
    for k, metric in enumerate(["Final", "Query_AUC"]):
        body = []
        for arm in common.ARMS:
            records = [next(r for r in rows if r["benchmark"] == task and r["arm"] == arm)
                       for task in common.TASKS]
            assert [r["n"] for r in records] == ["70", "35", "70"]
            assert [r["cohort"] for r in records] == ["native70_sensitivity", "original_main35", "native70_sensitivity"]
            rank = float(next(r["sensitivity_setting_mean_rank"] for r in ranks
                              if r["arm"] == arm and r["metric"] == metric))
            displays = ["{:.4f} ± {:.4f}".format(float(r[metric + "_mean"]), float(r[metric + "_sd"]))
                        for r in records]
            body.append([common.LABEL[arm]] + displays + ["{:.2f}".format(rank)])
            checks.append({"metric": metric, "arm": arm, "n": [70, 35, 70],
                           "means": [float(r[metric + "_mean"]) for r in records],
                           "sample_sds": [float(r[metric + "_sd"]) for r in records],
                           "displayed": displays, "mean_task_rank": rank})
        bottom = 0.545 if k == 0 else 0.09
        ax = fig.add_axes([0.012, bottom, 0.976, 0.36])
        ax.axis("off")
        table = ax.table(cellText=body,
                         colLabels=["Method", "GB1 (n = 70)", "PABP (n = 35)", "TrpB (n = 70)", "Mean rank"],
                         colWidths=[0.225, 0.212, 0.212, 0.212, 0.139],
                         cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.PAD = 0.035
            cell.set_facecolor("#EFF3F5" if row == 0 else ("#EEF5F7" if row == 1 else "white"))
            if row == 0:
                cell.get_text().set_fontweight("bold")
            if col == 0:
                cell.get_text().set_ha("left")
        for y in [0, 1]:
            ax.plot([0, 1], [y, y], transform=ax.transAxes, color="#47555D", linewidth=0.6, clip_on=False)
        ax.plot([0, 1], [5 / 6.0, 5 / 6.0], transform=ax.transAxes,
                color="#869299", linewidth=0.4, clip_on=False)
        title = "Final@480" if metric == "Final" else "Query-AUC"
        fig.text(0.014, 0.94 if k == 0 else 0.485,
                 title + " (mean ± SD)", fontsize=9.5, fontweight="bold")
    fig.text(0.985, 0.018, "Lower mean rank is better.", ha="right", fontsize=8.0)
    common.save(fig, "Appendix_Sensitivity", checks)


if __name__ == "__main__":
    appendix_g20()
    appendix_runtime()
    appendix_sensitivity()
    (common.QA / "appendix_remaining_data_audit.json").write_text(
        json.dumps(common.AUDIT, indent=2), encoding="utf-8")
    print("Generated Appendix_G20, Appendix_Runtime, Appendix_Sensitivity.")
    print("Exact released ranks, counts, timings, means/SDs retained; no new uncertainty.")
