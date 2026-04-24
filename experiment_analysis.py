#!/usr/bin/env python3
"""
mojoKNN Experiment Analysis
===========================
Loads a results CSV from experiment_runner.py and produces:
  - Per-dataset:  summary statistics table + bar chart (time + speedup)
  - Combined:     scaling line plot (time vs dataset size)
                  speedup bar chart (all datasets side-by-side)
  - Multi-dataset method comparison (line chart: x=method, y=time, lines=datasets)
  - Speedup heatmap (datasets x algorithms)
  - Accuracy comparison chart
  - Boxplots of run distributions

All figures are saved to results/figures/.

Usage:
    python experiment_analysis.py                        # latest results/batch_*.csv
    python experiment_analysis.py results/batch_X.csv
"""

import sys
import os
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_RUNS   = 1        # number of leading runs to exclude from statistics
ALGO_ORDER    = ["sklearn_brute", "sklearn_kdtree", "brute_simd", "kdtree_simd"]
ALGO_LABELS   = {
    "sklearn_brute":  "sklearn brute",
    "sklearn_kdtree": "sklearn KD-tree",
    "brute_simd":     "Mojo brute SIMD",
    "kdtree_simd":    "Mojo KD-tree SIMD",
}
ALGO_COLORS   = {
    "sklearn_brute":  "#5B8DB8",   # steel blue
    "sklearn_kdtree": "#8B5DAA",   # purple
    "brute_simd":     "#E07B39",   # burnt orange
    "kdtree_simd":    "#4CAF7D",   # muted green
}
BASELINE_ALGO = "sklearn_brute"   # speedup denominator

FIG_DPI   = 150
FIG_EXT   = "png"


# ---------------------------------------------------------------------------
# Data loading + stats
# ---------------------------------------------------------------------------

def find_latest_csv(results_dir: str = "results") -> str:
    pattern = os.path.join(results_dir, "batch_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No batch CSVs found in {results_dir}/")
    return files[-1]


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce numeric columns
    for col in ["n_rows", "n_train", "n_test", "n_features", "k", "run"]:
        df[col] = df[col].astype(int)
    df["time_seconds"]  = df["time_seconds"].astype(float)
    df["accuracy_pct"]  = pd.to_numeric(df["accuracy_pct"], errors="coerce")
    return df


def compute_stats(df: pd.DataFrame, exclude_warmup: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame indexed by (dataset, algorithm) with columns:
    mean_s, std_s, median_s, mean_ms, std_ms, mean_acc, speedup, n_rows, n_features, k

    Warmup runs (run <= WARMUP_RUNS) are optionally excluded.
    kdtree_build rows are kept as-is (they only have run=0).
    """
    if exclude_warmup:
        # Keep kdtree_build (run==0) and steady-state runs (run > WARMUP_RUNS)
        mask = (df["algorithm"] == "kdtree_build") | (df["run"] > WARMUP_RUNS)
        df   = df[mask].copy()

    group_col = "group" if "group" in df.columns else None
    agg = (
        df.groupby(["dataset", "algorithm"])
        .agg(
            mean_s      = ("time_seconds",  "mean"),
            std_s       = ("time_seconds",  "std"),
            median_s    = ("time_seconds",  "median"),
            mean_acc    = ("accuracy_pct",  "mean"),
            n_rows      = ("n_rows",        "first"),
            n_features  = ("n_features",    "first"),
            k           = ("k",             "first"),
            n_runs      = ("time_seconds",  "count"),
            **({} if group_col is None else {"group": ("group", "first")}),
        )
        .reset_index()
    )
    agg["mean_ms"] = agg["mean_s"]   * 1000
    agg["std_ms"]  = agg["std_s"]    * 1000

    # Compute speedup vs baseline for each dataset
    baseline = (
        agg[agg["algorithm"] == BASELINE_ALGO]
        [["dataset", "mean_s"]]
        .rename(columns={"mean_s": "baseline_s"})
    )
    agg = agg.merge(baseline, on="dataset", how="left")
    agg["speedup"] = agg["baseline_s"] / agg["mean_s"]

    return agg.sort_values(["n_rows", "algorithm"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_time(val_ms: float) -> str:
    """Format time in human-readable units."""
    if val_ms < 1:
        return f"{val_ms*1000:.1f} us"
    if val_ms < 1000:
        return f"{val_ms:.2f} ms"
    return f"{val_ms/1000:.2f} s"


def get_datasets_ordered(stats: pd.DataFrame) -> list[str]:
    """Return dataset names ordered by n_rows."""
    return (
        stats[["dataset", "n_rows"]].drop_duplicates()
        .sort_values("n_rows")["dataset"].tolist()
    )


def print_dataset_table(stats: pd.DataFrame, dataset: str) -> None:
    sub = stats[stats["dataset"] == dataset].copy()
    sub = sub[sub["algorithm"].isin(ALGO_ORDER)]
    sub = sub.set_index("algorithm").reindex(ALGO_ORDER).dropna(subset=["mean_ms"])

    n_rows     = sub["n_rows"].iloc[0]
    n_features = sub["n_features"].iloc[0]
    k          = sub["k"].iloc[0]

    print(f"\n{'─'*80}")
    print(f"  Dataset : {dataset}  ({n_rows} rows x {n_features} features,  K={k})")
    print(f"{'─'*80}")
    header = f"  {'Algorithm':<22}  {'Mean (ms)':>10}  {'Std (ms)':>8}  {'Speedup':>8}  {'Accuracy':>9}"
    print(header)
    print(f"  {'─'*22}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}")
    for algo in ALGO_ORDER:
        if algo not in sub.index:
            continue
        row = sub.loc[algo]
        spd  = f"{row.speedup:.1f}x" if not math.isnan(row.speedup) else "—"
        acc  = f"{row.mean_acc:.1f}%" if not math.isnan(row.mean_acc) else "N/A"
        std  = f"{row.std_ms:.3f}" if not math.isnan(row.std_ms) else "—"
        print(f"  {ALGO_LABELS[algo]:<22}  {row.mean_ms:>10.3f}  {std:>8}  {spd:>8}  {acc:>9}")
    print()


def print_combined_table(stats: pd.DataFrame) -> None:
    datasets = stats[["dataset","n_rows"]].drop_duplicates().sort_values("n_rows")

    print(f"\n{'═'*110}")
    print("  COMBINED SUMMARY  (warmup run excluded)")
    print(f"{'═'*110}")
    print(f"  {'Dataset':<12}  {'Rows':>7}  {'Feats':>5}  "
          f"{'sk-brute(ms)':>12}  {'sk-kd(ms)':>10}  "
          f"{'Mojo-BF(ms)':>12}  {'BF spd':>7}  "
          f"{'Mojo-KD(ms)':>12}  {'KD spd':>7}  {'Acc':>7}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*7}  {'─'*12}  {'─'*7}  {'─'*7}")

    for _, drow in datasets.iterrows():
        ds  = drow["dataset"]
        sub = stats[stats["dataset"] == ds].set_index("algorithm")

        def get_val(algo, col, fmt=".3f"):
            if algo in sub.index:
                v = sub.loc[algo, col]
                if not math.isnan(v):
                    return f"{v:{fmt}}"
            return "—"

        sk_b  = get_val("sklearn_brute", "mean_ms")
        sk_kd = get_val("sklearn_kdtree", "mean_ms")
        bf_ms = get_val("brute_simd", "mean_ms")
        kd_ms = get_val("kdtree_simd", "mean_ms")
        bf_sp = get_val("brute_simd", "speedup", ".1f") + "x" if "brute_simd" in sub.index else "—"
        kd_sp = get_val("kdtree_simd", "speedup", ".1f") + "x" if "kdtree_simd" in sub.index else "—"
        acc   = get_val("brute_simd", "mean_acc", ".1f") + "%" if "brute_simd" in sub.index else "—"
        nf    = int(sub["n_features"].iloc[0])

        print(f"  {ds:<12}  {drow['n_rows']:>7}  {nf:>5}  "
              f"{sk_b:>12}  {sk_kd:>10}  "
              f"{bf_ms:>12}  {bf_sp:>7}  "
              f"{kd_ms:>12}  {kd_sp:>7}  {acc:>7}")
    print()


# ---------------------------------------------------------------------------
# Per-dataset figure
# ---------------------------------------------------------------------------

def plot_dataset(df_raw: pd.DataFrame, stats: pd.DataFrame,
                 dataset: str, out_dir: str) -> None:
    """
    Two-panel figure for a single dataset:
      Left:  grouped bar chart — mean inference time per algorithm (steady-state)
             with individual run dots overlaid for transparency
      Right: speedup vs sklearn bar chart
    """
    sub_stats = stats[(stats["dataset"] == dataset) &
                      (stats["algorithm"].isin(ALGO_ORDER))].copy()
    sub_stats = sub_stats.set_index("algorithm").reindex(ALGO_ORDER).dropna(subset=["mean_ms"])

    # Raw per-run data (warmup already excluded via stats pipeline)
    sub_raw = df_raw[
        (df_raw["dataset"] == dataset) &
        (df_raw["algorithm"].isin(ALGO_ORDER)) &
        (df_raw["run"] > WARMUP_RUNS)
    ].copy()
    sub_raw["time_ms"] = sub_raw["time_seconds"] * 1000

    algos   = [a for a in ALGO_ORDER if a in sub_stats.index]
    labels  = [ALGO_LABELS[a] for a in algos]
    means   = [sub_stats.loc[a, "mean_ms"]  for a in algos]
    stds    = [sub_stats.loc[a, "std_ms"]   for a in algos]
    speedups = [sub_stats.loc[a, "speedup"] for a in algos]
    colors  = [ALGO_COLORS[a] for a in algos]

    n_rows     = sub_stats["n_rows"].iloc[0]
    n_features = sub_stats["n_features"].iloc[0]
    k          = sub_stats["k"].iloc[0]

    fig, (ax_time, ax_spd) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{dataset}  —  {n_rows:,} rows x {n_features} features,  K={k}",
                 fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(algos))
    bar_w = 0.55

    # ---- Left: time ----
    bars = ax_time.bar(x, means, width=bar_w, color=colors,
                       edgecolor="white", linewidth=0.8, zorder=3)
    ax_time.errorbar(x, means, yerr=stds, fmt="none",
                     color="black", capsize=4, linewidth=1.2, zorder=4)

    # Overlay individual run dots
    jitter = 0.12
    for i, algo in enumerate(algos):
        run_vals = sub_raw[sub_raw["algorithm"] == algo]["time_ms"].values
        ax_time.scatter(
            np.full(len(run_vals), i) + np.random.uniform(-jitter, jitter, len(run_vals)),
            run_vals, color="black", s=18, alpha=0.4, zorder=5
        )

    # Speedup annotation above each bar
    for i, (bar, spd) in enumerate(zip(bars, speedups)):
        if not math.isnan(spd) and algos[i] != BASELINE_ALGO:
            ax_time.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(stds) * 0.15,
                         f"{spd:.1f}x", ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color="#333333")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax_time.set_ylabel("Inference time (ms)", fontsize=10)
    ax_time.set_title("Mean inference time +/- std\n(dots = individual runs, run 1 excluded)",
                       fontsize=9)
    ax_time.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_time.set_axisbelow(True)

    # ---- Right: speedup ----
    mojo_algos  = [a for a in algos if a != BASELINE_ALGO]
    mojo_labels = [ALGO_LABELS[a] for a in mojo_algos]
    mojo_spds   = [sub_stats.loc[a, "speedup"] for a in mojo_algos]
    mojo_colors = [ALGO_COLORS[a] for a in mojo_algos]

    x2 = np.arange(len(mojo_algos))
    bars2 = ax_spd.bar(x2, mojo_spds, width=0.45, color=mojo_colors,
                        edgecolor="white", linewidth=0.8, zorder=3)
    ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
                   label="sklearn baseline (1x)", zorder=2)

    for bar, spd in zip(bars2, mojo_spds):
        ax_spd.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{spd:.1f}x", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax_spd.set_xticks(x2)
    ax_spd.set_xticklabels(mojo_labels, fontsize=8, rotation=15, ha="right")
    ax_spd.set_ylabel("Speedup vs sklearn brute", fontsize=10)
    ax_spd.set_title("Speedup over sklearn brute-force KNN\n(higher is better)", fontsize=9)
    ax_spd.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_spd.set_axisbelow(True)
    ax_spd.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset}_analysis.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined figure (scaling + speedup)
# ---------------------------------------------------------------------------

def plot_combined(df_raw: pd.DataFrame, stats: pd.DataFrame, out_dir: str) -> None:
    """
    Two-panel combined figure:
      Top:    Absolute inference time vs dataset size (line plot)
      Bottom: Speedup vs sklearn for each dataset (grouped bar chart)
    """
    datasets_ordered = get_datasets_ordered(stats)
    n_ds = len(datasets_ordered)

    fig = plt.figure(figsize=(max(14, n_ds * 1.2), 10))
    fig.suptitle("mojoKNN: Scaling & Speedup Across Datasets",
                 fontsize=14, fontweight="bold", y=1.01)

    # ---- Panel 1: Absolute time vs n_rows (log-log line plot) ----
    ax_scale = fig.add_subplot(2, 1, 1)

    for algo in ALGO_ORDER:
        sub = stats[stats["algorithm"] == algo].sort_values("n_rows")
        if sub.empty:
            continue
        ax_scale.plot(
            sub["n_rows"], sub["mean_ms"],
            marker="o", markersize=7, linewidth=2,
            color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
        )
        ax_scale.fill_between(
            sub["n_rows"],
            sub["mean_ms"] - sub["std_ms"],
            sub["mean_ms"] + sub["std_ms"],
            color=ALGO_COLORS[algo], alpha=0.15,
        )

    ax_scale.set_xscale("log")
    ax_scale.set_yscale("log")
    ax_scale.set_xlabel("Dataset size (rows)", fontsize=10)
    ax_scale.set_ylabel("Inference time (ms, log)", fontsize=10)
    ax_scale.set_title("Inference time vs dataset size  (warmup excluded,  shaded = +/-1 std)",
                        fontsize=10)
    ax_scale.legend(fontsize=9, loc="upper left")
    ax_scale.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_scale.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_scale.set_axisbelow(True)

    n_rows_vals = [
        stats[stats["dataset"] == ds]["n_rows"].iloc[0]
        for ds in datasets_ordered
    ]
    ax_scale.set_xticks(n_rows_vals)
    ax_scale.set_xticklabels(
        [f"{ds}\n({r:,})" for ds, r in zip(datasets_ordered, n_rows_vals)],
        fontsize=7, rotation=30, ha="right",
    )
    ax_scale.xaxis.set_minor_locator(mticker.NullLocator())

    # ---- Panel 2: Speedup grouped bar chart ----
    ax_spd = fig.add_subplot(2, 1, 2)

    mojo_algos  = [a for a in ALGO_ORDER if a != BASELINE_ALGO]
    bar_w       = 0.25
    n_mojo      = len(mojo_algos)
    offsets     = np.linspace(-(n_mojo - 1) * bar_w / 2,
                               (n_mojo - 1) * bar_w / 2,
                               n_mojo)
    x = np.arange(n_ds)

    for j, algo in enumerate(mojo_algos):
        spds = []
        for ds in datasets_ordered:
            row = stats[(stats["dataset"] == ds) & (stats["algorithm"] == algo)]
            spds.append(row["speedup"].values[0] if not row.empty else 0.0)

        bars = ax_spd.bar(
            x + offsets[j], spds, width=bar_w,
            color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        for bar, spd in zip(bars, spds):
            if spd > 0:
                ax_spd.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{spd:.1f}x",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax_spd.axhline(1.0, color=ALGO_COLORS[BASELINE_ALGO],
                   linestyle="--", linewidth=1.2,
                   label="sklearn baseline (1x)", zorder=2)
    ax_spd.set_xticks(x)
    ax_spd.set_xticklabels(
        [f"{ds}\n({r:,})" for ds, r in zip(datasets_ordered, n_rows_vals)],
        fontsize=8,
    )
    ax_spd.set_ylabel("Speedup vs sklearn brute KNN", fontsize=10)
    ax_spd.set_title("Mojo speedup vs sklearn — all datasets  (higher is better)", fontsize=10)
    ax_spd.legend(fontsize=9, loc="upper right")
    ax_spd.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_spd.set_axisbelow(True)
    ax_spd.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"combined_analysis.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Multi-dataset method comparison line chart
# Each line is a dataset, x-axis is method, y-axis is time.
# Shows all speedups at a glance.
# ---------------------------------------------------------------------------

def plot_method_comparison(stats: pd.DataFrame, out_dir: str) -> None:
    """
    Line chart: x = algorithm (method), y = mean inference time (ms),
    each line/color = a different dataset.
    Lets you see all speedups at once — lines that drop steeply from sklearn
    to Mojo are big wins.
    """
    datasets_ordered = get_datasets_ordered(stats)

    # Use a colormap for many datasets
    cmap = plt.cm.tab10 if len(datasets_ordered) <= 10 else plt.cm.tab20
    ds_colors = {ds: cmap(i / max(len(datasets_ordered) - 1, 1))
                 for i, ds in enumerate(datasets_ordered)}

    # Use distinct markers
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

    fig, ax = plt.subplots(figsize=(10, 6))

    algo_positions = {algo: i for i, algo in enumerate(ALGO_ORDER)}
    x_ticks = list(range(len(ALGO_ORDER)))

    for idx, ds in enumerate(datasets_ordered):
        sub = stats[(stats["dataset"] == ds) & (stats["algorithm"].isin(ALGO_ORDER))].copy()
        if sub.empty:
            continue

        # Build points in ALGO_ORDER so the line goes straight left-to-right
        xs, ys = [], []
        for a in ALGO_ORDER:
            row = sub[sub["algorithm"] == a]
            if not row.empty:
                xs.append(algo_positions[a])
                ys.append(row["mean_ms"].values[0])

        n_rows = sub["n_rows"].iloc[0]
        marker = markers[idx % len(markers)]
        ax.plot(xs, ys, marker=marker, markersize=8, linewidth=2,
                color=ds_colors[ds], label=f"{ds} ({n_rows:,})",
                alpha=0.85)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], fontsize=9,
                        rotation=15, ha="right")
    ax.set_ylabel("Mean inference time (ms)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Method Comparison Across All Datasets\n"
                 "(each line = one dataset;  steep drops = large Mojo speedup)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", title="Dataset (rows)",
              title_fontsize=9, ncol=2 if len(datasets_ordered) > 5 else 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"method_comparison_lines.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Speedup heatmap (datasets x algorithms)
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(stats: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap showing speedup of each algorithm vs sklearn brute, for every dataset.
    Rows = datasets, columns = algorithms. Color intensity = speedup magnitude.
    """
    datasets_ordered = get_datasets_ordered(stats)
    algos = [a for a in ALGO_ORDER if a != BASELINE_ALGO]

    # Build matrix
    matrix = np.full((len(datasets_ordered), len(algos)), np.nan)
    for i, ds in enumerate(datasets_ordered):
        for j, algo in enumerate(algos):
            row = stats[(stats["dataset"] == ds) & (stats["algorithm"] == algo)]
            if not row.empty:
                matrix[i, j] = row["speedup"].values[0]

    fig, ax = plt.subplots(figsize=(8, max(4, len(datasets_ordered) * 0.5)))

    # Color map: 1x = white, higher = greener
    vmin = min(0.5, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 0.5
    vmax = max(2.0, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 2.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, origin="upper")

    # Annotate cells
    for i in range(len(datasets_ordered)):
        for j in range(len(algos)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < (vmin + vmax) / 3 or val > 2 * vmax / 3 else "black"
                ax.text(j, i, f"{val:.1f}x", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos], fontsize=9,
                        rotation=15, ha="right")
    ax.set_yticks(range(len(datasets_ordered)))
    n_rows_map = {ds: stats[stats["dataset"]==ds]["n_rows"].iloc[0]
                  for ds in datasets_ordered}
    ax.set_yticklabels([f"{ds} ({n_rows_map[ds]:,})" for ds in datasets_ordered],
                        fontsize=9)
    ax.set_title("Speedup vs sklearn brute-force  (higher / greener = better)",
                 fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Speedup (x)", shrink=0.8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"speedup_heatmap.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Accuracy comparison chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(stats: pd.DataFrame, out_dir: str) -> None:
    """
    Grouped bar chart: accuracy by algorithm for each dataset.
    Verifies that Mojo and sklearn agree on predictions.
    """
    datasets_ordered = get_datasets_ordered(stats)

    # Filter to algorithms that report accuracy
    acc_algos = [a for a in ALGO_ORDER
                 if stats[(stats["algorithm"] == a) & stats["mean_acc"].notna()].shape[0] > 0]
    if not acc_algos:
        print("  Skipping accuracy chart — no accuracy data available.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(datasets_ordered) * 1.2), 5))

    bar_w = 0.18
    n_algos = len(acc_algos)
    offsets = np.linspace(-(n_algos - 1) * bar_w / 2,
                           (n_algos - 1) * bar_w / 2, n_algos)
    x = np.arange(len(datasets_ordered))

    for j, algo in enumerate(acc_algos):
        accs = []
        for ds in datasets_ordered:
            row = stats[(stats["dataset"] == ds) & (stats["algorithm"] == algo)]
            if not row.empty and not math.isnan(row["mean_acc"].values[0]):
                accs.append(row["mean_acc"].values[0])
            else:
                accs.append(0)

        ax.bar(x + offsets[j], accs, width=bar_w,
               color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
               edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_ordered, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Classification Accuracy by Algorithm and Dataset\n"
                 "(should be nearly identical across implementations)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=105)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"accuracy_comparison.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Per-group comparison (finance vs classic_ml vs image)
# ---------------------------------------------------------------------------

def plot_group_summary(stats: pd.DataFrame, out_dir: str) -> None:
    """
    For each group, produce a focused bar chart comparing all algorithms
    across datasets within that group.
    """
    if "group" not in stats.columns:
        return
    groups = [g for g in stats["group"].dropna().unique() if g != ""]
    if not groups:
        return

    for group_name in groups:
        g_stats = stats[stats["group"] == group_name]
        g_datasets = get_datasets_ordered(g_stats)
        if len(g_datasets) < 2:
            continue

        fig, (ax_time, ax_spd) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Group: {group_name.replace('_', ' ').title()}",
                     fontsize=13, fontweight="bold")

        # Time comparison
        bar_w = 0.18
        n_algos = len(ALGO_ORDER)
        offsets = np.linspace(-(n_algos - 1) * bar_w / 2,
                               (n_algos - 1) * bar_w / 2, n_algos)
        x = np.arange(len(g_datasets))

        for j, algo in enumerate(ALGO_ORDER):
            times = []
            for ds in g_datasets:
                row = g_stats[(g_stats["dataset"] == ds) & (g_stats["algorithm"] == algo)]
                times.append(row["mean_ms"].values[0] if not row.empty else 0)

            ax_time.bar(x + offsets[j], times, width=bar_w,
                       color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
                       edgecolor="white", linewidth=0.8, zorder=3)

        ax_time.set_xticks(x)
        n_rows_map = {ds: g_stats[g_stats["dataset"]==ds]["n_rows"].iloc[0]
                      for ds in g_datasets}
        ax_time.set_xticklabels(
            [f"{ds}\n({n_rows_map[ds]:,})" for ds in g_datasets],
            fontsize=8)
        ax_time.set_ylabel("Inference time (ms)", fontsize=10)
        ax_time.set_title("Inference Time by Algorithm", fontsize=10)
        ax_time.legend(fontsize=7, loc="upper left")
        ax_time.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_time.set_axisbelow(True)

        # Speedup comparison
        mojo_algos = [a for a in ALGO_ORDER if a != BASELINE_ALGO]
        bar_w2 = 0.25
        offsets2 = np.linspace(-(len(mojo_algos) - 1) * bar_w2 / 2,
                                (len(mojo_algos) - 1) * bar_w2 / 2,
                                len(mojo_algos))
        for j, algo in enumerate(mojo_algos):
            spds = []
            for ds in g_datasets:
                row = g_stats[(g_stats["dataset"] == ds) & (g_stats["algorithm"] == algo)]
                spds.append(row["speedup"].values[0] if not row.empty else 0)
            bars = ax_spd.bar(x + offsets2[j], spds, width=bar_w2,
                             color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
                             edgecolor="white", linewidth=0.8, zorder=3)
            for bar, spd in zip(bars, spds):
                if spd > 0:
                    ax_spd.text(bar.get_x() + bar.get_width() / 2,
                               bar.get_height() + 0.05,
                               f"{spd:.1f}x", ha="center", va="bottom",
                               fontsize=8, fontweight="bold")

        ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
                       label="sklearn baseline (1x)")
        ax_spd.set_xticks(x)
        ax_spd.set_xticklabels(
            [f"{ds}\n({n_rows_map[ds]:,})" for ds in g_datasets],
            fontsize=8)
        ax_spd.set_ylabel("Speedup vs sklearn brute", fontsize=10)
        ax_spd.set_title("Speedup (higher = better)", fontsize=10)
        ax_spd.legend(fontsize=7, loc="upper left")
        ax_spd.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_spd.set_axisbelow(True)
        ax_spd.set_ylim(bottom=0)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"group_{group_name}.{FIG_EXT}")
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Scaling-group figure  (the "money" plot for same-feature scaling experiments)
# ---------------------------------------------------------------------------

def plot_scaling_group(df_raw: pd.DataFrame, stats: pd.DataFrame,
                       group_name: str, out_dir: str) -> None:
    """
    For a set of datasets sharing a group label (same n_features, varying n_rows),
    produce a 2-panel figure:
      Left:  Absolute inference time vs n_rows (log-log) with +/-1 std band.
      Right: Speedup vs sklearn as a function of n_rows.
    """
    group_datasets = sorted(
        df_raw[df_raw["group"] == group_name][["dataset", "n_rows"]]
        .drop_duplicates()
        .sort_values("n_rows")
        .itertuples(index=False),
        key=lambda r: r.n_rows,
    )
    if not group_datasets:
        return

    # Only meaningful if n_rows actually varies
    n_rows_vals = [r.n_rows for r in group_datasets]
    if len(set(n_rows_vals)) < 2:
        return

    ds_names    = [r.dataset   for r in group_datasets]
    n_features  = stats[stats["group"] == group_name]["n_features"].iloc[0]
    k           = stats[stats["group"] == group_name]["k"].iloc[0]

    fig, (ax_time, ax_spd) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Scaling experiment — {n_features} features, K={k}  "
        f"({group_name.replace('_', ' ').title()})",
        fontsize=13, fontweight="bold",
    )

    # ---- Left: time vs n_rows ----
    for algo in ALGO_ORDER:
        sub = stats[(stats["group"] == group_name) & (stats["algorithm"] == algo)]
        if sub.empty:
            continue
        sub = sub.sort_values("n_rows")
        ax_time.plot(sub["n_rows"], sub["mean_ms"],
                     marker="o", markersize=7, linewidth=2.2,
                     color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
        ax_time.fill_between(
            sub["n_rows"],
            (sub["mean_ms"] - sub["std_ms"]).clip(lower=0),
            sub["mean_ms"] + sub["std_ms"],
            color=ALGO_COLORS[algo], alpha=0.13,
        )

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Training set size (rows)", fontsize=11)
    ax_time.set_ylabel("Mean inference time (ms)", fontsize=11)
    ax_time.set_title("Inference time vs dataset size\n(log-log, shaded = +/-1 std, warmup excluded)",
                       fontsize=9.5)
    ax_time.set_xticks(n_rows_vals)
    ax_time.set_xticklabels([f"{r:,}" for r in n_rows_vals], fontsize=8.5)
    ax_time.xaxis.set_minor_locator(mticker.NullLocator())
    ax_time.legend(fontsize=9.5)
    ax_time.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_time.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_time.set_axisbelow(True)

    # ---- Right: speedup vs n_rows ----
    mojo_algos = [a for a in ALGO_ORDER if a != BASELINE_ALGO]
    for algo in mojo_algos:
        sub = stats[(stats["group"] == group_name) & (stats["algorithm"] == algo)]
        if sub.empty:
            continue
        sub = sub.sort_values("n_rows")
        ax_spd.plot(sub["n_rows"], sub["speedup"],
                    marker="o", markersize=8, linewidth=2.5,
                    color=ALGO_COLORS[algo], label=ALGO_LABELS[algo])
        for _, row in sub.iterrows():
            ax_spd.annotate(
                f"{row['speedup']:.1f}x",
                xy=(row["n_rows"], row["speedup"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.5, fontweight="bold",
                color=ALGO_COLORS[algo],
            )

    ax_spd.axhline(1.0, color=ALGO_COLORS[BASELINE_ALGO],
                   linestyle="--", linewidth=1.2,
                   label="sklearn baseline (1x)")
    ax_spd.set_xscale("log")
    ax_spd.set_xlabel("Training set size (rows)", fontsize=11)
    ax_spd.set_ylabel("Speedup vs sklearn brute KNN", fontsize=11)
    ax_spd.set_title("Mojo speedup vs sklearn as dataset grows\n(higher is better)",
                      fontsize=9.5)
    ax_spd.set_xticks(n_rows_vals)
    ax_spd.set_xticklabels([f"{r:,}" for r in n_rows_vals], fontsize=8.5)
    ax_spd.xaxis.set_minor_locator(mticker.NullLocator())
    ax_spd.legend(fontsize=9.5, loc="upper left")
    ax_spd.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_spd.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_spd.set_axisbelow(True)
    ax_spd.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"scaling_{group_name}.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-dataset boxplot figure (raw run distribution)
# ---------------------------------------------------------------------------

def plot_boxplots(df_raw: pd.DataFrame, out_dir: str) -> None:
    """
    One figure with one subplot per dataset, showing the distribution of
    inference times across runs as a box plot (all runs, including warmup).
    """
    datasets_ordered = (
        df_raw[["dataset", "n_rows"]].drop_duplicates()
        .sort_values("n_rows")["dataset"].tolist()
    )
    n_ds = len(datasets_ordered)
    cols = min(n_ds, 3)
    rows = math.ceil(n_ds / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows),
                              squeeze=False)
    fig.suptitle("Run-to-run timing distribution (all runs, including warmup)",
                 fontsize=13, fontweight="bold")

    for idx, ds in enumerate(datasets_ordered):
        ax = axes[idx // cols][idx % cols]
        sub = df_raw[
            (df_raw["dataset"] == ds) &
            (df_raw["algorithm"].isin(ALGO_ORDER))
        ].copy()
        sub["time_ms"] = sub["time_seconds"] * 1000

        data_by_algo = [
            sub[sub["algorithm"] == a]["time_ms"].values
            for a in ALGO_ORDER if a in sub["algorithm"].values
        ]
        valid_algos = [a for a in ALGO_ORDER if a in sub["algorithm"].values]
        bp = ax.boxplot(data_by_algo, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5))

        for patch, algo in zip(bp["boxes"], valid_algos):
            patch.set_facecolor(ALGO_COLORS[algo])
            patch.set_alpha(0.75)

        n_rows = sub["n_rows"].iloc[0]
        ax.set_title(f"{ds}  ({n_rows:,} rows)", fontsize=10)
        ax.set_xticks(range(1, len(valid_algos) + 1))
        ax.set_xticklabels([ALGO_LABELS[a] for a in valid_algos],
                           fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("Time (ms)", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_ds, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Legend
    legend_handles = [
        Patch(facecolor=ALGO_COLORS[a], alpha=0.75, label=ALGO_LABELS[a])
        for a in ALGO_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(ALGO_ORDER), fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"boxplots_all_datasets.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: Normalized comparison (all datasets on same scale)
# ---------------------------------------------------------------------------

def plot_normalized_comparison(stats: pd.DataFrame, out_dir: str) -> None:
    """
    Bar chart with each dataset's algorithm times normalized to its sklearn_brute
    time (=1.0). Makes it easy to compare relative performance across datasets
    with wildly different absolute times.
    """
    datasets_ordered = get_datasets_ordered(stats)

    fig, ax = plt.subplots(figsize=(max(12, len(datasets_ordered) * 1.3), 5))

    bar_w = 0.18
    n_algos = len(ALGO_ORDER)
    offsets = np.linspace(-(n_algos - 1) * bar_w / 2,
                           (n_algos - 1) * bar_w / 2, n_algos)
    x = np.arange(len(datasets_ordered))

    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for ds in datasets_ordered:
            row = stats[(stats["dataset"] == ds) & (stats["algorithm"] == algo)]
            baseline_row = stats[(stats["dataset"] == ds) &
                                  (stats["algorithm"] == BASELINE_ALGO)]
            if not row.empty and not baseline_row.empty:
                vals.append(row["mean_s"].values[0] / baseline_row["mean_s"].values[0])
            else:
                vals.append(0)

        bars = ax.bar(x + offsets[j], vals, width=bar_w,
                     color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
                     edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_ordered, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Relative time (sklearn brute = 1.0)", fontsize=10)
    ax.set_title("Normalized Performance: All Algorithms Relative to sklearn Brute\n"
                 "(bars below 1.0 = faster than sklearn brute)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"normalized_comparison.{FIG_EXT}")
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Resolve input path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_csv()

    print(f"\nLoading results: {csv_path}")
    df = load_results(csv_path)

    batch_ts = df["batch_timestamp"].iloc[0]
    machine  = df["machine"].iloc[0]
    print(f"Batch: {batch_ts}   Machine: {machine}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Warmup exclusion: first {WARMUP_RUNS} run(s) dropped from statistics\n")

    # Output dir for figures
    out_dir = os.path.join(os.path.dirname(csv_path), "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Compute stats
    stats = compute_stats(df, exclude_warmup=True)

    # ---- Per-dataset: table + figure ----
    datasets_ordered = get_datasets_ordered(stats)

    print("Per-dataset figures:")
    for ds in datasets_ordered:
        print_dataset_table(stats, ds)
        plot_dataset(df, stats, ds, out_dir)

    # ---- Combined summary table ----
    print_combined_table(stats)

    # ---- Scaling-group figures (one per group) ----
    if "group" in df.columns:
        groups = [g for g in df["group"].dropna().unique() if g != ""]
        if groups:
            print("Scaling-group figures:")
            for g in groups:
                plot_scaling_group(df, stats, g, out_dir)

    # ---- Group summary figures ----
    print("Group summary figures:")
    plot_group_summary(stats, out_dir)

    # ---- Combined figures ----
    print("Combined figures:")
    plot_combined(df, stats, out_dir)
    plot_boxplots(df, out_dir)

    # ---- New comparison figures ----
    print("Comparison figures:")
    plot_method_comparison(stats, out_dir)
    plot_speedup_heatmap(stats, out_dir)
    plot_accuracy_comparison(stats, out_dir)
    plot_normalized_comparison(stats, out_dir)

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    np.random.seed(0)   # reproducible jitter in scatter plots
    main()
