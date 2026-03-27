#!/usr/bin/env python3
"""Box plot comparison: Mojo KD-Tree vs all other implementations.
Reads experiment_runner.py output (long-format CSV).

Output: boxplot_full_k50.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_times(csv_path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[~df["algorithm"].isin(["kdtree_build"])]
    df = df[df["run"] > 1]  # exclude warmup
    return {
        algo: grp["time_seconds"].values
        for algo, grp in df.groupby("algorithm")
    }


def main():
    if len(sys.argv) < 2:
        csvs = sorted(Path("results").glob("batch_*.csv"))
        csv_path = str(csvs[-1])
    else:
        csv_path = sys.argv[1]

    times = load_times(csv_path)

    # Read metadata from CSV for subtitle
    df = pd.read_csv(csv_path)
    row0 = df.iloc[0]
    n_rows = int(row0["n_rows"])
    n_train = int(row0["n_train"])
    n_test = int(row0["n_test"])
    n_features = int(row0["n_features"])
    k = int(row0["k"])
    n_runs = len(times.get("kdtree_simd", []))

    # Order: Mojo KD, Mojo Brute, sklearn Brute, sklearn KD
    algo_order = ["kdtree_simd", "brute_simd", "sklearn_brute", "sklearn_kdtree"]
    labels = ["Mojo\nKD-Tree", "Mojo\nBrute", "sklearn\nBrute", "sklearn\nKD-Tree"]
    colors = ["#2ecc71", "#27ae60", "#2980b9", "#3498db"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Plot 1: Execution Time ----
    ax1 = axes[0]
    time_data = [times[a] for a in algo_order]
    bp1 = ax1.boxplot(time_data, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    means = [np.mean(times[a]) for a in algo_order]
    ax1.scatter(range(1, 5), means, color="black", marker="D", s=50, zorder=5, label="Mean")

    for i, m in enumerate(means):
        if m >= 1.0:
            label = f"{m:.2f}s"
        else:
            label = f"{m * 1000:.0f}ms"
        ax1.annotate(label, (i + 1, m), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Execution Time (s)", fontsize=12)
    ax1.set_title(f"KNN Query Time: Mojo vs sklearn\n"
                  f"({n_rows:,} rows, {n_train:,} train, {n_test:,} test, "
                  f"{n_features} features, K={k}, n={n_runs})",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---- Plot 2: Speedup over Mojo KD-Tree ----
    ax2 = axes[1]
    kd_times = times["kdtree_simd"]

    # For cross-framework (independent samples), compute per-run ratios
    # using matched indices (both have same number of runs)
    speedup_data = []
    speedup_labels = []
    sp_colors = []

    comparisons = [
        ("brute_simd",     "vs Mojo\nBrute",    "#27ae60"),
        ("sklearn_brute",  "vs sklearn\nBrute",  "#2980b9"),
        ("sklearn_kdtree", "vs sklearn\nKD-Tree", "#3498db"),
    ]

    for algo, label, color in comparisons:
        other = times[algo]
        # Element-wise ratio (runs are independent across frameworks,
        # but same count — ratio of matched indices is standard practice)
        n = min(len(other), len(kd_times))
        ratios = other[:n] / kd_times[:n]
        speedup_data.append(ratios)
        speedup_labels.append(label)
        sp_colors.append(color)

    bp2 = ax2.boxplot(speedup_data, tick_labels=speedup_labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], sp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=1.5, label="Break-even (1.0x)")

    speedup_means = [np.mean(s) for s in speedup_data]
    ax2.scatter(range(1, 4), speedup_means, color="black", marker="D", s=50, zorder=5,
                label="Mean: " + ", ".join(f"{m:.1f}x" for m in speedup_means))

    for i, m in enumerate(speedup_means):
        ax2.annotate(f"{m:.1f}x", (i + 1, m), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Speedup (other / Mojo KD-Tree)", fontsize=12)
    ax2.set_title("Mojo KD-Tree Speedup vs Other Implementations\n"
                  "(>1.0x = Mojo KD-Tree faster)",
                  fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = "boxplot_full_k50.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
