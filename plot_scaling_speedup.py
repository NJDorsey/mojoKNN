#!/usr/bin/env python3
"""
Plot speedup vs dataset size from experiment_runner.py results.

Produces three line graphs:
  1. Mojo KD-tree vs Mojo Brute-force speedup  (the important one)
  2. Mojo KD-tree vs sklearn KD-tree speedup
  3. Mojo KD-tree vs sklearn Brute speedup

Usage:
    python plot_scaling_speedup.py results/batch_YYYYMMDD_HHMMSS.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop build rows and warmup (run 1) for each algorithm/dataset
    df = df[df["algorithm"] != "kdtree_build"]
    df = df[df["run"] > 1]
    return df


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean time per (dataset, n_rows, algorithm), then speedup ratios."""
    means = (
        df.groupby(["dataset", "n_rows", "algorithm"])["time_seconds"]
        .mean()
        .reset_index()
        .rename(columns={"time_seconds": "mean_time"})
    )

    # Pivot so each algorithm is a column
    pivot = means.pivot(index=["dataset", "n_rows"], columns="algorithm",
                        values="mean_time").reset_index()
    pivot = pivot.sort_values("n_rows")

    # Compute speedup ratios (baseline / target)
    if "brute_simd" in pivot.columns and "kdtree_simd" in pivot.columns:
        pivot["mojo_brute_over_kdtree"] = pivot["brute_simd"] / pivot["kdtree_simd"]
    if "sklearn_kdtree" in pivot.columns and "kdtree_simd" in pivot.columns:
        pivot["sklearn_kd_over_mojo_kd"] = pivot["sklearn_kdtree"] / pivot["kdtree_simd"]
    if "sklearn_brute" in pivot.columns and "kdtree_simd" in pivot.columns:
        pivot["sklearn_brute_over_mojo_kd"] = pivot["sklearn_brute"] / pivot["kdtree_simd"]

    return pivot


def plot_speedup_line(ax, sizes, speedups, label, color, marker="o"):
    ax.plot(sizes, speedups, marker=marker, color=color, linewidth=2,
            markersize=8, label=label, zorder=3)
    # Annotate each point
    for x, y in zip(sizes, speedups):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
                    color=color)


def make_plots(pivot: pd.DataFrame, output_prefix: str):
    sizes = pivot["n_rows"].values
    size_labels = []
    for s in sizes:
        if s >= 1000:
            size_labels.append(f"{s // 1000}K")
        else:
            size_labels.append(str(s))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.suptitle("KD-Tree Speedup vs Dataset Size (AAPL HFT, 16 features)",
                 fontsize=14, fontweight="bold", y=1.02)

    # --- Graph 1: Mojo KD-tree vs Mojo Brute ---
    ax = axes[0]
    if "mojo_brute_over_kdtree" in pivot.columns:
        vals = pivot["mojo_brute_over_kdtree"].values
        plot_speedup_line(ax, sizes, vals,
                          "Mojo Brute / Mojo KD-Tree", "#2196F3")
    ax.set_title("Mojo KD-Tree vs Mojo Brute\n(isolates algorithmic speedup)",
                 fontsize=11)
    ax.set_xlabel("Dataset Size (n rows)")
    ax.set_ylabel("Speedup (x)")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Graph 2: Mojo KD-tree vs sklearn KD-tree ---
    ax = axes[1]
    if "sklearn_kd_over_mojo_kd" in pivot.columns:
        vals = pivot["sklearn_kd_over_mojo_kd"].values
        plot_speedup_line(ax, sizes, vals,
                          "sklearn KD-Tree / Mojo KD-Tree", "#FF9800", marker="s")
    ax.set_title("Mojo KD-Tree vs sklearn KD-Tree\n(language + algorithmic differences)",
                 fontsize=11)
    ax.set_xlabel("Dataset Size (n rows)")
    ax.set_ylabel("Speedup (x)")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Graph 3: Mojo KD-tree vs sklearn Brute ---
    ax = axes[2]
    if "sklearn_brute_over_mojo_kd" in pivot.columns:
        vals = pivot["sklearn_brute_over_mojo_kd"].values
        plot_speedup_line(ax, sizes, vals,
                          "sklearn Brute / Mojo KD-Tree", "#4CAF50", marker="D")
    ax.set_title("Mojo KD-Tree vs sklearn Brute\n(language + algorithmic differences)",
                 fontsize=11)
    ax.set_xlabel("Dataset Size (n rows)")
    ax.set_ylabel("Speedup (x)")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{output_prefix}_scaling_speedup.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        # Find the most recent batch CSV
        results_dir = Path("results")
        csvs = sorted(results_dir.glob("batch_*.csv"))
        if not csvs:
            print("No batch CSV found. Run experiment_runner.py first.")
            sys.exit(1)
        csv_path = str(csvs[-1])
        print(f"Using most recent: {csv_path}")
    else:
        csv_path = sys.argv[1]

    df = load_and_prepare(csv_path)
    pivot = compute_speedups(df)

    print("\nMean times (excluding warmup run 1):")
    print(pivot.to_string(index=False))
    print()

    output_prefix = Path(csv_path).stem
    make_plots(pivot, output_prefix)


if __name__ == "__main__":
    main()
