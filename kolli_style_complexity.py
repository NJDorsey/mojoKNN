#!/usr/bin/env python3
"""
Kolli-Style Empirical Runtime Complexity (Cross-Dataset Regression)
===================================================================
Replicates the methodology of Kolli et al. Section 3.3 / Fig. 4:
each dataset contributes ONE data point (median inference time vs training
set size) and a single power-law T(n) = C * n^k is fit across all datasets.

This is the "apples-to-apples" comparison to Kolli's k ≈ 1.33 / C ≈ 2.07e-7.
Note: it conflates scaling in n_train, n_features, and n_test — see the
single-dataset sweep in runtime_complexity_analysis.py for a cleaner measurement.

Fits regressions for:
    - brute_simd         (your Mojo brute force)
    - kdtree_simd        (your Mojo kd-tree, query only)

Usage:
    python kolli_style_complexity.py                    # uses the default batch
    python kolli_style_complexity.py results/batch_X.csv
"""

import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless backend — avoids Qt/Wayland errors under WSL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = "results/batch_20260410_131940.csv"
WARMUP_RUNS = 1
FIG_DPI     = 150

ALGOS = ["brute_simd", "kdtree_simd"]
ALGO_COLORS = {
    "brute_simd":    "#E07B39",
    "kdtree_simd":   "#4CAF7D",
}
ALGO_LABELS = {
    "brute_simd":    "Mojo brute SIMD",
    "kdtree_simd":   "Mojo KD-tree (query only)",
}

# Kolli's reported values for overlay / reference
KOLLI_HIGHEND = {"k": 1.33, "C": 2.07e-7, "label": "Kolli high-end (k=1.33)"}
KOLLI_LOWEND  = {"k": 1.19, "C": 3.80e-6, "label": "Kolli low-end  (k=1.19)"}


def load_points(csv_path: str) -> pd.DataFrame:
    """
    One row per (dataset, algorithm) with the median post-warmup time.
    """
    df = pd.read_csv(csv_path)
    df["run"]          = df["run"].astype(int)
    df["time_seconds"] = df["time_seconds"].astype(float)
    df["n_train"]      = df["n_train"].astype(int)

    keep = df[(df["algorithm"].isin(ALGOS)) & (df["run"] > WARMUP_RUNS)]
    grp = (
        keep.groupby(["dataset", "algorithm"])
            .agg(n_train=("n_train", "first"),
                 n_features=("n_features", "first"),
                 time_s=("time_seconds", "median"))
            .reset_index()
    )
    return grp


def regress(points: pd.DataFrame) -> dict:
    points = points.sort_values("n_train")
    n = points["n_train"].values.astype(float)
    t = points["time_s"].values.astype(float)
    log_n, log_t = np.log(n), np.log(t)

    slope, intercept = np.polyfit(log_n, log_t, 1)
    pred   = slope * log_n + intercept
    ss_res = np.sum((log_t - pred) ** 2)
    ss_tot = np.sum((log_t - log_t.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"k": float(slope), "C": math.exp(intercept), "R2": float(r2),
            "n": n, "t": t, "datasets": points["dataset"].tolist(),
            "d": points["n_features"].tolist()}


def print_table(fits: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Kolli-style cross-dataset regression   T(n) = C * n^k")
    print(f"  (one median point per dataset;  warmup runs excluded: {WARMUP_RUNS})")
    print(f"{'=' * 78}")
    print(f"  {'Algorithm':<28}  {'k':>7}  {'C':>12}  {'R^2':>6}  {'points':>6}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*12}  {'-'*6}  {'-'*6}")
    for algo in ALGOS:
        if algo not in fits:
            continue
        f = fits[algo]
        print(f"  {ALGO_LABELS[algo]:<28}  {f['k']:>7.4f}  {f['C']:>12.4e}  "
              f"{f['R2']:>6.3f}  {len(f['n']):>6}")
    print()
    print(f"  Kolli baseline (high-end):  k = {KOLLI_HIGHEND['k']},  "
          f"C = {KOLLI_HIGHEND['C']:.2e}")
    print(f"  Kolli baseline (low-end) :  k = {KOLLI_LOWEND['k']},  "
          f"C = {KOLLI_LOWEND['C']:.2e}")
    print()


def print_dataset_points(fits: dict) -> None:
    algo = "brute_simd" if "brute_simd" in fits else list(fits.keys())[0]
    print(f"  Dataset points for {algo}:")
    print(f"  {'dataset':<12}  {'n_train':>8}  {'d':>5}  {'T (s)':>10}")
    for name, n, d, t in zip(fits[algo]["datasets"], fits[algo]["n"],
                              fits[algo]["d"], fits[algo]["t"]):
        print(f"  {name:<12}  {int(n):>8}  {int(d):>5}  {t:>10.6f}")
    print()


def plot(fits: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    all_n = np.concatenate([f["n"] for f in fits.values()])
    n_fit = np.logspace(np.log10(all_n.min()), np.log10(all_n.max()), 200)

    # Your fits (solid)
    for algo in ALGOS:
        if algo not in fits:
            continue
        f = fits[algo]
        color = ALGO_COLORS[algo]
        ax.scatter(f["n"], f["t"], color=color, s=70, zorder=5,
                   edgecolor="white", linewidth=1.3)
        t_fit = f["C"] * n_fit ** f["k"]
        eq = (f"{ALGO_LABELS[algo]}:   "
              f"T = {f['C']:.2e} n^{f['k']:.3f}   (R^2 = {f['R2']:.3f})")
        ax.plot(n_fit, t_fit, color=color, linewidth=2.2, label=eq)

    # Kolli reference lines (dashed, unobtrusive)
    for ref, style in [(KOLLI_HIGHEND, "--"), (KOLLI_LOWEND, ":")]:
        t_ref = ref["C"] * n_fit ** ref["k"]
        ax.plot(n_fit, t_ref, style, color="#888888", linewidth=1.6,
                label=f"{ref['label']}, C={ref['C']:.2e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size  n_train", fontsize=11)
    ax.set_ylabel("Inference time  T  (seconds)", fontsize=11)
    ax.set_title(
        "Empirical Runtime Complexity — Kolli-Style Cross-Dataset Regression\n"
        "(one median point per dataset;  dashed/dotted = Kolli reported values)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"\nLoading batch: {csv_path}")
    points = load_points(csv_path)

    print(f"  {points['dataset'].nunique()} datasets, "
          f"{len(ALGOS)} algorithms requested.")

    fits = {}
    for algo in ALGOS:
        sub = points[points["algorithm"] == algo]
        if len(sub) < 2:
            print(f"  Skipping {algo} — only {len(sub)} point(s).")
            continue
        fits[algo] = regress(sub)

    print_table(fits)
    print_dataset_points(fits)

    out_dir = os.path.join(os.path.dirname(csv_path) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(fits, os.path.join(out_dir, "kolli_style_complexity.png"))


if __name__ == "__main__":
    main()
