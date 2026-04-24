#!/usr/bin/env python3
"""
Empirical Runtime Complexity Analysis
=====================================
Replicates Kolli et al.'s log-log regression on the AAPL scaling sweep.

Fits T(n) approx C * n^k separately to:
    1. brute_simd                  — Kolli's k ≈ 1.33 reference
    2. kdtree_simd (query-only)    — low C, k approaches log(n) behavior
    3. kdtree_total (build+query)  — higher C, low k

Also prints the crossover n where brute_simd = kdtree_simd.

Usage:
    python runtime_complexity_analysis.py                         # latest batch
    python runtime_complexity_analysis.py results/batch_X.csv
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

GROUP       = "aapl_runtime_complexity"
WARMUP_RUNS = 3     # first N runs per (dataset, algo) are discarded
FIG_DPI     = 150

ALGO_COLORS = {
    "brute_simd":   "#E07B39",
    "kdtree_simd":  "#4CAF7D",
    "kdtree_total": "#2E5C8A",
}
ALGO_LABELS = {
    "brute_simd":   "Mojo brute SIMD",
    "kdtree_simd":  "Mojo KD-tree (query only)",
    "kdtree_total": "Mojo KD-tree (build + query)",
}
ALGO_ORDER = ["brute_simd", "kdtree_simd", "kdtree_total"]


def find_latest_csv(results_dir: str = "results") -> str:
    files = sorted(glob.glob(os.path.join(results_dir, "batch_*.csv")))
    if not files:
        raise FileNotFoundError(f"No batch CSVs found in {results_dir}/")
    return files[-1]


def load_group(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "group" not in df.columns:
        raise ValueError("CSV has no 'group' column.")
    df = df[df["group"] == GROUP].copy()
    if df.empty:
        raise ValueError(f"No rows with group='{GROUP}' in {csv_path}")
    df["run"]          = df["run"].astype(int)
    df["time_seconds"] = df["time_seconds"].astype(float)
    df["n_rows"]       = df["n_rows"].astype(int)
    return df


def compute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (dataset, algorithm), drop warmup runs and return the median time.
    kdtree_build is a single run (run==0), taken as-is.
    """
    rows = []
    for (ds, algo), sub in df.groupby(["dataset", "algorithm"]):
        n = int(sub["n_rows"].iloc[0])
        if algo == "kdtree_build":
            t = float(sub["time_seconds"].iloc[0])
        else:
            keep = sub[sub["run"] > WARMUP_RUNS]
            if keep.empty:
                continue
            t = float(keep["time_seconds"].median())
        rows.append({"dataset": ds, "algorithm": algo, "n_rows": n, "time_s": t})
    return pd.DataFrame(rows)


def build_total_row(medians: pd.DataFrame) -> pd.DataFrame:
    """Synthesize kdtree_total = kdtree_build + kdtree_simd per dataset."""
    out = []
    for ds, sub in medians.groupby("dataset"):
        by_algo = sub.set_index("algorithm")["time_s"]
        if "kdtree_build" in by_algo.index and "kdtree_simd" in by_algo.index:
            out.append({
                "dataset":   ds,
                "algorithm": "kdtree_total",
                "n_rows":    int(sub["n_rows"].iloc[0]),
                "time_s":    float(by_algo["kdtree_build"] + by_algo["kdtree_simd"]),
            })
    return pd.DataFrame(out)


def regress(points: pd.DataFrame) -> dict:
    """Fit log(T) = k * log(n) + log(C). Returns k, C, R^2, and raw points."""
    points = points.sort_values("n_rows")
    n = points["n_rows"].values.astype(float)
    t = points["time_s"].values.astype(float)
    log_n = np.log(n)
    log_t = np.log(t)

    slope, intercept = np.polyfit(log_n, log_t, 1)
    pred   = slope * log_n + intercept
    ss_res = np.sum((log_t - pred) ** 2)
    ss_tot = np.sum((log_t - log_t.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"k": float(slope), "C": math.exp(intercept), "R2": float(r2),
            "n": n, "t": t}


def find_crossover(a: dict, b: dict) -> float | None:
    """n where C_a * n^k_a == C_b * n^k_b  =>  n = exp((log C_b - log C_a)/(k_a - k_b))."""
    dk = a["k"] - b["k"]
    if abs(dk) < 1e-9:
        return None
    log_n = (math.log(b["C"]) - math.log(a["C"])) / dk
    return math.exp(log_n)


def print_table(fits: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Empirical runtime complexity   T(n) = C * n^k")
    print(f"  group = {GROUP}    warmup runs excluded: {WARMUP_RUNS}")
    print(f"{'=' * 78}")
    print(f"  {'Algorithm':<32}  {'k':>7}  {'C':>12}  {'R^2':>6}")
    print(f"  {'-'*32}  {'-'*7}  {'-'*12}  {'-'*6}")
    for algo in ALGO_ORDER:
        if algo not in fits:
            continue
        f = fits[algo]
        print(f"  {ALGO_LABELS[algo]:<32}  {f['k']:>7.4f}  {f['C']:>12.4e}  {f['R2']:>6.3f}")
    print()


def plot(fits: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    all_n = np.concatenate([f["n"] for f in fits.values()])
    n_fit = np.logspace(np.log10(all_n.min()), np.log10(all_n.max()), 200)

    for algo in ALGO_ORDER:
        if algo not in fits:
            continue
        f = fits[algo]
        color = ALGO_COLORS[algo]
        ax.scatter(f["n"], f["t"], color=color, s=70, zorder=5,
                   edgecolor="white", linewidth=1.3)
        t_fit = f["C"] * n_fit ** f["k"]
        eq = (f"{ALGO_LABELS[algo]}:   "
              f"T = {f['C']:.2e} * n^{f['k']:.3f}   "
              f"(R^2 = {f['R2']:.3f})")
        ax.plot(n_fit, t_fit, color=color, linewidth=2.2, label=eq)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size  n  (rows)", fontsize=11)
    ax.set_ylabel("Inference time  T  (seconds)", fontsize=11)
    ax.set_title(
        "Empirical Runtime Complexity — AAPL scaling sweep\n"
        f"(median of 5 post-warmup runs;   group = {GROUP})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
    print(f"\nLoading results: {csv_path}")

    df       = load_group(csv_path)
    medians  = compute_medians(df)
    totals   = build_total_row(medians)
    points   = pd.concat([medians, totals], ignore_index=True)

    n_ds = points["dataset"].nunique()
    print(f"  Group '{GROUP}':  {n_ds} dataset sizes,  "
          f"{points['n_rows'].min():,} -> {points['n_rows'].max():,} rows")

    fits = {}
    for algo in ALGO_ORDER:
        pts = points[points["algorithm"] == algo]
        if len(pts) < 2:
            print(f"  Skipping {algo} — only {len(pts)} point(s); need >= 2.")
            continue
        fits[algo] = regress(pts)

    print_table(fits)

    if "brute_simd" in fits and "kdtree_simd" in fits:
        x = find_crossover(fits["brute_simd"], fits["kdtree_simd"])
        if x is None:
            print("  Crossover: slopes equal — no crossover point.")
        else:
            print(f"  Crossover (brute_simd = kdtree_simd):   n ~= {x:,.0f} rows")
        print()

    out_dir = os.path.join(os.path.dirname(csv_path) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot(fits, os.path.join(out_dir, "runtime_complexity_aapl.png"))


if __name__ == "__main__":
    main()
