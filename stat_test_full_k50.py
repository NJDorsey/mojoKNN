#!/usr/bin/env python3
"""
Statistical comparison of all four KNN implementations on the full AAPL dataset.
Algorithms: mojo kdtree, mojo brute, sklearn kdtree, sklearn brute
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def load_times(csv_path: str) -> dict[str, np.ndarray]:
    """Load per-algorithm timing arrays, excluding warmup (run 1) and build."""
    df = pd.read_csv(csv_path)
    df = df[~df["algorithm"].isin(["kdtree_build"])]
    df = df[df["run"] > 1]
    return {
        algo: grp["time_seconds"].values
        for algo, grp in df.groupby("algorithm")
    }


def summary_stats(times: dict[str, np.ndarray]):
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'N':>5}")
    print("-" * 70)
    for algo, t in sorted(times.items()):
        print(f"{algo:<20} {np.mean(t):10.4f} {np.std(t, ddof=1):10.4f} "
              f"{np.median(t):10.4f} {np.min(t):10.4f} {np.max(t):10.4f} {len(t):5d}")
    print()


def speedup_table(times: dict[str, np.ndarray]):
    print("=" * 70)
    print("SPEEDUP RATIOS (mean baseline / mean target)")
    print("=" * 70)
    algos = sorted(times.keys())
    kd_mojo = times.get("kdtree_simd")
    comparisons = [
        ("Mojo KD-Tree vs Mojo Brute",    "brute_simd",     "kdtree_simd"),
        ("Mojo KD-Tree vs sklearn KD-Tree","sklearn_kdtree", "kdtree_simd"),
        ("Mojo KD-Tree vs sklearn Brute",  "sklearn_brute",  "kdtree_simd"),
        ("Mojo Brute vs sklearn Brute",    "sklearn_brute",  "brute_simd"),
    ]
    for label, baseline, target in comparisons:
        if baseline in times and target in times:
            speedup = np.mean(times[baseline]) / np.mean(times[target])
            print(f"  {label:<42} {speedup:.2f}x")
    print()


def normality_tests(times: dict[str, np.ndarray]):
    print("=" * 70)
    print("NORMALITY TESTS (Shapiro-Wilk, alpha=0.05)")
    print("=" * 70)
    results = {}
    for algo, t in sorted(times.items()):
        stat, p = stats.shapiro(t)
        normal = "normal" if p > 0.05 else "NOT normal"
        results[algo] = p > 0.05
        print(f"  {algo:<20}  W={stat:.4f}  p={p:.6f}  -> {normal}")
    print()
    return results


def confidence_intervals(times: dict[str, np.ndarray]):
    print("=" * 70)
    print("95% CONFIDENCE INTERVALS FOR MEAN TIME")
    print("=" * 70)
    for algo, t in sorted(times.items()):
        n = len(t)
        mean = np.mean(t)
        se = stats.sem(t)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        print(f"  {algo:<20}  {ci[0]:.4f}s — {ci[1]:.4f}s  (mean={mean:.4f}s)")
    print()


def pairwise_tests(times: dict[str, np.ndarray], is_normal: dict[str, bool]):
    """Run pairwise comparisons with Bonferroni correction."""
    algos = sorted(times.keys())
    pairs = list(combinations(algos, 2))
    n_pairs = len(pairs)
    alpha = 0.05
    alpha_adj = alpha / n_pairs

    print("=" * 70)
    print(f"PAIRWISE COMPARISONS (Bonferroni-corrected alpha = {alpha}/{n_pairs} = {alpha_adj:.4f})")
    print("=" * 70)

    # Within-framework pairs (paired samples) vs cross-framework (independent)
    mojo_algos = {"brute_simd", "kdtree_simd"}
    sklearn_algos = {"sklearn_brute", "sklearn_kdtree"}

    for a, b in pairs:
        ta, tb = times[a], times[b]
        same_framework = (
            (a in mojo_algos and b in mojo_algos) or
            (a in sklearn_algos and b in sklearn_algos)
        )

        if same_framework:
            # Paired test: both ran on same data in same process
            both_normal = is_normal.get(a, False) and is_normal.get(b, False)
            if both_normal:
                stat, p = stats.ttest_rel(ta, tb)
                test_name = "Paired t-test"
            else:
                stat, p = stats.wilcoxon(ta, tb)
                test_name = "Wilcoxon signed-rank"
            # Effect size: Cohen's d for paired
            diff = ta - tb
            d = np.mean(diff) / np.std(diff, ddof=1)
            effect_str = f"Cohen's d = {d:.3f}"
        else:
            # Independent test
            both_normal = is_normal.get(a, False) and is_normal.get(b, False)
            if both_normal:
                stat, p = stats.ttest_ind(ta, tb)
                test_name = "Independent t-test"
            else:
                stat, p = stats.mannwhitneyu(ta, tb, alternative="two-sided")
                test_name = "Mann-Whitney U"
            # Effect size: rank-biserial correlation
            n1, n2 = len(ta), len(tb)
            if test_name == "Mann-Whitney U":
                r = 1 - (2 * stat) / (n1 * n2)
                effect_str = f"rank-biserial r = {r:.3f}"
            else:
                pooled_std = np.sqrt(((len(ta)-1)*np.var(ta, ddof=1) + (len(tb)-1)*np.var(tb, ddof=1)) / (len(ta)+len(tb)-2))
                d = (np.mean(ta) - np.mean(tb)) / pooled_std
                effect_str = f"Cohen's d = {d:.3f}"

        sig = "SIGNIFICANT" if p < alpha_adj else "not significant"
        print(f"\n  {a} vs {b}")
        print(f"    Test: {test_name}")
        print(f"    stat={stat:.4f}  p={p:.2e}  -> {sig}")
        print(f"    {effect_str}")
        speedup = np.mean(times[a]) / np.mean(times[b])
        faster = b if speedup > 1 else a
        print(f"    Speedup: {max(speedup, 1/speedup):.2f}x ({faster} faster)")

    print()


def kruskal_wallis(times: dict[str, np.ndarray]):
    """Overall 4-way comparison."""
    print("=" * 70)
    print("KRUSKAL-WALLIS H-TEST (overall 4-way comparison)")
    print("=" * 70)
    arrays = [t for _, t in sorted(times.items())]
    stat, p = stats.kruskal(*arrays)
    sig = "SIGNIFICANT" if p < 0.01 else "not significant"
    print(f"  H={stat:.4f}  p={p:.2e}  -> {sig} (alpha=0.01)")
    print()


def main():
    if len(sys.argv) < 2:
        from pathlib import Path
        csvs = sorted(Path("results").glob("batch_*.csv"))
        csv_path = str(csvs[-1])
    else:
        csv_path = sys.argv[1]

    print(f"\nStatistical Analysis: {csv_path}\n")

    times = load_times(csv_path)
    summary_stats(times)
    speedup_table(times)
    confidence_intervals(times)
    is_normal = normality_tests(times)
    kruskal_wallis(times)
    pairwise_tests(times, is_normal)


if __name__ == "__main__":
    main()
