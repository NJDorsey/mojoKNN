"""Best Mojo Implementations vs NumPy: Benchmark Analysis

Dataset : AAPL_LONG (203 586 rows, 22 features, labels: {-1, 0, 1})
Split   : 80/20  →  162 868 train / 40 718 test
K       : 100
Runs    : 30 per implementation

Implementations compared:
  - Mojo K-D Tree SIMD   ] paired (same 30-run loop in KNN.mojo)
  - Mojo Brute Force SIMD]
  - NumPy Brute Force      (independent 30-run loop in numpy_knn_benchmark.py)

Statistical approach:
  - Kruskal-Wallis H-test for overall 3-way comparison.
  - Pairwise post-hoc tests with Bonferroni correction for C(3,2) = 3 comparisons:
      KD-Tree SIMD vs Brute SIMD → Wilcoxon signed-rank (paired)
      KD-Tree SIMD vs NumPy      → Mann-Whitney U (independent)
      Brute SIMD   vs NumPy      → Mann-Whitney U (independent)

Output: analysis_best_vs_numpy.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('display.precision', 6)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

# =============================================================
# 1. Load Benchmark Data
# =============================================================
df = pd.read_csv('benchmark_results.csv')
print(f"Loaded {len(df)} benchmark runs")
print(df[['run', 'kdtree_simd_time', 'brute_simd_time', 'numpy_time']].to_string())
print()

variants = {
    'KD-Tree SIMD': 'kdtree_simd_time',
    'Brute SIMD':   'brute_simd_time',
    'NumPy':        'numpy_time',
}

# =============================================================
# 2. Summary Statistics
# =============================================================
summary_stats = pd.DataFrame({
    label: {
        'Mean (s)':   df[col].mean(),
        'Std Dev (s)': df[col].std(),
        'Min (s)':    df[col].min(),
        'Max (s)':    df[col].max(),
        'Median (s)': df[col].median(),
    }
    for label, col in variants.items()
}).T
summary_stats.index.name = 'Variant'

print("Summary Statistics (all times in seconds)")
print(summary_stats.to_string(float_format='{:.6f}'.format))
print()

kd_simd_mean   = df['kdtree_simd_time'].mean()
brute_simd_mean = df['brute_simd_time'].mean()
numpy_mean      = df['numpy_time'].mean()

print("Speedup over NumPy (mean-based):")
print(f"  KD-Tree SIMD vs NumPy : {numpy_mean / kd_simd_mean:.3f}x  "
      f"({'Mojo faster' if kd_simd_mean < numpy_mean else 'NumPy faster'})")
print(f"  Brute SIMD   vs NumPy : {numpy_mean / brute_simd_mean:.3f}x  "
      f"({'Mojo faster' if brute_simd_mean < numpy_mean else 'NumPy faster'})")
print(f"  KD-Tree SIMD vs Brute SIMD : {brute_simd_mean / kd_simd_mean:.3f}x  "
      f"({'KD-Tree SIMD faster' if kd_simd_mean < brute_simd_mean else 'Brute SIMD faster'})")
print()

# =============================================================
# 3. Confidence Intervals (95%)
# =============================================================
def confidence_interval(data, confidence=0.95):
    n    = len(data)
    mean = np.mean(data)
    se   = stats.sem(data)
    h    = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h, h

ci_results = {
    label: confidence_interval(df[col])
    for label, col in variants.items()
}

ci_df = pd.DataFrame({
    label: {
        'Mean (s)':        ci[0],
        '95% CI Lower':    ci[1],
        '95% CI Upper':    ci[2],
        'Margin of Error': ci[3],
    }
    for label, ci in ci_results.items()
}).T
ci_df.index.name = 'Variant'

print("95% Confidence Intervals")
print(ci_df.to_string(float_format='{:.6f}'.format))
print()

# =============================================================
# 4. Normality Testing (Shapiro-Wilk)
# =============================================================
normality_alpha = 0.05
sw_results = {}
for label, col in variants.items():
    stat, p = stats.shapiro(df[col])
    sw_results[label] = (stat, p)

normality_df = pd.DataFrame({
    'W-statistic':     [v[0] for v in sw_results.values()],
    'p-value':         [f'{v[1]:.3e}' for v in sw_results.values()],
    'Normal (α=0.05)': ['Yes' if v[1] > normality_alpha else 'No' for v in sw_results.values()],
}, index=sw_results.keys())
normality_df.index.name = 'Variant'

print("Shapiro-Wilk Normality Tests")
print(normality_df.to_string())
if any(v[1] <= normality_alpha for v in sw_results.values()):
    print("  At least one distribution is non-normal — using non-parametric tests.")
else:
    print("  All distributions appear normal.")
print()

# =============================================================
# 5. Hypothesis Testing
#
# Overall: Kruskal-Wallis (3 groups, independent).
#
# Post-hoc with Bonferroni (C(3,2) = 3 comparisons):
#   KD-Tree SIMD vs Brute SIMD → Wilcoxon signed-rank (paired)
#   KD-Tree SIMD vs NumPy      → Mann-Whitney U
#   Brute SIMD   vs NumPy      → Mann-Whitney U
# =============================================================
alpha     = 0.01
n_pairs   = 3
alpha_adj = alpha / n_pairs

# --- Kruskal-Wallis ---
kw_stat, kw_p = stats.kruskal(
    df['kdtree_simd_time'],
    df['brute_simd_time'],
    df['numpy_time']
)

print(f"Kruskal-Wallis H-test (overall 3-way, α={alpha})")
print(f"  H = {kw_stat:.4f},  p = {kw_p:.3e}")
print(f"  {'Significant — proceed to post-hoc tests.' if kw_p < alpha else 'Not significant.'}")
print()

# --- Pairwise post-hoc ---
posthoc_rows = []

# Wilcoxon: KD-Tree SIMD vs Brute SIMD (paired)
w_stat, p_raw = stats.wilcoxon(df['kdtree_simd_time'], df['brute_simd_time'], alternative='two-sided')
p_adj = min(p_raw * n_pairs, 1.0)
diff  = df['kdtree_simd_time'] - df['brute_simd_time']
d     = abs(diff.mean()) / diff.std()
faster = 'KD-Tree SIMD' if kd_simd_mean < brute_simd_mean else 'Brute SIMD'
posthoc_rows.append({
    'Comparison':           'KD-Tree SIMD vs Brute SIMD',
    'Test':                 'Wilcoxon (paired)',
    'Statistic':            f'{w_stat:.2f}',
    'p-value (raw)':        f'{p_raw:.3e}',
    'p-value (Bonferroni)': f'{p_adj:.3e}',
    f'Sig (α={alpha})':     'Yes' if p_adj < alpha else 'No',
    "Effect size":          f"Cohen's d = {d:.3f}",
    'Faster':               faster,
})

# Mann-Whitney: KD-Tree SIMD vs NumPy
u_stat, p_raw = stats.mannwhitneyu(df['kdtree_simd_time'], df['numpy_time'], alternative='two-sided')
p_adj = min(p_raw * n_pairs, 1.0)
z = stats.norm.ppf(1 - p_raw / 2)
r = abs(z) / np.sqrt(2 * len(df))
faster = 'KD-Tree SIMD' if kd_simd_mean < numpy_mean else 'NumPy'
posthoc_rows.append({
    'Comparison':           'KD-Tree SIMD vs NumPy',
    'Test':                 'Mann-Whitney U',
    'Statistic':            f'{u_stat:.2f}',
    'p-value (raw)':        f'{p_raw:.3e}',
    'p-value (Bonferroni)': f'{p_adj:.3e}',
    f'Sig (α={alpha})':     'Yes' if p_adj < alpha else 'No',
    "Effect size":          f'r = {r:.3f}',
    'Faster':               faster,
})

# Mann-Whitney: Brute SIMD vs NumPy
u_stat, p_raw = stats.mannwhitneyu(df['brute_simd_time'], df['numpy_time'], alternative='two-sided')
p_adj = min(p_raw * n_pairs, 1.0)
z = stats.norm.ppf(1 - p_raw / 2)
r = abs(z) / np.sqrt(2 * len(df))
faster = 'Brute SIMD' if brute_simd_mean < numpy_mean else 'NumPy'
posthoc_rows.append({
    'Comparison':           'Brute SIMD vs NumPy',
    'Test':                 'Mann-Whitney U',
    'Statistic':            f'{u_stat:.2f}',
    'p-value (raw)':        f'{p_raw:.3e}',
    'p-value (Bonferroni)': f'{p_adj:.3e}',
    f'Sig (α={alpha})':     'Yes' if p_adj < alpha else 'No',
    "Effect size":          f'r = {r:.3f}',
    'Faster':               faster,
})

posthoc_df = pd.DataFrame(posthoc_rows).set_index('Comparison')
print(f"Post-hoc Pairwise Tests  (Bonferroni α_adj = {alpha_adj:.4f})")
print(posthoc_df.to_string())
print()
print("Effect size guide — Cohen's d: Small ≥ 0.2 | Medium ≥ 0.5 | Large ≥ 0.8")
print("Effect size guide — r:         Small ≥ 0.1 | Medium ≥ 0.3 | Large ≥ 0.5")
print()

# =============================================================
# 6. Interpretation
# =============================================================
print("=" * 65)
print("INTERPRETATION")
print("=" * 65)
print(f"  Overall test : Kruskal-Wallis  H={kw_stat:.4f}  p={kw_p:.3e}")
print(f"  Post-hoc     : Wilcoxon (paired SIMD pair) / Mann-Whitney (vs NumPy)")
print(f"                 Bonferroni α_adj = {alpha_adj:.4f}  ({n_pairs} comparisons)")
print()

if kw_p < alpha:
    print("Overall: at least one variant differs significantly (Kruskal-Wallis).")
else:
    print("Overall: no significant difference detected (Kruskal-Wallis).")

print()
print("Pairwise results (Bonferroni-corrected):")
for row in posthoc_rows:
    sig  = row[f'Sig (α={alpha})']
    mark = 'Y' if sig == 'Yes' else 'N'
    print(f"  [{mark}] {row['Comparison']:35s}  p_adj={row['p-value (Bonferroni)']:>10s}  "
          f"{row['Effect size']}  faster={row['Faster']}")
print()

# =============================================================
# 7. Box and Whisker Plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution Time — 3 variants
ax1 = axes[0]
times_ms = [df[col] * 1000 for col in variants.values()]
tick_labels = ['KD-Tree\nSIMD', 'Brute\nSIMD', 'NumPy']
bp1 = ax1.boxplot(times_ms, tick_labels=tick_labels, patch_artist=True)

colors = ['#27ae60', '#c0392b', '#3498db']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means_ms = [df[col].mean() * 1000 for col in variants.values()]
ax1.scatter(range(1, 4), means_ms, color='black', marker='D', s=50, zorder=5, label='Mean')
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Best Mojo Variants vs NumPy\n(n=30 runs each)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Speedup of Mojo SIMD variants over NumPy
ax2 = axes[1]
kd_speedup    = df['numpy_time'] / df['kdtree_simd_time']
brute_speedup = df['numpy_time'] / df['brute_simd_time']

bp2 = ax2.boxplot(
    [kd_speedup, brute_speedup],
    tick_labels=['KD-Tree SIMD\nvs NumPy', 'Brute SIMD\nvs NumPy'],
    patch_artist=True
)
bp2['boxes'][0].set_facecolor('#27ae60')
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_facecolor('#c0392b')
bp2['boxes'][1].set_alpha(0.7)

ax2.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5, label='No speedup (1.0x)')
ax2.scatter(
    [1, 2],
    [kd_speedup.mean(), brute_speedup.mean()],
    color='black', marker='D', s=50, zorder=5,
    label=f'Mean ({kd_speedup.mean():.2f}x, {brute_speedup.mean():.2f}x)'
)
ax2.set_ylabel('Speedup (NumPy time / Mojo time)', fontsize=12)
ax2.set_title('Mojo SIMD Speedup over NumPy\n(>1.0 = Mojo faster)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('analysis_best_vs_numpy.png', dpi=150, bbox_inches='tight')
print("Plot saved to analysis_best_vs_numpy.png")
plt.show()

# =============================================================
# 8. Accuracy Comparison
# =============================================================
acc_cols = {
    'KD-Tree SIMD': 'kdtree_simd_accuracy',
    'Brute SIMD':   'brute_simd_accuracy',
    'NumPy':        'numpy_accuracy',
}

accuracy_check = pd.DataFrame({
    label: {
        'Mean Accuracy (%)':      df[col].mean(),
        'Std Dev':                 df[col].std(),
        'Consistent Across Runs': 'Yes' if df[col].std() == 0 else 'No',
    }
    for label, col in acc_cols.items()
}).T
accuracy_check.index.name = 'Variant'

print(accuracy_check.to_string())
print()

# =============================================================
# 9. Final Summary Table
# =============================================================
print("=" * 65)
print("BENCHMARK RESULTS: Best Mojo SIMD Variants vs NumPy")
print("=" * 65)
print(f"Number of runs : {len(df)}")
print()
for label, col in variants.items():
    m, lo, hi, _ = ci_results[label]
    sp_label = ''
    if col != 'numpy_time':
        sp = numpy_mean / df[col].mean()
        sp_label = f'  ({sp:.3f}x over NumPy)'
    print(f"  {label:<14}: {m*1000:8.2f} ms  [{lo*1000:.2f}, {hi*1000:.2f}] ms (95% CI){sp_label}")
print()
print(f"Kruskal-Wallis p-value : {kw_p:.3e}  (α={alpha})")
