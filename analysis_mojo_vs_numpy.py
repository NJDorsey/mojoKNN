"""Mojo (4 variants) vs NumPy KNN: Benchmark Analysis

Dataset : AAPL_LONG (203 586 rows, 22 features, labels: {-1, 0, 1})
Split   : 80/20  →  162 868 train / 40 718 test
K       : 100
Runs    : 30 per implementation

Implementations compared:
  - Mojo K-D Tree Scalar   ]
  - Mojo K-D Tree SIMD     ] paired (same 30-run loop in KNN.mojo)
  - Mojo Brute Force Scalar]
  - Mojo Brute Force SIMD  ]
  - NumPy Brute Force        (independent 30-run loop in numpy_knn_benchmark.py)

Statistical approach:
  - Kruskal-Wallis H-test for overall 5-way comparison (treats all groups as
    independent — conservative but valid since NumPy runs break pairing).
  - Pairwise post-hoc tests with Bonferroni correction for C(5,2) = 10 comparisons:
      Mojo vs Mojo  → Wilcoxon signed-rank (exploits pairing within Mojo runs)
      Mojo vs NumPy → Mann-Whitney U      (independent samples)
  - Effect size: Cohen's d on paired differences (Mojo pairs) or on raw
    differences from the pooled mean (Mojo vs NumPy).

Output: analysis_mojo_vs_numpy.png
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
print(df[['run', 'kdtree_scalar_time', 'kdtree_simd_time',
          'brute_scalar_time', 'brute_simd_time', 'numpy_time']].to_string())
print()

variants = {
    'KD-Tree Scalar': 'kdtree_scalar_time',
    'KD-Tree SIMD':   'kdtree_simd_time',
    'Brute Scalar':   'brute_scalar_time',
    'Brute SIMD':     'brute_simd_time',
    'NumPy':          'numpy_time',
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

numpy_mean        = df['numpy_time'].mean()
kd_scalar_mean    = df['kdtree_scalar_time'].mean()
kd_simd_mean      = df['kdtree_simd_time'].mean()
brute_scalar_mean = df['brute_scalar_time'].mean()
brute_simd_mean   = df['brute_simd_time'].mean()

print("Speedup of Mojo variants over NumPy (mean-based):")
print(f"  KD-Tree Scalar vs NumPy : {numpy_mean / kd_scalar_mean:.3f}x  "
      f"({'Mojo faster' if kd_scalar_mean < numpy_mean else 'NumPy faster'})")
print(f"  KD-Tree SIMD   vs NumPy : {numpy_mean / kd_simd_mean:.3f}x  "
      f"({'Mojo faster' if kd_simd_mean < numpy_mean else 'NumPy faster'})")
print(f"  Brute Scalar   vs NumPy : {numpy_mean / brute_scalar_mean:.3f}x  "
      f"({'Mojo faster' if brute_scalar_mean < numpy_mean else 'NumPy faster'})")
print(f"  Brute SIMD     vs NumPy : {numpy_mean / brute_simd_mean:.3f}x  "
      f"({'Mojo faster' if brute_simd_mean < numpy_mean else 'NumPy faster'})")
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
    'W-statistic':    [v[0] for v in sw_results.values()],
    'p-value':        [f'{v[1]:.3e}' for v in sw_results.values()],
    'Normal (α=0.05)': ['Yes' if v[1] > normality_alpha else 'No' for v in sw_results.values()],
}, index=sw_results.keys())
normality_df.index.name = 'Variant'

print("Shapiro-Wilk Normality Tests")
print(normality_df.to_string())
any_non_normal = any(v[1] <= normality_alpha for v in sw_results.values())
if any_non_normal:
    print("  At least one distribution is non-normal — using non-parametric tests.")
else:
    print("  All distributions appear normal.")
print()

# =============================================================
# 5. Hypothesis Testing
#
# Overall: Kruskal-Wallis (all 5 treated as independent groups).
#
# Post-hoc pairwise with Bonferroni (C(5,2) = 10 comparisons):
#   Mojo vs Mojo  → Wilcoxon signed-rank (paired)
#   Mojo vs NumPy → Mann-Whitney U (independent)
# =============================================================
alpha     = 0.01
n_pairs   = 10   # C(5,2)
alpha_adj = alpha / n_pairs

# --- Kruskal-Wallis ---
kw_stat, kw_p = stats.kruskal(
    df['kdtree_scalar_time'], df['kdtree_simd_time'],
    df['brute_scalar_time'],  df['brute_simd_time'],
    df['numpy_time']
)

print(f"Kruskal-Wallis H-test (overall 5-way, α={alpha})")
print(f"  H = {kw_stat:.4f},  p = {kw_p:.3e}")
print(f"  {'Significant — proceed to post-hoc tests.' if kw_p < alpha else 'Not significant.'}")
print()

# --- Pairwise post-hoc ---
mojo_pairs = [
    ('KD-Tree Scalar', 'KD-Tree SIMD',   'kdtree_scalar_time', 'kdtree_simd_time'),
    ('KD-Tree Scalar', 'Brute Scalar',   'kdtree_scalar_time', 'brute_scalar_time'),
    ('KD-Tree Scalar', 'Brute SIMD',     'kdtree_scalar_time', 'brute_simd_time'),
    ('KD-Tree SIMD',   'Brute Scalar',   'kdtree_simd_time',   'brute_scalar_time'),
    ('KD-Tree SIMD',   'Brute SIMD',     'kdtree_simd_time',   'brute_simd_time'),
    ('Brute Scalar',   'Brute SIMD',     'brute_scalar_time',  'brute_simd_time'),
]
numpy_pairs = [
    ('KD-Tree Scalar', 'NumPy', 'kdtree_scalar_time', 'numpy_time'),
    ('KD-Tree SIMD',   'NumPy', 'kdtree_simd_time',   'numpy_time'),
    ('Brute Scalar',   'NumPy', 'brute_scalar_time',  'numpy_time'),
    ('Brute SIMD',     'NumPy', 'brute_simd_time',    'numpy_time'),
]

posthoc_rows = []

for name_a, name_b, col_a, col_b in mojo_pairs:
    w_stat, p_raw = stats.wilcoxon(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)
    diff  = df[col_a] - df[col_b]
    d     = abs(diff.mean()) / diff.std()
    mean_a, mean_b = df[col_a].mean(), df[col_b].mean()
    faster = name_a if mean_a < mean_b else name_b
    posthoc_rows.append({
        'Comparison':            f'{name_a} vs {name_b}',
        'Test':                  'Wilcoxon (paired)',
        'Statistic':             f'{w_stat:.2f}',
        'p-value (raw)':         f'{p_raw:.3e}',
        'p-value (Bonferroni)':  f'{p_adj:.3e}',
        f'Sig (α={alpha})':      'Yes' if p_adj < alpha else 'No',
        "Cohen's d":             f'{d:.3f}',
        'Faster':                faster,
    })

for name_a, name_b, col_a, col_b in numpy_pairs:
    u_stat, p_raw = stats.mannwhitneyu(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)
    # Effect size r = Z / sqrt(N) — approximated via normal quantile
    z = stats.norm.ppf(1 - p_raw / 2)
    r = abs(z) / np.sqrt(2 * len(df))   # each group has n obs
    mean_a, mean_b = df[col_a].mean(), df[col_b].mean()
    faster = name_a if mean_a < mean_b else name_b
    posthoc_rows.append({
        'Comparison':            f'{name_a} vs {name_b}',
        'Test':                  'Mann-Whitney U',
        'Statistic':             f'{u_stat:.2f}',
        'p-value (raw)':         f'{p_raw:.3e}',
        'p-value (Bonferroni)':  f'{p_adj:.3e}',
        f'Sig (α={alpha})':      'Yes' if p_adj < alpha else 'No',
        "Cohen's d":             f'{r:.3f} (r)',
        'Faster':                faster,
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
print(f"  Post-hoc     : Wilcoxon (Mojo pairs) / Mann-Whitney (vs NumPy)")
print(f"                 Bonferroni α_adj = {alpha_adj:.4f}  ({n_pairs} comparisons)")
print()

if kw_p < alpha:
    print("Overall: at least one variant differs significantly (Kruskal-Wallis).")
else:
    print("Overall: no significant difference detected (Kruskal-Wallis).")

print()
print("Speedup of Mojo vs NumPy (mean-based):")
for label, col in [
    ('KD-Tree Scalar', 'kdtree_scalar_time'),
    ('KD-Tree SIMD',   'kdtree_simd_time'),
    ('Brute Scalar',   'brute_scalar_time'),
    ('Brute SIMD',     'brute_simd_time'),
]:
    sp = numpy_mean / df[col].mean()
    direction = 'Mojo faster' if df[col].mean() < numpy_mean else 'NumPy faster'
    print(f"  {label:<18} vs NumPy : {sp:.3f}x  ({direction})")

print()
print("Pairwise results (Bonferroni-corrected):")
for row in posthoc_rows:
    sig  = row[f'Sig (α={alpha})']
    mark = 'Y' if sig == 'Yes' else 'N'
    print(f"  [{mark}] {row['Comparison']:35s}  p_adj={row['p-value (Bonferroni)']:>10s}  "
          f"effect={row[\"Cohen's d\"]}  faster={row['Faster']}")
print()

# =============================================================
# 7. Box and Whisker Plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Execution Time — all 5 variants
ax1 = axes[0]
times_ms = [df[col] * 1000 for col in variants.values()]
tick_labels = ['KD-Tree\nScalar', 'KD-Tree\nSIMD', 'Brute\nScalar', 'Brute\nSIMD', 'NumPy']
bp1 = ax1.boxplot(times_ms, tick_labels=tick_labels, patch_artist=True)

colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b', '#3498db']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means_ms = [df[col].mean() * 1000 for col in variants.values()]
ax1.scatter(range(1, 6), means_ms, color='black', marker='D', s=50, zorder=5, label='Mean')
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Execution Time: All 5 Variants\n(n=30 runs each)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Mojo speedup over NumPy (per-run ratio)
ax2 = axes[1]
mojo_cols = ['kdtree_scalar_time', 'kdtree_simd_time', 'brute_scalar_time', 'brute_simd_time']
speedup_data = [df['numpy_time'] / df[col] for col in mojo_cols]
speedup_labels = ['KD-Tree\nScalar', 'KD-Tree\nSIMD', 'Brute\nScalar', 'Brute\nSIMD']

bp2 = ax2.boxplot(speedup_data, tick_labels=speedup_labels, patch_artist=True)
for patch, color in zip(bp2['boxes'], ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5, label='No speedup (1.0x)')
speedup_means = [s.mean() for s in speedup_data]
ax2.scatter(range(1, 5), speedup_means, color='black', marker='D', s=50, zorder=5,
            label='Mean: ' + ', '.join(f'{m:.2f}x' for m in speedup_means))
ax2.set_ylabel('Speedup over NumPy (NumPy time / Mojo time)', fontsize=12)
ax2.set_title('Mojo Speedup over NumPy\n(>1.0 = Mojo faster)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('analysis_mojo_vs_numpy.png', dpi=150, bbox_inches='tight')
print("Plot saved to analysis_mojo_vs_numpy.png")
plt.show()

# =============================================================
# 8. Accuracy Comparison
# =============================================================
acc_cols = {
    'KD-Tree Scalar': 'kdtree_scalar_accuracy',
    'KD-Tree SIMD':   'kdtree_simd_accuracy',
    'Brute Scalar':   'brute_scalar_accuracy',
    'Brute SIMD':     'brute_simd_accuracy',
    'NumPy':          'numpy_accuracy',
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

mojo_acc_cols = [col for label, col in acc_cols.items() if label != 'NumPy']
all_mojo_match = all(
    (df[mojo_acc_cols[0]] == df[c]).all()
    for c in mojo_acc_cols[1:]
)
print(f"All 4 Mojo variants produce identical predictions: {'Yes' if all_mojo_match else 'No'}")
print(accuracy_check.to_string())
print()

# =============================================================
# 9. Final Summary Table
# =============================================================
print("=" * 75)
print("BENCHMARK RESULTS: Mojo (4 variants) vs NumPy")
print("=" * 75)
print(f"Number of runs : {len(df)}")
print()
for label, col in variants.items():
    m, lo, hi, _ = ci_results[label]
    sp_label = ''
    if col != 'numpy_time':
        sp = numpy_mean / df[col].mean()
        sp_label = f'  ({sp:.3f}x over NumPy)'
    print(f"  {label:<18}: {m*1000:8.2f} ms  [{lo*1000:.2f}, {hi*1000:.2f}] ms (95% CI){sp_label}")
print()
print(f"Kruskal-Wallis p-value : {kw_p:.3e}  (α={alpha})")
