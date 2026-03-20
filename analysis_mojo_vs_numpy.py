"""Mojo vs sklearn KNN: Benchmark Analysis

Dataset : AAPL_LONG (203 586 rows, 16 features after trimming, labels: {-1, 0, 1})
Split   : 90/10  ->  183 227 train / 20 359 test
K       : 100
Runs    : 30 per implementation

Implementations compared:
  - Mojo K-D Tree    (SIMD leaf scan, contiguous buffer, leaf_size)
  - Mojo Brute Force (SIMD distance + max-heap top-K)
  - sklearn KD-tree  (KNeighborsClassifier, algorithm='kd_tree', n_jobs=-1)
  - sklearn Brute    (KNeighborsClassifier, algorithm='brute',   n_jobs=-1)

Statistical approach:
  - Kruskal-Wallis H-test for overall 4-way comparison (treats all groups as
    independent — conservative but valid since Mojo and sklearn runs are from
    separate processes).
  - Pairwise post-hoc tests with Bonferroni correction for C(4,2) = 6 comparisons:
      Within-framework  -> Wilcoxon signed-rank (paired within the same run loop)
      Cross-framework   -> Mann-Whitney U       (independent samples)
  - Effect size: Cohen's d (paired) or rank-biserial r (independent).

Features trimmed from 22 -> 16:
  Removed columns (0-indexed from original 22):
    5  - longest cumulative return window (highly correlated with cols 0-4)
    17 - integer feature with only 3 unique values (low discriminative power)
    18 - minute within the hour (time-of-day index)
    19 - minute of trading day  (redundant with col 18)
    20 - ALL ZEROS (no information)
    21 - near-constant binary flag (97.3% zeros)

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
print(df.head(5).to_string())
print()

variants = {
    'Mojo KD-Tree':   'kdtree_time',
    'Mojo Brute':     'brute_time',
    'sklearn KD-Tree': 'sklearn_kdtree_time',
    'sklearn Brute':   'sklearn_brute_time',
}

# =============================================================
# 2. Summary Statistics
# =============================================================
summary_stats = pd.DataFrame({
    label: {
        'Mean (s)':    df[col].mean(),
        'Std Dev (s)': df[col].std(),
        'Min (s)':     df[col].min(),
        'Max (s)':     df[col].max(),
        'Median (s)':  df[col].median(),
    }
    for label, col in variants.items()
}).T
summary_stats.index.name = 'Variant'

print("Summary Statistics (all times in seconds)")
print(summary_stats.to_string(float_format='{:.6f}'.format))
print()

mojo_kd_mean     = df['kdtree_time'].mean()
mojo_brute_mean  = df['brute_time'].mean()
sk_kd_mean       = df['sklearn_kdtree_time'].mean()
sk_brute_mean    = df['sklearn_brute_time'].mean()

print("Speedup Ratios (mean-based):")
print(f"  Mojo KD-Tree   vs Mojo Brute   : {mojo_brute_mean / mojo_kd_mean:.3f}x  "
      f"({'Mojo KD faster' if mojo_kd_mean < mojo_brute_mean else 'Mojo Brute faster'})")
print(f"  Mojo KD-Tree   vs sklearn KD   : {sk_kd_mean / mojo_kd_mean:.3f}x  "
      f"({'Mojo faster' if mojo_kd_mean < sk_kd_mean else 'sklearn faster'})")
print(f"  Mojo Brute     vs sklearn Brute: {sk_brute_mean / mojo_brute_mean:.3f}x  "
      f"({'Mojo faster' if mojo_brute_mean < sk_brute_mean else 'sklearn faster'})")
print(f"  sklearn KD     vs sklearn Brute: {sk_kd_mean / sk_brute_mean:.3f}x  "
      f"({'KD faster' if sk_kd_mean < sk_brute_mean else 'Brute faster'})")
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
    'Normal (a=0.05)': ['Yes' if v[1] > normality_alpha else 'No' for v in sw_results.values()],
}, index=sw_results.keys())
normality_df.index.name = 'Variant'

print("Shapiro-Wilk Normality Tests")
print(normality_df.to_string())
any_non_normal = any(v[1] <= normality_alpha for v in sw_results.values())
if any_non_normal:
    print("  -> At least one distribution is non-normal; using non-parametric tests.")
else:
    print("  -> All distributions appear normal.")
print()

# =============================================================
# 5. Hypothesis Testing
#
# Overall: Kruskal-Wallis (all 4 as independent groups — conservative).
# Post-hoc with Bonferroni (C(4,2) = 6 comparisons):
#   Within-framework -> Wilcoxon signed-rank (paired)
#   Cross-framework  -> Mann-Whitney U (independent)
# =============================================================
alpha     = 0.01
n_pairs   = 6   # C(4,2)
alpha_adj = alpha / n_pairs

# --- Kruskal-Wallis ---
kw_stat, kw_p = stats.kruskal(
    df['kdtree_time'], df['brute_time'],
    df['sklearn_kdtree_time'], df['sklearn_brute_time']
)

print(f"Kruskal-Wallis H-test (overall 4-way, a={alpha})")
print(f"  H = {kw_stat:.4f},  p = {kw_p:.3e}")
print(f"  {'Significant -> proceed to post-hoc tests.' if kw_p < alpha else 'Not significant.'}")
print()

# --- Define pairwise comparisons ---
# Within-framework: paired (Wilcoxon signed-rank)
paired_tests = [
    ('Mojo KD-Tree',    'Mojo Brute',      'kdtree_time',         'brute_time'),
    ('sklearn KD-Tree', 'sklearn Brute',    'sklearn_kdtree_time', 'sklearn_brute_time'),
]
# Cross-framework: independent (Mann-Whitney U)
independent_tests = [
    ('Mojo KD-Tree',  'sklearn KD-Tree', 'kdtree_time',  'sklearn_kdtree_time'),
    ('Mojo KD-Tree',  'sklearn Brute',   'kdtree_time',  'sklearn_brute_time'),
    ('Mojo Brute',    'sklearn KD-Tree', 'brute_time',   'sklearn_kdtree_time'),
    ('Mojo Brute',    'sklearn Brute',   'brute_time',   'sklearn_brute_time'),
]

posthoc_rows = []

for name_a, name_b, col_a, col_b in paired_tests:
    w_stat, p_raw = stats.wilcoxon(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)
    diff  = df[col_a] - df[col_b]
    d     = abs(diff.mean()) / diff.std()
    mean_a, mean_b = df[col_a].mean(), df[col_b].mean()
    faster = name_a if mean_a < mean_b else name_b
    posthoc_rows.append({
        'Comparison':           f'{name_a} vs {name_b}',
        'Test':                 'Wilcoxon (paired)',
        'Statistic':            f'{w_stat:.2f}',
        'p-value (raw)':        f'{p_raw:.3e}',
        'p-value (Bonferroni)': f'{p_adj:.3e}',
        f'Sig (a={alpha})':     'Yes' if p_adj < alpha else 'No',
        'Effect Size':          f'd={d:.3f}',
        'Faster':               faster,
    })

for name_a, name_b, col_a, col_b in independent_tests:
    u_stat, p_raw = stats.mannwhitneyu(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)
    # Rank-biserial r as effect size
    n1, n2 = len(df[col_a]), len(df[col_b])
    r = 1 - (2 * u_stat) / (n1 * n2)
    mean_a, mean_b = df[col_a].mean(), df[col_b].mean()
    faster = name_a if mean_a < mean_b else name_b
    posthoc_rows.append({
        'Comparison':           f'{name_a} vs {name_b}',
        'Test':                 'Mann-Whitney U',
        'Statistic':            f'{u_stat:.2f}',
        'p-value (raw)':        f'{p_raw:.3e}',
        'p-value (Bonferroni)': f'{p_adj:.3e}',
        f'Sig (a={alpha})':     'Yes' if p_adj < alpha else 'No',
        'Effect Size':          f'r={abs(r):.3f}',
        'Faster':               faster,
    })

posthoc_df = pd.DataFrame(posthoc_rows).set_index('Comparison')
print(f"Post-hoc Pairwise Tests  (Bonferroni a_adj = {alpha_adj:.4f})")
print(posthoc_df.to_string())
print()
print("Effect size guide:")
print("  Cohen's d:      Small >= 0.2 | Medium >= 0.5 | Large >= 0.8")
print("  Rank-biserial r: Small >= 0.1 | Medium >= 0.3 | Large >= 0.5")
print()

# =============================================================
# 6. Interpretation
# =============================================================
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(f"  Overall test : Kruskal-Wallis  H={kw_stat:.4f}  p={kw_p:.3e}")
print(f"  Post-hoc     : Wilcoxon (within-framework) / Mann-Whitney (cross)")
print(f"                 Bonferroni a_adj = {alpha_adj:.4f}  ({n_pairs} comparisons)")
print()

if kw_p < alpha:
    print("Overall: at least one variant differs significantly (Kruskal-Wallis).")
else:
    print("Overall: no significant difference detected (Kruskal-Wallis).")

print()
print("Speedup summary (mean-based):")
print(f"  Mojo KD-Tree   vs sklearn KD-Tree : {sk_kd_mean / mojo_kd_mean:.2f}x  Mojo faster")
print(f"  Mojo KD-Tree   vs Mojo Brute      : {mojo_brute_mean / mojo_kd_mean:.2f}x  KD-Tree faster")
print(f"  Mojo KD-Tree   vs sklearn Brute   : {mojo_kd_mean / sk_brute_mean:.2f}x  "
      f"{'Mojo faster' if mojo_kd_mean < sk_brute_mean else 'sklearn Brute faster'}")
print()

print("Pairwise results (Bonferroni-corrected):")
for row in posthoc_rows:
    sig  = row[f'Sig (a={alpha})']
    mark = 'Y' if sig == 'Yes' else 'N'
    print(f"  [{mark}] {row['Comparison']:40s}  p_adj={row['p-value (Bonferroni)']:>10s}  "
          f"{row['Effect Size']:>10s}  faster={row['Faster']}")
print()

# =============================================================
# 7. Accuracy Comparison
# =============================================================
acc_variants = {
    'Mojo KD-Tree':    'kdtree_accuracy',
    'Mojo Brute':      'brute_accuracy',
    'sklearn KD-Tree': 'sklearn_kdtree_accuracy',
    'sklearn Brute':   'sklearn_brute_accuracy',
}

accuracy_check = pd.DataFrame({
    label: {
        'Mean Accuracy (%)':      df[col].mean(),
        'Std Dev':                 df[col].std(),
        'Consistent Across Runs': 'Yes' if df[col].std() == 0 else 'No',
    }
    for label, col in acc_variants.items()
}).T
accuracy_check.index.name = 'Variant'

print("Accuracy Comparison")
print(accuracy_check.to_string())
print()

# =============================================================
# 8. Box and Whisker Plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Execution Time — all 4 variants
ax1 = axes[0]
times_ms = [df[col] * 1000 for col in variants.values()]
tick_labels = ['Mojo\nKD-Tree', 'Mojo\nBrute', 'sklearn\nKD-Tree', 'sklearn\nBrute']
bp1 = ax1.boxplot(times_ms, tick_labels=tick_labels, patch_artist=True)

colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means_ms = [df[col].mean() * 1000 for col in variants.values()]
ax1.scatter(range(1, 5), means_ms, color='black', marker='D', s=50, zorder=5, label='Mean')

for i, m in enumerate(means_ms):
    ax1.annotate(f'{m:.0f} ms', (i + 1, m), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('KNN Query Time: Mojo vs sklearn\n'
              '(203K train, 20K test, 16 features, K=100, n=30)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Mojo KD-Tree speedup over each other variant (per-run ratio)
ax2 = axes[1]
speedup_vs_mojo_brute = df['brute_time'] / df['kdtree_time']
speedup_vs_sk_kd      = df['sklearn_kdtree_time'] / df['kdtree_time']
speedup_vs_sk_brute   = df['sklearn_brute_time'] / df['kdtree_time']

speedup_data   = [speedup_vs_mojo_brute, speedup_vs_sk_kd, speedup_vs_sk_brute]
speedup_labels = ['vs Mojo\nBrute', 'vs sklearn\nKD-Tree', 'vs sklearn\nBrute']

bp2 = ax2.boxplot(speedup_data, tick_labels=speedup_labels, patch_artist=True)
sp_colors = ['#27ae60', '#3498db', '#2980b9']
for patch, color in zip(bp2['boxes'], sp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5, label='Break-even (1.0x)')
speedup_means = [s.mean() for s in speedup_data]
ax2.scatter(range(1, 4), speedup_means, color='black', marker='D', s=50, zorder=5,
            label='Mean: ' + ', '.join(f'{m:.2f}x' for m in speedup_means))

for i, m in enumerate(speedup_means):
    ax2.annotate(f'{m:.2f}x', (i + 1, m), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

ax2.set_ylabel('Speedup (other / Mojo KD-Tree)', fontsize=12)
ax2.set_title('Mojo KD-Tree Speedup vs Other Implementations\n'
              '(>1.0 = Mojo KD-Tree faster)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('analysis_mojo_vs_numpy.png', dpi=150, bbox_inches='tight')
print("Plot saved to analysis_mojo_vs_numpy.png")
plt.close()

# =============================================================
# 9. Final Summary Table
# =============================================================
print()
print("=" * 70)
print("FINAL RESULTS: Mojo vs sklearn KNN")
print("=" * 70)
print(f"Dataset       : AAPL_LONG (203,586 rows)")
print(f"Features      : 16 (trimmed from 22)")
print(f"Split         : 90/10 (183,227 train / 20,359 test)")
print(f"K             : 100")
print(f"Runs          : {len(df)}")
print()
for label, col in variants.items():
    m, lo, hi, _ = ci_results[label]
    print(f"  {label:<18}: {m*1000:8.2f} ms  [{lo*1000:.2f}, {hi*1000:.2f}] ms (95% CI)")
print()
print(f"Kruskal-Wallis p-value : {kw_p:.3e}  (a={alpha})")
print()
print("Features removed (from original 22-column feature set):")
print("  Col 5  : longest cumulative return window (correlated with cols 0-4)")
print("  Col 17 : integer with 3 unique values (low discriminative power)")
print("  Col 18 : minute within the hour")
print("  Col 19 : minute of trading day (redundant time index)")
print("  Col 20 : all zeros (zero information)")
print("  Col 21 : near-constant binary flag (97.3% zeros)")
