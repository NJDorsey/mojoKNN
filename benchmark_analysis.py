"""K-D Tree vs Brute Force (Scalar & SIMD): Benchmark Analysis

Dataset : AAPL_LONG (203 586 rows, 22 features, labels: {-1, 0, 1})
Split   : 80/20  →  162 868 train / 40 718 test
K       : 100
Runs    : 30 per implementation

Implementations compared (all 4 run in the same 30-run loop → paired data):
  - Mojo K-D Tree Scalar
  - Mojo K-D Tree SIMD
  - Mojo Brute Force Scalar
  - Mojo Brute Force SIMD

Statistical approach:
  - Friedman test for overall 4-way comparison (non-parametric repeated-measures —
    all 4 variants are measured in the same session, so data is paired).
  - Pairwise Wilcoxon signed-rank tests (post-hoc) with Bonferroni correction
    for C(4,2) = 6 comparisons (α_adj = α / 6).
  - Effect size: Cohen's d on paired differences.
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
print(df)
print()

# =============================================================
# 2. Summary Statistics
# =============================================================
variants = {
    'KD-Tree Scalar': 'kdtree_scalar_time',
    'KD-Tree SIMD':   'kdtree_simd_time',
    'Brute Scalar':   'brute_scalar_time',
    'Brute SIMD':     'brute_simd_time',
}

summary_stats = pd.DataFrame({
    label: {
        'Mean':   df[col].mean(),
        'Std Dev': df[col].std(),
        'Min':    df[col].min(),
        'Max':    df[col].max(),
        'Median': df[col].median(),
    }
    for label, col in variants.items()
})
summary_stats.index.name = 'Metric'

print("Summary Statistics")
print(summary_stats.to_string(float_format='{:.6f}'.format))
print()

# Computed speedup ratios (mean-based)
kd_scalar_mean  = df['kdtree_scalar_time'].mean()
kd_simd_mean    = df['kdtree_simd_time'].mean()
brute_scalar_mean = df['brute_scalar_time'].mean()
brute_simd_mean = df['brute_simd_time'].mean()

print("Speedup Ratios (mean-based):")
print(f"  KD-Tree SIMD    vs KD-Tree Scalar : {kd_scalar_mean / kd_simd_mean:.3f}x")
print(f"  Brute SIMD      vs Brute Scalar   : {brute_scalar_mean / brute_simd_mean:.3f}x")
print(f"  KD-Tree Scalar  vs Brute Scalar   : {brute_scalar_mean / kd_scalar_mean:.3f}x")
print(f"  KD-Tree SIMD    vs Brute SIMD     : {brute_simd_mean / kd_simd_mean:.3f}x")
print()

# =============================================================
# 3. Confidence Intervals (95%)
# =============================================================
def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h, h

ci_results = {
    label: confidence_interval(df[col])
    for label, col in variants.items()
}

ci_df = pd.DataFrame({
    label: {
        'Mean':           ci[0],
        '95% CI Lower':   ci[1],
        '95% CI Upper':   ci[2],
        'Margin of Error': ci[3],
    }
    for label, ci in ci_results.items()
}).T
ci_df.index.name = 'Measure'

print("95% Confidence Intervals")
print(ci_df.to_string(float_format='{:.6f}'.format))
print()

# =============================================================
# 4. Normality Testing (Shapiro-Wilk)
# Shapiro-Wilk is preferred over K-S for small samples (n=30).
# =============================================================
normality_alpha = 0.05

sw_results = {}
for label, col in [
    ('KD-Tree Scalar Times', 'kdtree_scalar_time'),
    ('KD-Tree SIMD Times',   'kdtree_simd_time'),
    ('Brute Scalar Times',   'brute_scalar_time'),
    ('Brute SIMD Times',     'brute_simd_time'),
]:
    stat, p = stats.shapiro(df[col])
    sw_results[label] = (stat, p)

normality_results = pd.DataFrame({
    'Series': list(sw_results.keys()),
    'W-statistic': [v[0] for v in sw_results.values()],
    'p-value': [f'{v[1]:.3e}' for v in sw_results.values()],
    'Normal (α=0.05)': ['Yes' if v[1] > normality_alpha else 'No' for v in sw_results.values()],
}).set_index('Series')

print("Shapiro-Wilk Normality Tests")
print("  (Null hypothesis: data is normally distributed)")
print(normality_results.to_string())
print()

any_non_normal = any(v[1] <= normality_alpha for v in sw_results.values())
if any_non_normal:
    print("  At least one distribution is non-normal — using non-parametric tests (Friedman / Wilcoxon).")
else:
    print("  All distributions appear normal.")
print()

# =============================================================
# 5. Hypothesis Testing
#
# Overall: Friedman test (non-parametric repeated-measures 4-way comparison).
#   H0: all four variants have the same time distribution.
#   H1: at least one differs.
#
# Post-hoc: Pairwise Wilcoxon signed-rank (two-sided) with Bonferroni
#   correction for C(4,2) = 6 comparisons (α_corrected = α / 6).
#
# Note: Friedman and Wilcoxon are both paired tests — consistent with
#   each other and with the data structure (all 4 variants measured in
#   the same 30-run loop).
# =============================================================
alpha     = 0.01   # primary significance threshold
n_pairs   = 6      # C(4,2) pairwise comparisons
alpha_adj = alpha / n_pairs  # Bonferroni-corrected threshold

# --- Friedman test ---
friedman_stat, friedman_p = stats.friedmanchisquare(
    df['kdtree_scalar_time'], df['kdtree_simd_time'],
    df['brute_scalar_time'],  df['brute_simd_time']
)

print(f"Friedman Test (overall repeated-measures, α={alpha})")
print(f"  χ² = {friedman_stat:.4f},  p = {friedman_p:.3e}")
print(f"  {'Significant — proceed to post-hoc tests.' if friedman_p < alpha else 'Not significant.'}")
print()

# --- Pairwise Wilcoxon signed-rank (two-sided) + Bonferroni ---
pairs = [
    ('KD-Tree Scalar', 'KD-Tree SIMD',  'kdtree_scalar_time', 'kdtree_simd_time'),
    ('KD-Tree Scalar', 'Brute Scalar',  'kdtree_scalar_time', 'brute_scalar_time'),
    ('KD-Tree Scalar', 'Brute SIMD',    'kdtree_scalar_time', 'brute_simd_time'),
    ('KD-Tree SIMD',   'Brute Scalar',  'kdtree_simd_time',   'brute_scalar_time'),
    ('KD-Tree SIMD',   'Brute SIMD',    'kdtree_simd_time',   'brute_simd_time'),
    ('Brute Scalar',   'Brute SIMD',    'brute_scalar_time',  'brute_simd_time'),
]

posthoc_rows = []
for name_a, name_b, col_a, col_b in pairs:
    w_stat, p_raw = stats.wilcoxon(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)

    # Cohen's d on paired differences
    diff = df[col_a] - df[col_b]
    d = abs(diff.mean()) / diff.std()
    mean_a, mean_b = df[col_a].mean(), df[col_b].mean()
    faster = name_a if mean_a < mean_b else name_b

    posthoc_rows.append({
        'Comparison': f'{name_a} vs {name_b}',
        'W-statistic': w_stat,
        'p-value (raw)': f'{p_raw:.3e}',
        'p-value (Bonferroni)': f'{p_adj:.3e}',
        f'Sig (α={alpha})': 'Yes' if p_adj < alpha else 'No',
        "Cohen's d": f'{d:.3f}',
        'Faster': faster,
    })

posthoc_df = pd.DataFrame(posthoc_rows).set_index('Comparison')

print(f"Post-hoc: Pairwise Wilcoxon Signed-Rank  (Bonferroni α_adj = {alpha_adj:.4f})")
print(posthoc_df.to_string())
print()

# --- Effect size legend ---
print("Effect size guide (Cohen's d):  Small ≥ 0.2 | Medium ≥ 0.5 | Large ≥ 0.8")
print()

# =============================================================
# 6. Interpretation
# =============================================================
print("=" * 65)
print("INTERPRETATION")
print("=" * 65)
print(f"  Primary test : Friedman  χ²={friedman_stat:.4f}  p={friedman_p:.3e}")
print(f"  Post-hoc     : Wilcoxon signed-rank with Bonferroni (α_adj={alpha_adj:.4f})")
print()

if friedman_p < alpha:
    print("✓ Overall: at least one variant differs significantly (Friedman).")
else:
    print("✗ Overall: no significant difference detected (Friedman).")

print()
print("Speedup Ratios (mean-based):")
print(f"  KD-Tree SIMD   vs KD-Tree Scalar : {kd_scalar_mean / kd_simd_mean:.3f}x  "
      f"({'KD-Tree SIMD faster' if kd_simd_mean < kd_scalar_mean else 'KD-Tree Scalar faster'})")
print(f"  Brute SIMD     vs Brute Scalar   : {brute_scalar_mean / brute_simd_mean:.3f}x  "
      f"({'Brute SIMD faster' if brute_simd_mean < brute_scalar_mean else 'Brute Scalar faster'})")
print(f"  KD-Tree Scalar vs Brute Scalar   : {brute_scalar_mean / kd_scalar_mean:.3f}x  "
      f"({'KD-Tree Scalar faster' if kd_scalar_mean < brute_scalar_mean else 'Brute Scalar faster'})")
print(f"  KD-Tree SIMD   vs Brute SIMD     : {brute_simd_mean / kd_simd_mean:.3f}x  "
      f"({'KD-Tree SIMD faster' if kd_simd_mean < brute_simd_mean else 'Brute SIMD faster'})")

print()
print("Pairwise results (Bonferroni-corrected):")
for row in posthoc_rows:
    sig = row[f'Sig (α={alpha})']
    mark = '✓' if sig == 'Yes' else '✗'
    cohens_d = row["Cohen's d"]
    print(f"  {mark} {row['Comparison']:35s}  p_adj={row['p-value (Bonferroni)']:>10s}  "
          f"d={cohens_d}  faster={row['Faster']}")
print()

# =============================================================
# 7. Box and Whisker Plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution Time Comparison (all 4 variants)
ax1 = axes[0]
times_data = [
    df['kdtree_scalar_time'] * 1000,
    df['kdtree_simd_time']   * 1000,
    df['brute_scalar_time']  * 1000,
    df['brute_simd_time']    * 1000,
]
tick_labels = ['KD-Tree\nScalar', 'KD-Tree\nSIMD', 'Brute\nScalar', 'Brute\nSIMD']
bp1 = ax1.boxplot(times_data, tick_labels=tick_labels, patch_artist=True)

colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means_ms = [df[col].mean() * 1000 for col in variants.values()]
ax1.scatter([1, 2, 3, 4], means_ms, color='black', marker='D', s=50, zorder=5, label='Mean')
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Execution Time Comparison\n(n=30 runs each)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: SIMD speedup for each method family
ax2 = axes[1]
kd_simd_speedup    = df['kdtree_scalar_time'] / df['kdtree_simd_time']
brute_simd_speedup = df['brute_scalar_time']  / df['brute_simd_time']

bp2 = ax2.boxplot(
    [kd_simd_speedup, brute_simd_speedup],
    tick_labels=['KD-Tree SIMD\nSpeedup', 'Brute SIMD\nSpeedup'],
    patch_artist=True
)
bp2['boxes'][0].set_facecolor('#27ae60')
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_facecolor('#c0392b')
bp2['boxes'][1].set_alpha(0.7)

ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)')
ax2.scatter(
    [1, 2],
    [kd_simd_speedup.mean(), brute_simd_speedup.mean()],
    color='black', marker='D', s=50, zorder=5,
    label=f'Mean ({kd_simd_speedup.mean():.2f}x, {brute_simd_speedup.mean():.2f}x)'
)
ax2.set_ylabel('Speedup (Scalar / SIMD)', fontsize=12)
ax2.set_title('SIMD Speedup over Scalar\n(>1.0 = SIMD faster)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('benchmark_boxplot.png', dpi=150, bbox_inches='tight')
print("Plot saved to benchmark_boxplot.png")
plt.show()

# =============================================================
# 8. Accuracy Verification
# =============================================================
acc_cols = {
    'KD-Tree Scalar': 'kdtree_scalar_accuracy',
    'KD-Tree SIMD':   'kdtree_simd_accuracy',
    'Brute Scalar':   'brute_scalar_accuracy',
    'Brute SIMD':     'brute_simd_accuracy',
}

all_match = all(
    (df[col_a] == df[col_b]).all()
    for col_a in acc_cols.values()
    for col_b in acc_cols.values()
)
print(f"All 4 variants produce identical predictions: {'Yes ✓' if all_match else 'No ✗'}")
print()

accuracy_check = pd.DataFrame({
    label: {
        'Mean Accuracy (%)':       df[col].mean(),
        'Std Dev':                  df[col].std(),
        'Consistent Across Runs':  'Yes' if df[col].std() == 0 else 'No',
    }
    for label, col in acc_cols.items()
}).T
accuracy_check.index.name = 'Metric'

print(accuracy_check.to_string())
print()

# =============================================================
# 9. Final Summary Table
# =============================================================
final_summary = pd.DataFrame({
    label: {
        'Mean Execution Time':    f"{ci_results[label][0]*1000:.2f} ms",
        '95% Confidence Interval': (
            f"[{ci_results[label][1]*1000:.2f}, {ci_results[label][2]*1000:.2f}] ms"
        ),
        'Standard Deviation':     f"{df[col].std()*1000:.2f} ms",
        'Median':                 f"{df[col].median()*1000:.2f} ms",
    }
    for label, col in variants.items()
}).T
final_summary.index.name = 'Variant'

print("=" * 75)
print("BENCHMARK RESULTS: Mojo K-D Tree vs Brute Force (Scalar & SIMD)")
print("=" * 75)
print(f"Number of runs        : {len(df)}")
print(f"KD-Tree SIMD  vs Scalar : {kd_scalar_mean / kd_simd_mean:.3f}x")
print(f"Brute SIMD    vs Scalar : {brute_scalar_mean / brute_simd_mean:.3f}x")
print(f"KD-Tree Scalar vs Brute : {brute_scalar_mean / kd_scalar_mean:.3f}x")
print(f"KD-Tree SIMD  vs Brute  : {brute_simd_mean / kd_simd_mean:.3f}x")
print(f"Friedman p-value        : {friedman_p:.3e}  (α={alpha})")
print()
print("Performance Comparison")
print(final_summary.to_string())
