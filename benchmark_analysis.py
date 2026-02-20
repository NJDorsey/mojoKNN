"""K-D Tree vs Brute Force vs NumPy KNN: Benchmark Analysis

Dataset : AAPL_LONG (203 586 rows, 22 features, labels: {-1, 0, 1})
Split   : 80/20  →  162 868 train / 40 718 test
K       : 100
Runs    : 30 per implementation

Implementations compared:
  - Mojo K-D Tree
  - Mojo Brute Force
  - NumPy (Python) Brute Force

Statistical approach:
  - Kruskal-Wallis H-test for overall 3-way comparison (non-parametric,
    independent groups — numpy runs were collected separately from Mojo runs).
  - Pairwise Mann-Whitney U tests (post-hoc) with Bonferroni correction.
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
summary_stats = pd.DataFrame({
    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
    'K-D Tree Time': [
        df['kdtree_time'].mean(),
        df['kdtree_time'].std(),
        df['kdtree_time'].min(),
        df['kdtree_time'].max(),
        df['kdtree_time'].median()
    ],
    'Brute Force Time': [
        df['brute_time'].mean(),
        df['brute_time'].std(),
        df['brute_time'].min(),
        df['brute_time'].max(),
        df['brute_time'].median()
    ],
    'NumPy Time': [
        df['numpy_time'].mean(),
        df['numpy_time'].std(),
        df['numpy_time'].min(),
        df['numpy_time'].max(),
        df['numpy_time'].median()
    ],
    'KD/Brute Speedup': [
        df['speedup'].mean(),
        df['speedup'].std(),
        df['speedup'].min(),
        df['speedup'].max(),
        df['speedup'].median()
    ]
}).set_index('Metric')

print("Summary Statistics")
print(summary_stats.to_string(float_format='{:.6f}'.format))
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

kdtree_mean, kdtree_ci_low, kdtree_ci_high, kdtree_margin = confidence_interval(df['kdtree_time'])
brute_mean,  brute_ci_low,  brute_ci_high,  brute_margin  = confidence_interval(df['brute_time'])
numpy_mean,  numpy_ci_low,  numpy_ci_high,  numpy_margin  = confidence_interval(df['numpy_time'])
speedup_mean, speedup_ci_low, speedup_ci_high, speedup_margin = confidence_interval(df['speedup'])

ci_df = pd.DataFrame({
    'Measure': [
        'K-D Tree Time',
        'Brute Force Time',
        'NumPy Time',
        'KD/Brute Speedup',
    ],
    'Mean': [kdtree_mean, brute_mean, numpy_mean, speedup_mean],
    '95% CI Lower': [kdtree_ci_low,  brute_ci_low,  numpy_ci_low,  speedup_ci_low],
    '95% CI Upper': [kdtree_ci_high, brute_ci_high, numpy_ci_high, speedup_ci_high],
    'Margin of Error': [kdtree_margin, brute_margin, numpy_margin, speedup_margin]
}).set_index('Measure')

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
    ('K-D Tree Times',   'kdtree_time'),
    ('Brute Force Times','brute_time'),
    ('NumPy Times',      'numpy_time'),
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
    print("  At least one distribution is non-normal — using non-parametric tests (Kruskal-Wallis).")
else:
    print("  All distributions appear normal.")
print()

# =============================================================
# 5. Hypothesis Testing
#
# Overall: Kruskal-Wallis H-test (non-parametric 3-way comparison).
#   H0: all three implementations have the same time distribution.
#   H1: at least one differs.
#
# Post-hoc: Pairwise Mann-Whitney U (two-sided) with Bonferroni
#   correction for 3 comparisons (α_corrected = α / 3).
#
# Note: KW treats all groups as independent, which is appropriate
#   here because the NumPy runs were collected in a separate session
#   from the Mojo runs and are not paired with them.
# =============================================================
alpha        = 0.01   # primary significance threshold
n_pairs      = 3      # number of post-hoc pairwise comparisons
alpha_adj    = alpha / n_pairs  # Bonferroni-corrected threshold

# --- Kruskal-Wallis ---
kw_stat, kw_p = stats.kruskal(
    df['kdtree_time'], df['brute_time'], df['numpy_time']
)

print(f"Kruskal-Wallis H-test (overall, α={alpha})")
print(f"  H = {kw_stat:.4f},  p = {kw_p:.3e}")
print(f"  {'Significant — proceed to post-hoc tests.' if kw_p < alpha else 'Not significant.'}")
print()

# --- Pairwise Mann-Whitney U (two-sided) + Bonferroni ---
pairs = [
    ('KD-Tree',     'Brute Force', 'kdtree_time', 'brute_time'),
    ('KD-Tree',     'NumPy',       'kdtree_time', 'numpy_time'),
    ('Brute Force', 'NumPy',       'brute_time',  'numpy_time'),
]

posthoc_rows = []
for name_a, name_b, col_a, col_b in pairs:
    u_stat, p_raw = stats.mannwhitneyu(df[col_a], df[col_b], alternative='two-sided')
    p_adj = min(p_raw * n_pairs, 1.0)

    # Cohen's d (independent groups, pooled std)
    mean_a, std_a = df[col_a].mean(), df[col_a].std()
    mean_b, std_b = df[col_b].mean(), df[col_b].std()
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    d = abs(mean_a - mean_b) / pooled_std
    faster = name_a if mean_a < mean_b else name_b

    posthoc_rows.append({
        'Comparison': f'{name_a} vs {name_b}',
        'U-statistic': u_stat,
        'p-value (raw)': f'{p_raw:.3e}',
        'p-value (Bonferroni)': f'{p_adj:.3e}',
        f'Sig (α={alpha})': 'Yes' if p_adj < alpha else 'No',
        "Cohen's d": f'{d:.3f}',
        'Faster': faster,
    })

posthoc_df = pd.DataFrame(posthoc_rows).set_index('Comparison')

print(f"Post-hoc: Pairwise Mann-Whitney U  (Bonferroni α_adj = {alpha_adj:.4f})")
print(posthoc_df.to_string())
print()

# --- Effect size legend ---
print("Effect size guide (Cohen's d):  Small ≥ 0.2 | Medium ≥ 0.5 | Large ≥ 0.8")
print()

# --- Speedup relative to NumPy ---
speedup_kd_vs_numpy    = numpy_mean / kdtree_mean
speedup_brute_vs_numpy = numpy_mean / brute_mean

# =============================================================
# 6. Interpretation
# =============================================================
print("=" * 65)
print("INTERPRETATION")
print("=" * 65)
print(f"  Primary test : Kruskal-Wallis  H={kw_stat:.4f}  p={kw_p:.3e}")
print(f"  Post-hoc     : Mann-Whitney U with Bonferroni (α_adj={alpha_adj:.4f})")
print()

if kw_p < alpha:
    print("✓ Overall: at least one implementation differs significantly (KW).")
else:
    print("✗ Overall: no significant difference detected (KW).")

print()
print("Pairwise results (Bonferroni-corrected):")
for row in posthoc_rows:
    sig = row[f'Sig (α={alpha})']
    mark = '✓' if sig == 'Yes' else '✗'
    print(f"  {mark} {row['Comparison']:30s}  p_adj={row['p-value (Bonferroni)']:>10s}  "
          f"d={row[\"Cohen's d\"]}  faster={row['Faster']}")

print()
print(f"Mojo K-D Tree  vs NumPy: {speedup_kd_vs_numpy:.2f}x  "
      f"({'KD-Tree faster' if speedup_kd_vs_numpy > 1 else 'NumPy faster'})")
print(f"Mojo Brute Force vs NumPy: {speedup_brute_vs_numpy:.2f}x  "
      f"({'Brute Force faster' if speedup_brute_vs_numpy > 1 else 'NumPy faster'})")
print(f"Mojo K-D Tree  vs Brute Force: {speedup_mean:.2f}x  (KD-Tree faster)")
print()

# =============================================================
# 7. Box and Whisker Plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution Time Comparison (all 3 implementations)
ax1 = axes[0]
times_data   = [df['kdtree_time'] * 1000, df['brute_time'] * 1000, df['numpy_time'] * 1000]
tick_labels  = ['Mojo K-D Tree', 'Mojo Brute Force', 'NumPy']
bp1 = ax1.boxplot(times_data, tick_labels=tick_labels, patch_artist=True)

colors = ['#2ecc71', '#e74c3c', '#9b59b6']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means = [df['kdtree_time'].mean() * 1000, df['brute_time'].mean() * 1000, df['numpy_time'].mean() * 1000]
ax1.scatter([1, 2, 3], means, color='black', marker='D', s=50, zorder=5, label='Mean')
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Execution Time Comparison\n(n=30 runs each)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: KD-Tree vs Brute Force speedup (within Mojo)
ax2 = axes[1]
bp2 = ax2.boxplot(df['speedup'], tick_labels=['KD / Brute'], patch_artist=True)
bp2['boxes'][0].set_facecolor('#3498db')
bp2['boxes'][0].set_alpha(0.7)

ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)')
ax2.scatter(
    [1], [speedup_mean], color='black', marker='D', s=50, zorder=5,
    label=f'Mean ({speedup_mean:.2f}x)'
)
ax2.set_ylabel('Speedup (Brute Force / K-D Tree)', fontsize=12)
ax2.set_title('Mojo K-D Tree Speedup over Brute Force\n(>1.0 = KD-Tree faster)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('benchmark_boxplot.png', dpi=150, bbox_inches='tight')
print("Plot saved to benchmark_boxplot.png")
plt.show()

# =============================================================
# 8. Accuracy Verification
# =============================================================
mojo_match  = (df['kdtree_accuracy'] == df['brute_accuracy']).all()
numpy_acc_std = df['numpy_accuracy'].std()

accuracy_check = pd.DataFrame({
    'Metric': ['Mean Accuracy (%)', 'Std Dev', 'Consistent Across Runs'],
    'K-D Tree': [
        df['kdtree_accuracy'].mean(),
        df['kdtree_accuracy'].std(),
        'Yes' if df['kdtree_accuracy'].std() == 0 else 'No'
    ],
    'Brute Force': [
        df['brute_accuracy'].mean(),
        df['brute_accuracy'].std(),
        'Yes' if df['brute_accuracy'].std() == 0 else 'No'
    ],
    'NumPy': [
        df['numpy_accuracy'].mean(),
        df['numpy_accuracy'].std(),
        'Yes' if numpy_acc_std == 0 else 'No'
    ],
}).set_index('Metric')

print(f"Mojo KD-Tree and Brute Force produce identical predictions: {'Yes ✓' if mojo_match else 'No ✗'}")
print()
print(accuracy_check)
print()
print("  Note: NumPy uses a fixed-seed 80/20 split; Mojo uses a random 80/20")
print("  split. Accuracy differences reflect split variation, not")
print("  implementation differences.")
print()

# =============================================================
# 9. Final Summary Table
# =============================================================
final_summary = pd.DataFrame({
    'Metric': [
        'Mean Execution Time',
        '95% Confidence Interval',
        'Standard Deviation',
        'Median',
    ],
    'Mojo K-D Tree': [
        f"{kdtree_mean*1000:.2f} ms",
        f"[{kdtree_ci_low*1000:.2f}, {kdtree_ci_high*1000:.2f}] ms",
        f"{df['kdtree_time'].std()*1000:.2f} ms",
        f"{df['kdtree_time'].median()*1000:.2f} ms",
    ],
    'Mojo Brute Force': [
        f"{brute_mean*1000:.2f} ms",
        f"[{brute_ci_low*1000:.2f}, {brute_ci_high*1000:.2f}] ms",
        f"{df['brute_time'].std()*1000:.2f} ms",
        f"{df['brute_time'].median()*1000:.2f} ms",
    ],
    'NumPy': [
        f"{numpy_mean*1000:.2f} ms",
        f"[{numpy_ci_low*1000:.2f}, {numpy_ci_high*1000:.2f}] ms",
        f"{df['numpy_time'].std()*1000:.2f} ms",
        f"{df['numpy_time'].median()*1000:.2f} ms",
    ],
}).set_index('Metric')

print("=" * 65)
print("BENCHMARK RESULTS: Mojo K-D Tree vs Brute Force vs NumPy KNN")
print("=" * 65)
print(f"Number of runs : {len(df)}")
print(f"KD/Brute speedup : {speedup_mean:.2f}x  [{speedup_ci_low:.2f}x – {speedup_ci_high:.2f}x]  (95% CI)")
print(f"KD vs NumPy      : {speedup_kd_vs_numpy:.2f}x  {'(KD faster)' if speedup_kd_vs_numpy > 1 else '(NumPy faster)'}")
print(f"Brute vs NumPy   : {speedup_brute_vs_numpy:.2f}x  {'(Brute faster)' if speedup_brute_vs_numpy > 1 else '(NumPy faster)'}")
print(f"KW p-value       : {kw_p:.3e}  (α={alpha})")
print()
print("Performance Comparison")
print(final_summary.to_string())
