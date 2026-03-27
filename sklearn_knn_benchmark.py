"""
sklearn KNN Benchmark
=====================
Benchmarks sklearn's KNeighborsClassifier on the AAPL_LONG dataset with both
kd_tree and brute force algorithms, for comparison against the Mojo KNN
implementation.

  Data   : HFTData/AAPL_LONG_features_causal.csv  (203 586 rows, 22 features)
           HFTData/AAPL_LONG_target_causal.csv     (labels: -1, 0, 1)
  Split  : 95/5  →  ~193 406 train / ~10 180 test
  K      : 100
  Runs   : 30

Times are stored in SECONDS to match the existing benchmark_results.csv.

Run with:
    pixi run python sklearn_knn_benchmark.py
"""

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier

# ── Config (must match KNN.mojo) ─────────────────────────────────────────────
K           = 100
N_RUNS      = 30
TRAIN_FRAC  = 0.90
SEED        = 42
LEAF_SIZE   = 30

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading AAPL_LONG data...", flush=True)
X = np.loadtxt('HFTData/AAPL_LONG_features_causal.csv', delimiter=',', dtype=np.float32)
y = np.loadtxt('HFTData/AAPL_LONG_target_causal.csv',   delimiter=',', dtype=np.float32)
print(f"  X: {X.shape}  y: {y.shape}")

# ── Reproducible 99/1 split ──────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
perm    = rng.permutation(len(X))
n_train = int(len(X) * TRAIN_FRAC)

X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]

print(f"  Train: {len(X_train)}  Test: {len(X_test)}  K={K}", flush=True)

# ── Fit models once (tree construction is not part of the timed query) ───────
print("\nFitting sklearn KD-tree model...", flush=True)
knn_kdtree = KNeighborsClassifier(
    n_neighbors=K,
    algorithm='kd_tree',
    leaf_size=LEAF_SIZE,
    metric='euclidean',
    n_jobs=-1,
)
knn_kdtree.fit(X_train, y_train)

print("Fitting sklearn brute force model...", flush=True)
knn_brute = KNeighborsClassifier(
    n_neighbors=K,
    algorithm='brute',
    metric='euclidean',
    n_jobs=-1,
)
knn_brute.fit(X_train, y_train)

# ── Timed runs ───────────────────────────────────────────────────────────────
print(f"\nRunning {N_RUNS} timed benchmark runs:", flush=True)

kdtree_times = []
kdtree_accs  = []
brute_times  = []
brute_accs   = []

for i in range(N_RUNS):
    # sklearn KD-tree
    t0   = time.perf_counter()
    pred = knn_kdtree.predict(X_test)
    t1   = time.perf_counter()
    kdtree_elapsed = t1 - t0
    kdtree_acc     = float(np.mean(pred == y_test) * 100)
    kdtree_times.append(kdtree_elapsed)
    kdtree_accs.append(kdtree_acc)

    # sklearn brute force
    t0   = time.perf_counter()
    pred = knn_brute.predict(X_test)
    t1   = time.perf_counter()
    brute_elapsed = t1 - t0
    brute_acc     = float(np.mean(pred == y_test) * 100)
    brute_times.append(brute_elapsed)
    brute_accs.append(brute_acc)

    print(f"  Run {i+1:2d}: kdtree={kdtree_elapsed:.4f}s  brute={brute_elapsed:.4f}s  "
          f"acc_kd={kdtree_acc:.2f}%  acc_br={brute_acc:.2f}%", flush=True)

print(f"\n{'='*60}")
print(f"SUMMARY ({N_RUNS} runs)")
print(f"{'='*60}")
print(f"sklearn KD-tree mean:     {np.mean(kdtree_times):.4f} s")
print(f"sklearn brute force mean: {np.mean(brute_times):.4f} s")
print(f"sklearn KD-tree acc:      {np.mean(kdtree_accs):.4f}%")
print(f"sklearn brute force acc:  {np.mean(brute_accs):.4f}%")

# ── Append to benchmark_results.csv ──────────────────────────────────────────
df = pd.read_csv('benchmark_results.csv')

for col in ['sklearn_kdtree_time', 'sklearn_kdtree_accuracy',
            'sklearn_brute_time', 'sklearn_brute_accuracy']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Align lengths: if Mojo CSV has fewer rows, pad sklearn results; if more, truncate
n_mojo = len(df)
if n_mojo < N_RUNS:
    print(f"\nNote: benchmark_results.csv has {n_mojo} rows, sklearn has {N_RUNS}. "
          f"Only first {n_mojo} sklearn runs will be saved.")

df['sklearn_kdtree_time']     = kdtree_times[:n_mojo]
df['sklearn_kdtree_accuracy'] = kdtree_accs[:n_mojo]
df['sklearn_brute_time']      = brute_times[:n_mojo]
df['sklearn_brute_accuracy']  = brute_accs[:n_mojo]

df.to_csv('benchmark_results.csv', index=False)
print(f"\nbenchmark_results.csv updated with sklearn columns.")
