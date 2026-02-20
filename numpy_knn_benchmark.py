"""
NumPy KNN Benchmark
===================
Runs pure-NumPy brute-force KNN on the AAPL_LONG dataset (same data,
same split, same K as the Mojo benchmark in KNN.mojo).

  Data   : HFTData/AAPL_LONG_features_causal.csv  (203 586 rows, 22 features)
           HFTData/AAPL_LONG_target_causal.csv     (labels: -1, 0, 1)
  Split  : 80/20  →  162 868 train / 40 718 test
  K      : 100
  Runs   : 30

Distance computation uses the algebraic identity
    ||a-b||² = ||a||² + ||b||² - 2·aᵀb
and processes the test set in batches to stay within RAM limits
(the full 40 718 × 162 868 distance matrix would be ~26 GB).

Times are stored in SECONDS to match the existing benchmark_results.csv.

Run with:
    /usr/bin/python3 numpy_knn_benchmark.py
"""

import numpy as np
import pandas as pd
import time

# ── Config (must match KNN.mojo) ─────────────────────────────────────────────
K           = 100
N_RUNS      = 30
TRAIN_FRAC  = 0.80
SEED        = 42
BATCH_SIZE  = 256   # test-point batch size; keeps per-batch RAM ~170 MB

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading AAPL_LONG data...", flush=True)
X = np.loadtxt('HFTData/AAPL_LONG_features_causal.csv', delimiter=',', dtype=np.float32)
y = np.loadtxt('HFTData/AAPL_LONG_target_causal.csv',   delimiter=',', dtype=np.float32)
print(f"  X: {X.shape}  y: {y.shape}")

# ── Reproducible 80/20 split ─────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
perm    = rng.permutation(len(X))
n_train = int(len(X) * TRAIN_FRAC)

X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]

print(f"  Train: {len(X_train)}  Test: {len(X_test)}  K={K}", flush=True)


# ── Vectorized batched KNN ────────────────────────────────────────────────────
def knn_predict(X_tr, y_tr, X_te, k, batch_size=BATCH_SIZE):
    """
    Brute-force KNN using squared Euclidean distance.

    Uses the identity ||a-b||² = ||a||² + ||b||² - 2·aᵀb so the inner loop
    is a dense matrix multiply (BLAS dgemm), keeping memory bounded to
    batch_size × n_train × 4 bytes per iteration.

    Labels may be any real values (handles {-1, 0, 1} via np.unique).
    """
    n_test = len(X_te)
    predictions = np.empty(n_test, dtype=y_tr.dtype)

    # Squared norms — computed once per call, O(n · d) each
    train_sq = np.einsum('ij,ij->i', X_tr, X_tr)   # (n_train,)
    test_sq  = np.einsum('ij,ij->i', X_te, X_te)   # (n_test,)

    for start in range(0, n_test, batch_size):
        end     = min(start + batch_size, n_test)
        X_batch = X_te[start:end]                   # (b, d)

        # (b, n_train) squared distances via BLAS
        dot   = X_batch @ X_tr.T                    # (b, n_train)
        dists = train_sq[np.newaxis, :] + test_sq[start:end, np.newaxis] - 2.0 * dot

        # k nearest neighbours (partial sort)
        k_idx    = np.argpartition(dists, k, axis=1)[:, :k]
        k_labels = y_tr[k_idx]                      # (b, k)

        # Majority vote — works for any label set including {-1, 0, 1}
        for i, row in enumerate(k_labels):
            unique, counts = np.unique(row, return_counts=True)
            predictions[start + i] = unique[np.argmax(counts)]

    return predictions


# ── Warm-up (prime BLAS / OS page cache) ─────────────────────────────────────
print("Warm-up run...", flush=True)
knn_predict(X_train, y_train, X_test, K)

# ── Timed runs ────────────────────────────────────────────────────────────────
print(f"\nRunning {N_RUNS} timed benchmark runs:", flush=True)
times_s = []
accs    = []

for i in range(N_RUNS):
    t0   = time.perf_counter()
    pred = knn_predict(X_train, y_train, X_test, K)
    t1   = time.perf_counter()

    elapsed = t1 - t0
    acc     = float(np.mean(pred == y_test) * 100)

    times_s.append(elapsed)
    accs.append(acc)
    print(f"  Run {i+1:2d}: {elapsed:.4f} s   acc={acc:.2f}%", flush=True)

print(f"\nMean : {np.mean(times_s):.4f} s")
print(f"Std  : {np.std(times_s):.4f} s")
print(f"Acc  : {np.mean(accs):.4f}%")

# ── Append to benchmark_results.csv ──────────────────────────────────────────
df = pd.read_csv('benchmark_results.csv')

if 'numpy_time' in df.columns:
    print("\nNote: numpy_time column already exists — overwriting.")
    df = df.drop(columns=['numpy_time', 'numpy_accuracy'], errors='ignore')

df['numpy_time']     = times_s
df['numpy_accuracy'] = accs

df.to_csv('benchmark_results.csv', index=False)
print("\nbenchmark_results.csv updated with numpy_time (seconds) and numpy_accuracy (%).")
