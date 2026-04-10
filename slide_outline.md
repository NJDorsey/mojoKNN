# Lab Update — mojoKNN: High-Performance KNN in Mojo
## Paper Outline & Slide Deck

---

## Slide 1: Title Slide

- **Title:** mojoKNN: Benchmarking K-Nearest Neighbors in Mojo vs Python/sklearn
- Nathan Dorsey
- Lab Update — April 2026
- Advisor / Course info as needed

---

## Slide 2: Introduction / Motivation

- KNN is one of the most fundamental ML algorithms — simple, interpretable, widely used as a baseline
- Python (sklearn) is the standard, but it has performance limitations for large-scale or latency-sensitive applications (e.g., HFT, real-time classification)
- Mojo is a new systems language designed for AI/ML — claims Python-level ergonomics with C/C++-level performance
- **Research question:** How much faster is a SIMD-optimized Mojo KNN implementation compared to sklearn across diverse datasets and data domains?

---

## Slide 3: Background — KNN Algorithm

- KNN classifies a query point by finding the K closest training samples (Euclidean distance) and majority-voting on their labels
- Two main search strategies:
  - **Brute-force:** compute distance to every training point — O(n * d) per query
  - **KD-tree:** spatial index that prunes large portions of the search space — O(d * log n) average case, but degrades in high dimensions
- Key performance levers: SIMD vectorization, parallelization, memory layout

---

## Slide 4: Background — Mojo Language

- Mojo: superset of Python syntax with systems-level control (manual memory, SIMD intrinsics, compile-time metaprogramming)
- Key features leveraged in this work:
  - **SIMD vectorization** for distance computation (processes multiple floats per instruction)
  - **Compile-time parameters** for matrix dimensions (zero-cost abstraction, no dynamic dispatch)
  - **Parallelization** via `parallelize` built-in (distributes test queries across CPU cores)
- Comparison target: sklearn's KNeighborsClassifier (Cython + BLAS backend)

---

## Slide 5: Methodology — Implementations

- **Mojo brute-force SIMD:** custom Matrix struct with compile-time dimensions, SIMD-vectorized Euclidean distance, parallel query evaluation
- **Mojo KD-tree SIMD:** custom KD-tree with SIMD distance in leaf nodes, max-heap for K nearest tracking, parallel queries
- **sklearn brute-force:** KNeighborsClassifier(algorithm="brute", metric="euclidean")
- **sklearn KD-tree:** KNeighborsClassifier(algorithm="kd_tree", leaf_size=30)
- All 4 methods use the same train/test split (random_state=42, 80/20) for fair comparison

---

## Slide 6: Methodology — Benchmarking Protocol

- **K = 5** (fixed, per standard benchmarking protocol)
- **10 runs per experiment**, first run excluded as warmup
- Metrics: wall-clock inference time (seconds), classification accuracy (%)
- Identical train/test data: Python splits data, writes CSV, both sklearn and Mojo load the same files
- Mojo code is auto-generated per experiment (compile-time constants for dataset dimensions)
- Machine: [fill in specs of machine where experiments are run]

---

## Slide 7: Datasets Overview

| Dataset       | Domain    | Rows      | Features | Classes | Notes                              |
|---------------|-----------|-----------|----------|---------|------------------------------------|
| Iris          | Classic   | 150       | 4        | 3       | Baseline, tiny                     |
| Wine          | Classic   | 178       | 13       | 3       | Small, moderate features           |
| Breast Cancer | Classic   | 569       | 30       | 2       | Medical, binary classification     |
| Digits        | Classic   | 1,797     | 64       | 10      | 8x8 pixel images                   |
| MNIST         | Image     | 10,000    | 784      | 10      | 28x28 handwritten digits (capped)  |
| CIFAR-10      | Image     | 10,000    | 3,072    | 10      | 32x32 color images (capped)        |
| AAPL          | Finance   | 203,586   | 16       | 2       | 1-min OHLCV, direction prediction  |
| WMT           | Finance   | ~200,000  | 16       | 2       | Walmart 1-min bars                 |
| CPRI          | Finance   | ~200,000  | 16       | 2       | Capri Holdings 1-min bars          |
| JPM           | Finance   | ~200,000  | 16       | 2       | JP Morgan 1-min bars               |

- Finance features: 16 engineered causal features from OHLCV (returns, volatility, RSI, moving averages, volume ratios, momentum)
- Target: binary direction of next-bar return (+1 / -1)

---

## Slide 8: Results — Overall Speedup Summary

- **Show:** Combined speedup bar chart or heatmap (all datasets x all methods)
- Key callouts:
  - Mojo brute-force SIMD: [X]x faster than sklearn brute on average
  - Mojo KD-tree SIMD: [X]x faster than sklearn brute on average
  - Speedup tends to increase with dataset size (more data = more SIMD/parallel benefit)
  - sklearn KD-tree vs sklearn brute: [note whether KD-tree helps or hurts in high dimensions]

---

## Slide 9: Results — Method Comparison Line Chart

- **Show:** The multi-dataset line chart (x = method, y = time, each line = a dataset)
- Each line connects the 4 methods for one dataset — steep drops from sklearn to Mojo = big wins
- Observation: larger datasets show steeper drops (more absolute time saved)
- High-dimensional datasets (CIFAR-10, MNIST) may show different patterns than low-dimensional (finance, iris)

---

## Slide 10: Results — Finance Datasets (AAPL, WMT, CPRI, JPM)

- **Show:** Group bar chart for finance datasets
- All 4 tickers have ~200k rows, 16 features — nearly identical problem structure
- Speedup should be consistent across tickers (same dimensionality)
- Accuracy: ~50% expected (random-walk nature of stock returns — this is expected, not a failure)
- Key point: even on real-world HFT-scale data, Mojo delivers [X]x speedup

---

## Slide 11: Results — Classic ML Datasets

- **Show:** Group bar chart for classic ML datasets (iris, wine, cancer, digits)
- Small datasets: Mojo advantage may be smaller (overhead dominates) or brute-force is already fast
- As features/rows increase (digits: 64 features, 1797 rows), Mojo advantage grows
- Accuracy: high across all implementations (verifying correctness)

---

## Slide 12: Results — Image Datasets (MNIST, CIFAR-10)

- **Show:** Per-dataset bar charts for MNIST and CIFAR-10
- High dimensionality (784 and 3,072 features): tests whether SIMD advantage scales with feature count
- KD-tree performance: expected to degrade in very high dimensions (curse of dimensionality)
- SIMD brute-force may outperform KD-tree in these cases

---

## Slide 13: Results — Accuracy Verification

- **Show:** Accuracy comparison chart
- sklearn and Mojo implementations produce nearly identical accuracy
- Small differences (< 1%) due to floating-point ordering / tie-breaking
- Confirms: Mojo implementation is correct, not just fast

---

## Slide 14: Discussion / Analysis

- **Why is Mojo faster?**
  - SIMD: processes 8-16 floats per instruction vs scalar loops
  - Parallelization: test queries distributed across all CPU cores
  - Memory layout: compile-time matrix dimensions enable contiguous, cache-friendly access
  - No Python GIL, no interpreter overhead
- **When does Mojo help most?**
  - Large datasets (more work to parallelize and vectorize)
  - Low-to-moderate dimensionality (SIMD utilization is high)
- **When does the advantage shrink?**
  - Tiny datasets (overhead dominates)
  - Very high dimensions (memory bandwidth becomes bottleneck)

---

## Slide 15: Discussion — KD-tree vs Brute Force

- KD-tree expected to excel in low dimensions (finance: 16 features) — logarithmic search vs linear
- In high dimensions (MNIST 784, CIFAR 3072), KD-tree degrades toward brute-force performance
- sklearn KD-tree uses different splitting heuristics than our Mojo implementation — direct comparison is informative
- Mojo KD-tree may still win via SIMD distance computation in leaf nodes

---

## Slide 16: Current Status & Next Steps

- **Completed:**
  - Mojo KNN implementation (brute-force SIMD + KD-tree SIMD)
  - Benchmarking framework (automated experiment runner + analysis pipeline)
  - Experiments across 10 datasets spanning 3 domains (classic ML, image, finance)
  - Comprehensive visualization and analysis pipeline
- **In progress:**
  - Paper draft being established based on these results
- **Next steps:**
  - Finalize paper sections (related work, detailed methodology, full results tables)
  - Potentially explore additional optimizations (variance-based pruning, ball trees)
  - Consider GPU comparison if time permits

---

## Slide 17: Summary / Takeaway

- Mojo delivers significant speedups over sklearn for KNN classification across diverse datasets
- Speedup scales with dataset size — most impactful for large, real-world datasets
- Implementation correctness verified via accuracy parity with sklearn
- Mojo is a viable alternative for performance-critical ML workloads while maintaining Python-like readability
