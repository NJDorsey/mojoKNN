# Mojo KNN Optimization & Benchmarking Plan

## Background

This project implements k-nearest neighbors (KNN) in Mojo with KD-tree and brute force variants, extending Kolli, Wu, and Han's work which benchmarked Mojo brute-force KNN against sklearn brute-force. The current implementation has several performance bottlenecks that prevent it from beating sklearn's KD-tree implementation. This plan addresses those bottlenecks and establishes proper benchmarking against sklearn.

### Current State
- **Dataset:** AAPL_LONG (203,586 rows, 22 features, 3 classes), 80/20 split, K=100
- **Training:** 162,868 samples, **Testing:** 40,718 samples
- **Current benchmark results (30 runs):**
  - KD-Tree SIMD: ~13.2s mean
  - KD-Tree Scalar: ~15.8s mean
  - Brute Force SIMD: ~24.9s mean
  - Brute Force Scalar: ~27.8s mean
  - Pure NumPy brute force: ~57.3s mean
- **Problem:** Mojo beats pure NumPy but needs to be competitive with sklearn's optimized C++ KD-tree implementation
- **All Mojo variants achieve ~48.158% accuracy; NumPy achieves ~48.308% (different random split)**

### Key Files
- `KNN.mojo` (1088 lines) — Main implementation: `Matrix`, `Vector`, `SQRT`, `Sorting`, distance functions, `runKNN` variants, `classification_report`, `main()` benchmark loop
- `kdtree.mojo` (448 lines) — `Point`, `Neighbor`, `MaxHeap`, `KDNode`, `KDTree[use_simd: Bool]`, distance functions
- `csv.mojo` (191 lines) — CSV data loading utilities
- `split.mojo` (248 lines) — Train/test split utilities
- `numpy_knn_benchmark.py` (126 lines) — Pure NumPy baseline benchmark
- `benchmark_results.csv` — Current 30-run results
- `analysis_mojo_vs_numpy.py`, `analysis_mojo.py`, `analysis_best_vs_numpy.py`, `benchmark_analysis.py` — Statistical analysis scripts (Friedman, Wilcoxon, effect sizes, box plots)

---

## Phase 1: Remove Matrix/Vector Destructors

**Files to modify:** `KNN.mojo`

**Problem:** The `Matrix` and `Vector` structs have `__del__` methods (KNN.mojo lines 45-46 and 121-122) that call `self.data.free()`. Inside the `parallelize` closures, each of the ~40,718 test point iterations creates local `Matrix[162868, 1]`, `Matrix[cols, 1]`, and `Vector[162868, 1]` instances. When these go out of scope, the destructor fires — that's ~120K `free()` calls per benchmark run, creating significant allocator contention across threads.

**Fix:** Delete the `__del__` method from both `Matrix` and `Vector`:

```mojo
# DELETE these from Matrix (lines 45-46):
fn __del__(owned self):
    self.data.free()

# DELETE these from Vector (lines 121-122):
fn __del__(owned self):
    self.data.free()
```

**Rationale:** The benchmark process is short-lived. The OS reclaims all memory on exit. The memory "leak" is acceptable for benchmarking. No other files are affected — `csv.mojo` and `split.mojo` have their own data structures.

**Verification:** Run a single benchmark iteration and confirm brute force times decrease. Accuracy must remain unchanged (48.158%).

---

## Phase 2: KD-Tree Refactoring (Contiguous Memory + Leaf Size)

**Files to modify:** `kdtree.mojo`, `KNN.mojo`

This is the core change. The leaf_size and contiguous-memory changes are tightly coupled because both fundamentally alter how `KDNode` stores and accesses point data. Implement them together.

### 2a: Replace List-based Point with Contiguous Coordinate Buffer

**Problem:** Each `Point` has `coords: List[KDFloat]` — a heap-allocated dynamic array with bounds checking. With 162,868 training points, that's 162,868 separate heap allocations for the `List` backing stores. Every distance computation goes through `List.__getitem__` (bounds check + pointer indirection). Additionally, `row_to_point()` in KNN.mojo (lines 428-433) allocates a new `List[KDFloat]` for every query point — 40,718 heap allocations per run.

**New design:**
1. Store all point coordinates in a single `UnsafePointer[KDFloat]` buffer inside `KDTree` (row-major layout: point `i`, feature `j` at `buffer[i * n_dims + j]`)
2. Replace the `Point` struct usage with a lightweight `PointRef`:
   ```mojo
   @value
   struct PointRef:
       var buffer_index: Int      # Row index into the coordinate buffer
       var original_index: Int    # Original index in the dataset (for class label lookup)
   ```
3. `KDNode` stores `PointRef` instead of `Point`, plus `split_value: KDFloat`
4. Distance functions take `UnsafePointer[KDFloat]` for the query point — direct pointer arithmetic, no allocation

**KDTree struct changes:**
```mojo
struct KDTree[use_simd: Bool = False]:
    var root: UnsafePointer[KDNode]
    var num_dimensions: Int
    var size: Int
    var coord_buffer: UnsafePointer[KDFloat]   # NEW: all points contiguous
    var original_indices: UnsafePointer[Int]     # NEW: maps buffer row -> dataset row
    var leaf_size: Int                           # NEW: see Phase 2b
```

**Constructor:** Keep `List[Point]` as input (backward compat with `matrix_to_points()`). Convert to contiguous buffer on construction:
```mojo
fn __init__(inout self, owned points: List[Point], leaf_size: Int = 30):
    # ... allocate coord_buffer, original_indices ...
    # Copy from points into contiguous buffer
    for i in range(self.size):
        self.original_indices[i] = points[i].original_index
        for j in range(self.num_dimensions):
            self.coord_buffer[i * self.num_dimensions + j] = points[i].coords[j]
    # Build tree
```

**Distance function replacement:** Replace `euclidean_distance_squared(p1: Point, p2: Point)` with:
```mojo
@always_inline
fn distance_squared_to_buffer_point(
    self, query_ptr: UnsafePointer[KDFloat], buffer_index: Int
) -> KDFloat:
    var point_ptr = self.coord_buffer + buffer_index * self.num_dimensions
    var sum_sq: KDFloat = 0.0
    @parameter
    if use_simd:
        var k = 0
        while k + KD_SIMD_W <= self.num_dimensions:
            var diff = query_ptr.load[width=KD_SIMD_W](k) - point_ptr.load[width=KD_SIMD_W](k)
            sum_sq += (diff * diff).reduce_add()
            k += KD_SIMD_W
        while k < self.num_dimensions:
            var diff = query_ptr[k] - point_ptr[k]
            sum_sq += diff * diff
            k += 1
    else:
        for i in range(self.num_dimensions):
            var diff = query_ptr[i] - point_ptr[i]
            sum_sq += diff * diff
    return sum_sq
```

**Helper methods:**
```mojo
@always_inline
fn get_coord(self, buffer_index: Int, dim: Int) -> KDFloat:
    return self.coord_buffer[buffer_index * self.num_dimensions + dim]

@always_inline
fn get_point_ptr(self, buffer_index: Int) -> UnsafePointer[KDFloat]:
    return self.coord_buffer + buffer_index * self.num_dimensions
```

**Update `_sort_by_axis`:** Change `self.points[key_idx][axis]` to `self.get_coord(key_idx, axis)`.

**Note on sorting:** The insertion sort in `_sort_by_axis` is O(n^2) per level, which is slow for the root level (162K points). This could be improved to merge sort / intro sort in the future, but it only runs once during tree construction (not during queries), so it's lower priority than query-path optimizations.

### 2b: Implement Leaf Size Parameter

**Problem:** The tree recurses to single-point leaves. With 162,868 points, there are ~162,868 leaf nodes. Each leaf traversal is a pointer dereference + function call + branch misprediction. sklearn defaults to `leaf_size=30`, creating ~5,400 internal nodes instead.

**KDNode changes:**
```mojo
@value
struct KDNode:
    var axis: Int                               # Split axis (-1 for leaf nodes)
    var split_value: KDFloat                    # Split value for internal nodes
    var point_ref: PointRef                     # The point at this internal node
    var left: UnsafePointer[KDNode]
    var right: UnsafePointer[KDNode]
    # Leaf fields:
    var leaf_indices: UnsafePointer[PointRef]   # Flat array of PointRefs in leaf
    var leaf_count: Int                         # Number of points (0 for internal)
```

**Note:** Remove `@value` from `KDNode` to avoid auto-generated copy constructors that would share `UnsafePointer` — implement explicit `__init__` only.

**`_build_recursive` changes:**
```mojo
fn _build_recursive(inout self, owned indices: List[Int], depth: Int) -> UnsafePointer[KDNode]:
    if len(indices) == 0:
        return UnsafePointer[KDNode]()

    # LEAF NODE
    if len(indices) <= self.leaf_size:
        var leaf_refs = UnsafePointer[PointRef].alloc(len(indices))
        for i in range(len(indices)):
            var buf_idx = indices[i]
            leaf_refs[i] = PointRef(buf_idx, self.original_indices[buf_idx])
        # Create leaf node with axis=-1
        var node_ptr = UnsafePointer[KDNode].alloc(1)
        node_ptr.init_pointee_move(KDNode(
            axis=-1, split_value=0.0, point_ref=PointRef(0, 0),
            left=UnsafePointer[KDNode](), right=UnsafePointer[KDNode](),
            leaf_indices=leaf_refs, leaf_count=len(indices)
        ))
        return node_ptr

    # INTERNAL NODE (same logic as before, using buffer indices)
    var axis = depth % self.num_dimensions
    self._sort_by_axis(indices, axis)
    var median_pos = len(indices) // 2
    var median_buf_idx = indices[median_pos]
    # ... create internal node, recurse left/right ...
```

**`_knn_search` changes:**
```mojo
fn _knn_search(self, node_ptr: UnsafePointer[KDNode], query_ptr: UnsafePointer[KDFloat], inout heap: MaxHeap):
    if not node_ptr:
        return
    var node = node_ptr[]

    # LEAF: brute-force scan
    if node.leaf_count > 0:
        for i in range(node.leaf_count):
            var ref = node.leaf_indices[i]
            var dist_sq = self.distance_squared_to_buffer_point(query_ptr, ref.buffer_index)
            heap.push(Neighbor(ref.original_index, dist_sq))
        return

    # INTERNAL: standard KD-tree logic with split_value
    var dist_sq = self.distance_squared_to_buffer_point(query_ptr, node.point_ref.buffer_index)
    heap.push(Neighbor(node.point_ref.original_index, dist_sq))

    var diff = query_ptr[node.axis] - node.split_value
    var diff_sq = diff * diff
    # ... search closer subtree first, prune other ...
```

**Public API change:**
```mojo
fn k_nearest_neighbors(self, query_ptr: UnsafePointer[KDFloat], k: Int) -> List[Neighbor]:
    var heap = MaxHeap(k)
    self._knn_search(self.root, query_ptr, heap)
    var result = heap.to_sorted_list()
    for i in range(len(result)):
        result[i].distance = sqrt(result[i].distance)
    return result
```

### 2c: Update KNN.mojo Call Sites

1. **Remove `row_to_point` function** (lines 428-433). Replace usage in predict closures:
   ```mojo
   # OLD (40K heap allocations per run):
   var query = row_to_point(testing, i)
   var neighbors = tree_scalar.k_nearest_neighbors(query, K)

   # NEW (zero allocations):
   var query_ptr = testing.data + i * testing.cols
   var neighbors = tree_scalar.k_nearest_neighbors(query_ptr, K)
   ```
   The `testing.data` pointer is a contiguous row-major buffer, so `testing.data + i * testing.cols` points to row `i`.

2. **Pass leaf_size** when constructing trees (lines 930-931):
   ```mojo
   var tree_scalar = KDTree[False](train_points_scalar^, leaf_size=30)
   var tree_simd   = KDTree[True](train_points_simd^, leaf_size=30)
   ```

3. **Update imports** (line 21):
   ```mojo
   from kdtree import KDTree, PointRef, Neighbor, KDFloat
   ```
   Keep `Point` imported if `matrix_to_points` still uses it.

4. **`matrix_to_points`** (lines 417-425): Keep as-is. It builds `List[Point]` which the KDTree constructor converts into the contiguous buffer. This runs once before benchmarking, not on the hot path.

5. **`predict_class_from_neighbors`** (lines 403-414): No change needed — receives `List[Neighbor]` which is unchanged.

**Safety note:** The `testing` Matrix must remain alive while query pointers are in use. It is declared in `main()` scope, so this is safe.

---

## Phase 3: sklearn Benchmark Script

**New file:** `sklearn_knn_benchmark.py`

Benchmark sklearn's `KNeighborsClassifier` with both `algorithm='kd_tree'` and `algorithm='brute'`.

**Parameters:**
- `n_neighbors=100` (K=100)
- `algorithm='kd_tree'` with `leaf_size=30` (default) — apples-to-apples with our implementation
- `algorithm='brute'` — baseline comparison
- `metric='euclidean'`
- `n_jobs=-1` (use all cores, matching Mojo's `parallelize`)
- Same AAPL_LONG dataset, same 80/20 split
- 30 runs, 1 warm-up run
- Use `np.random.default_rng(42)` split to match `numpy_knn_benchmark.py`

**Output:** Append columns to `benchmark_results.csv`:
- `sklearn_kdtree_time`, `sklearn_kdtree_accuracy`
- `sklearn_brute_time`, `sklearn_brute_accuracy`

**Dependency:** Add `scikit-learn` to `pixi.toml`.

**Note on split consistency:** The Mojo benchmark uses an unseeded random split (`split.mojo`) while NumPy/sklearn use seed 42. Accuracies will differ slightly (~48.16% vs ~48.31%). Timing comparisons are still valid since both operate on the same data dimensions.

---

## Phase 4: Update Analysis Scripts

**Files:** Extend `analysis_mojo_vs_numpy.py` or create new `analysis_full.py`

After Phase 3 populates `benchmark_results.csv` with sklearn columns:
- Add sklearn KD-tree and sklearn brute force to the comparison
- Update statistical tests for additional variants (Friedman with 7 groups, Wilcoxon pairwise with Bonferroni correction for C(7,2)=21 pairs)
- Key headline comparison: **Mojo KD-tree SIMD vs sklearn KD-tree speedup**
- Updated box plots with all 7 variants

---

## Phase 5: Verify Brute Force vs Kolli's Implementation

**Investigative — no code changes unless discrepancies found.**

Check whether the current Mojo brute force matches Kolli, Wu, and Han's approach. Key things to verify:

1. **Top-K selection method:** The current implementation **quicksorts ALL 162,868 distances** to find the top K=100 (`Sorting.simd_sort_quick` at KNN.mojo lines 512, 537, 1007). This is O(n log n). Alternatives:
   - **Max-heap** (already implemented in `kdtree.mojo`'s `MaxHeap`): O(n log k) — scan distances once, maintain heap of k best
   - **Partial sort** (like NumPy's `argpartition`): O(n) average case
   - **Check if Kolli's implementation also quicksorts all distances.** If it does, the current implementation is faithful but suboptimal. If Kolli uses a different method, document what differs and consider aligning.

2. **Distance computation:** The scalar brute force `distMatvec` (line 324) vectorizes over the column dimension of a column vector (which is always 1 — questionable utility). The SIMD variant `euclidean_dist_simd` (line 342) is properly vectorized over features. Check which approach Kolli used.

3. **If Kolli's implementation differs:** User will replicate more faithfully. Add as a follow-up task.

---

## Execution Order

```
Phase 1 (Remove destructors)           <- Do first, trivial
    |
Phase 2 (KD-tree refactoring)          <- Core work, most complex
    |   2a: Contiguous buffer
    |   2b: Leaf size parameter
    |   2c: Update call sites
    |
Phase 3 (sklearn benchmark script)     <- Independent of Phase 2, can start after Phase 1
    |
Phase 4 (Update analysis scripts)      <- Depends on Phase 3 output
    |
Phase 5 (Verify brute force vs Kolli)  <- Independent, investigative
```

## Verification Checklist
- [ ] After Phase 1: single benchmark run shows time decrease, accuracy unchanged
- [ ] After Phase 2: accuracy matches pre-refactor (~48.158%), times improve
- [ ] After Phase 2: tree depth is shallower (log2(162868/30) ≈ 12 vs log2(162868) ≈ 17)
- [ ] After Phase 3: sklearn accuracies are consistent across 30 runs
- [ ] Final: full 30-run benchmark with all variants, statistical analysis confirms results
