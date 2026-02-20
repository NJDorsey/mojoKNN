"""K-D Tree implementation for fast k-nearest neighbor search.

This module provides an efficient k-d tree data structure optimized for
k-nearest neighbor queries in high-dimensional spaces.
"""

from collections import List
from math import sqrt
from memory import UnsafePointer
from sys import simdwidthof


alias KDFloat = Float32
alias KD_SIMD_W = simdwidthof[KDFloat]()


@value
struct Point:
    """A point in d-dimensional space."""
    var coords: List[KDFloat]
    var original_index: Int  # Track original position in dataset

    fn __init__(inout self, coords: List[KDFloat], original_index: Int = -1):
        """Initialize a point from a list of coordinates."""
        self.coords = coords
        self.original_index = original_index

    fn __getitem__(self, idx: Int) -> KDFloat:
        """Get coordinate at index."""
        return self.coords[idx]

    fn __len__(self) -> Int:
        """Get dimensionality."""
        return len(self.coords)


@value
struct Neighbor:
    """A neighbor with its index and distance."""
    var index: Int
    var distance: KDFloat

    fn __lt__(self, other: Neighbor) -> Bool:
        """Less than comparison for sorting."""
        return self.distance < other.distance

    fn __gt__(self, other: Neighbor) -> Bool:
        """Greater than comparison."""
        return self.distance > other.distance


struct MaxHeap:
    """A max-heap for maintaining k nearest neighbors.

    We use a max-heap so we can efficiently check and remove the
    worst (farthest) neighbor when we find a better one.
    """
    var data: List[Neighbor]
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        """Initialize heap with given capacity (k for k-NN)."""
        self.data = List[Neighbor]()
        self.capacity = capacity

    fn __len__(self) -> Int:
        """Get current size."""
        return len(self.data)

    fn is_full(self) -> Bool:
        """Check if heap is at capacity."""
        return len(self.data) >= self.capacity

    fn max_distance(self) -> KDFloat:
        """Get the maximum distance in the heap (root of max-heap)."""
        if len(self.data) == 0:
            return KDFloat.MAX
        return self.data[0].distance

    fn push(inout self, neighbor: Neighbor):
        """Add a neighbor to the heap."""
        if not self.is_full():
            # Just add and bubble up
            self.data.append(neighbor)
            self._bubble_up(len(self.data) - 1)
        elif neighbor.distance < self.data[0].distance:
            # Replace the max (worst) with this better neighbor
            self.data[0] = neighbor
            self._bubble_down(0)

    fn _bubble_up(inout self, idx: Int):
        """Restore heap property upward."""
        var current = idx
        while current > 0:
            var parent = (current - 1) // 2
            if self.data[current].distance > self.data[parent].distance:
                # Swap using temporary copies to avoid aliasing
                var temp_current_idx = self.data[current].index
                var temp_current_dist = self.data[current].distance
                var temp_parent_idx = self.data[parent].index
                var temp_parent_dist = self.data[parent].distance
                self.data[current] = Neighbor(temp_parent_idx, temp_parent_dist)
                self.data[parent] = Neighbor(temp_current_idx, temp_current_dist)
                current = parent
            else:
                break

    fn _bubble_down(inout self, idx: Int):
        """Restore heap property downward."""
        var current = idx
        var size = len(self.data)

        while True:
            var largest = current
            var left = 2 * current + 1
            var right = 2 * current + 2

            if left < size and self.data[left].distance > self.data[largest].distance:
                largest = left
            if right < size and self.data[right].distance > self.data[largest].distance:
                largest = right

            if largest != current:
                # Swap using temporary copies to avoid aliasing
                var temp_current_idx = self.data[current].index
                var temp_current_dist = self.data[current].distance
                var temp_largest_idx = self.data[largest].index
                var temp_largest_dist = self.data[largest].distance
                self.data[current] = Neighbor(temp_largest_idx, temp_largest_dist)
                self.data[largest] = Neighbor(temp_current_idx, temp_current_dist)
                current = largest
            else:
                break

    fn to_sorted_list(inout self) -> List[Neighbor]:
        """Extract all neighbors sorted by distance (ascending)."""
        var result = List[Neighbor]()

        # Copy data
        for i in range(len(self.data)):
            result.append(Neighbor(self.data[i].index, self.data[i].distance))

        # Simple insertion sort (k is typically small)
        for i in range(1, len(result)):
            var key_idx = result[i].index
            var key_dist = result[i].distance
            var j = i - 1
            while j >= 0 and result[j].distance > key_dist:
                # Copy to avoid aliasing
                var prev_idx = result[j].index
                var prev_dist = result[j].distance
                result[j + 1] = Neighbor(prev_idx, prev_dist)
                j -= 1
            result[j + 1] = Neighbor(key_idx, key_dist)

        return result


@always_inline
fn euclidean_distance_squared(p1: Point, p2: Point) -> KDFloat:
    """Calculate squared Euclidean distance between two points.

    We use squared distance to avoid sqrt operations during search.
    Only compute actual distance when needed for final results.
    """
    var sum_sq: KDFloat = 0.0
    var dims = len(p1)

    for i in range(dims):
        var diff = p1[i] - p2[i]
        sum_sq += diff * diff

    return sum_sq


@always_inline
fn euclidean_distance_squared_simd(p1: Point, p2: Point) -> KDFloat:
    """SIMD-vectorized squared Euclidean distance between two points.

    Uses List.unsafe_ptr() to get a contiguous pointer to coords data,
    then processes KD_SIMD_W floats at a time with SIMD load + reduce_add.
    Falls back to scalar for the tail (dims % KD_SIMD_W remainder).
    """
    var dims = len(p1)
    var p1_ptr = p1.coords.unsafe_ptr()
    var p2_ptr = p2.coords.unsafe_ptr()
    var sum_sq: KDFloat = 0.0
    var k = 0
    while k + KD_SIMD_W <= dims:
        var diff = p1_ptr.load[width=KD_SIMD_W](k) - p2_ptr.load[width=KD_SIMD_W](k)
        sum_sq += (diff * diff).reduce_add()
        k += KD_SIMD_W
    while k < dims:
        var diff = p1_ptr[k] - p2_ptr[k]
        sum_sq += diff * diff
        k += 1
    return sum_sq


@always_inline
fn euclidean_distance(p1: Point, p2: Point) -> KDFloat:
    """Calculate Euclidean distance between two points."""
    return sqrt(euclidean_distance_squared(p1, p2))


@value
struct KDNode:
    """A node in the k-d tree."""
    var point: Point
    var axis: Int  # Which dimension this node splits on
    var left: UnsafePointer[KDNode]
    var right: UnsafePointer[KDNode]

    fn __init__(inout self, owned point: Point, axis: Int):
        """Create a new k-d tree node."""
        self.point = point
        self.axis = axis
        self.left = UnsafePointer[KDNode]()
        self.right = UnsafePointer[KDNode]()

    fn has_left(self) -> Bool:
        """Check if left child exists."""
        return self.left.__bool__()

    fn has_right(self) -> Bool:
        """Check if right child exists."""
        return self.right.__bool__()


struct KDTree[use_simd: Bool = False]:
    """A k-d tree for efficient nearest neighbor search."""
    var root: UnsafePointer[KDNode]
    var num_dimensions: Int
    var size: Int
    var points: List[Point]  # Store reference to points for search

    fn __init__(inout self, owned points: List[Point]):
        """Build a k-d tree from a list of points."""
        self.size = len(points)
        self.root = UnsafePointer[KDNode]()
        self.points = points^

        if self.size == 0:
            self.num_dimensions = 0
            return

        self.num_dimensions = len(self.points[0])

        # Create list of indices to sort
        var indices = List[Int]()
        for i in range(len(self.points)):
            indices.append(i)

        self.root = self._build_recursive(indices, 0)

    fn _build_recursive(
        inout self,
        owned indices: List[Int],
        depth: Int,
    ) -> UnsafePointer[KDNode]:
        """Recursively build the k-d tree."""
        if len(indices) == 0:
            return UnsafePointer[KDNode]()

        var axis = depth % self.num_dimensions

        # Sort indices by the current axis coordinate
        self._sort_by_axis(indices, axis)

        # Find median
        var median_pos = len(indices) // 2
        var median_idx = indices[median_pos]

        # Create node with a copy of the point
        var node_ptr = UnsafePointer[KDNode].alloc(1)
        var point_copy = Point(self.points[median_idx].coords, self.points[median_idx].original_index)
        node_ptr.init_pointee_move(KDNode(point_copy^, axis))

        # Split indices for left and right subtrees
        var left_indices = List[Int]()
        var right_indices = List[Int]()

        for i in range(median_pos):
            left_indices.append(indices[i])

        for i in range(median_pos + 1, len(indices)):
            right_indices.append(indices[i])

        # Recursively build subtrees
        node_ptr[].left = self._build_recursive(left_indices^, depth + 1)
        node_ptr[].right = self._build_recursive(right_indices^, depth + 1)

        return node_ptr

    fn _sort_by_axis(self, inout indices: List[Int], axis: Int):
        """Sort indices by coordinate value on given axis (insertion sort)."""
        for i in range(1, len(indices)):
            var key_idx = indices[i]
            var key_val = self.points[key_idx][axis]
            var j = i - 1

            while j >= 0 and self.points[indices[j]][axis] > key_val:
                indices[j + 1] = indices[j]
                j -= 1

            indices[j + 1] = key_idx

    fn nearest_neighbor(self, query: Point) -> Neighbor:
        """Find the nearest neighbor to the query point."""
        var best = Neighbor(-1, KDFloat.MAX)
        self._nn_search(self.root, query, best)
        # Convert squared distance to actual distance
        best.distance = sqrt(best.distance)
        return best

    fn _nn_search(
        self,
        node_ptr: UnsafePointer[KDNode],
        query: Point,
        inout best: Neighbor,
    ):
        """Recursive nearest neighbor search with pruning."""
        if not node_ptr:
            return

        var node = node_ptr[]
        var dist_sq: KDFloat = 0.0
        @parameter
        if use_simd:
            dist_sq = euclidean_distance_squared_simd(query, node.point)
        else:
            dist_sq = euclidean_distance_squared(query, node.point)

        # Update best if this node is closer
        if dist_sq < best.distance:
            best = Neighbor(node.point.original_index, dist_sq)

        # Determine which subtree to search first
        var axis = node.axis
        var diff = query[axis] - node.point[axis]
        var diff_sq = diff * diff

        var first: UnsafePointer[KDNode]
        var second: UnsafePointer[KDNode]

        if diff < 0:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left

        # Search the closer subtree first
        self._nn_search(first, query, best)

        # Only search the other subtree if the splitting plane is closer
        # than the current best distance (pruning!)
        if diff_sq < best.distance:
            self._nn_search(second, query, best)

    fn k_nearest_neighbors(self, query: Point, k: Int) -> List[Neighbor]:
        """Find the k nearest neighbors to the query point.

        Uses a max-heap to efficiently maintain the k best candidates.
        Returns neighbors sorted by distance (ascending).
        """
        var heap = MaxHeap(k)
        self._knn_search(self.root, query, heap)

        # Convert squared distances to actual distances
        var result = heap.to_sorted_list()
        for i in range(len(result)):
            result[i].distance = sqrt(result[i].distance)

        return result

    fn _knn_search(
        self,
        node_ptr: UnsafePointer[KDNode],
        query: Point,
        inout heap: MaxHeap,
    ):
        """Recursive k-nearest neighbor search with pruning."""
        if not node_ptr:
            return

        var node = node_ptr[]
        var dist_sq: KDFloat = 0.0
        @parameter
        if use_simd:
            dist_sq = euclidean_distance_squared_simd(query, node.point)
        else:
            dist_sq = euclidean_distance_squared(query, node.point)

        # Add to heap (heap handles capacity logic)
        heap.push(Neighbor(node.point.original_index, dist_sq))

        # Determine which subtree to search first
        var axis = node.axis
        var diff = query[axis] - node.point[axis]
        var diff_sq = diff * diff

        var first: UnsafePointer[KDNode]
        var second: UnsafePointer[KDNode]

        if diff < 0:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left

        # Search the closer subtree first
        self._knn_search(first, query, heap)

        # Only search the other subtree if:
        # 1. We don't have k neighbors yet, OR
        # 2. The splitting plane is closer than our worst (farthest) neighbor
        if not heap.is_full() or diff_sq < heap.max_distance():
            self._knn_search(second, query, heap)

    fn get_depth(self) -> Int:
        """Get the maximum depth of the tree."""
        return self._get_depth_recursive(self.root)

    fn _get_depth_recursive(self, node_ptr: UnsafePointer[KDNode]) -> Int:
        """Recursively compute tree depth."""
        if not node_ptr:
            return 0
        var left_depth = self._get_depth_recursive(node_ptr[].left)
        var right_depth = self._get_depth_recursive(node_ptr[].right)
        if left_depth > right_depth:
            return left_depth + 1
        return right_depth + 1

    fn count_nodes(self) -> Int:
        """Count total nodes in the tree."""
        return self._count_recursive(self.root)

    fn _count_recursive(self, node_ptr: UnsafePointer[KDNode]) -> Int:
        """Recursively count nodes."""
        if not node_ptr:
            return 0
        return (
            1
            + self._count_recursive(node_ptr[].left)
            + self._count_recursive(node_ptr[].right)
        )
