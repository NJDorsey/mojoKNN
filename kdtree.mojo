"""K-D Tree implementation for fast k-nearest neighbor search.

This module provides an efficient k-d tree data structure optimized for
k-nearest neighbor queries in high-dimensional spaces. Uses a contiguous
coordinate buffer for cache-friendly access and SIMD-vectorized distance
computation. Supports a leaf_size parameter to reduce tree depth.
"""

from collections import List
from math import sqrt
from memory import UnsafePointer
from sys import simdwidthof


alias KDFloat = Float32
alias KD_SIMD_W = simdwidthof[KDFloat]()


@value
struct Point:
    """A point in d-dimensional space. Used as input to KDTree construction."""
    var coords: List[KDFloat]
    var original_index: Int

    fn __init__(inout self, coords: List[KDFloat], original_index: Int = -1):
        self.coords = coords
        self.original_index = original_index

    fn __getitem__(self, idx: Int) -> KDFloat:
        return self.coords[idx]

    fn __len__(self) -> Int:
        return len(self.coords)


@value
struct PointRef:
    """Lightweight reference to a point in the contiguous coordinate buffer."""
    var buffer_index: Int
    var original_index: Int


@value
struct Neighbor:
    """A neighbor with its index and distance."""
    var index: Int
    var distance: KDFloat

    fn __lt__(self, other: Neighbor) -> Bool:
        return self.distance < other.distance

    fn __gt__(self, other: Neighbor) -> Bool:
        return self.distance > other.distance


struct MaxHeap:
    """A max-heap for maintaining k nearest neighbors.

    We use a max-heap so we can efficiently check and remove the
    worst (farthest) neighbor when we find a better one.
    """
    var data: List[Neighbor]
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        self.data = List[Neighbor]()
        self.capacity = capacity

    fn __len__(self) -> Int:
        return len(self.data)

    fn is_full(self) -> Bool:
        return len(self.data) >= self.capacity

    fn max_distance(self) -> KDFloat:
        if len(self.data) == 0:
            return KDFloat.MAX
        return self.data[0].distance

    fn push(inout self, neighbor: Neighbor):
        if not self.is_full():
            self.data.append(neighbor)
            self._bubble_up(len(self.data) - 1)
        elif neighbor.distance < self.data[0].distance:
            self.data[0] = neighbor
            self._bubble_down(0)

    fn _bubble_up(inout self, idx: Int):
        var current = idx
        while current > 0:
            var parent = (current - 1) // 2
            if self.data[current].distance > self.data[parent].distance:
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

        for i in range(len(self.data)):
            result.append(Neighbor(self.data[i].index, self.data[i].distance))

        # Insertion sort (k is typically small)
        for i in range(1, len(result)):
            var key_idx = result[i].index
            var key_dist = result[i].distance
            var j = i - 1
            while j >= 0 and result[j].distance > key_dist:
                var prev_idx = result[j].index
                var prev_dist = result[j].distance
                result[j + 1] = Neighbor(prev_idx, prev_dist)
                j -= 1
            result[j + 1] = Neighbor(key_idx, key_dist)

        return result


struct KDNode:
    """A node in the k-d tree. Internal nodes split on an axis; leaf nodes
    store a flat array of PointRefs for brute-force scanning."""
    var axis: Int                               # Split axis (-1 for leaf)
    var split_value: KDFloat                    # Split value for internal nodes
    var point_ref: PointRef                     # The median point at this internal node
    var left: UnsafePointer[KDNode]
    var right: UnsafePointer[KDNode]
    var leaf_indices: UnsafePointer[PointRef]   # Flat array of PointRefs in leaf
    var leaf_count: Int                         # Number of points (0 for internal)

    fn __init__(inout self, axis: Int, split_value: KDFloat, point_ref: PointRef,
                left: UnsafePointer[KDNode], right: UnsafePointer[KDNode],
                leaf_indices: UnsafePointer[PointRef], leaf_count: Int):
        self.axis = axis
        self.split_value = split_value
        self.point_ref = point_ref
        self.left = left
        self.right = right
        self.leaf_indices = leaf_indices
        self.leaf_count = leaf_count

    fn __copyinit__(inout self, existing: Self):
        self.axis = existing.axis
        self.split_value = existing.split_value
        self.point_ref = existing.point_ref
        self.left = existing.left
        self.right = existing.right
        self.leaf_indices = existing.leaf_indices
        self.leaf_count = existing.leaf_count

    fn __moveinit__(inout self, owned existing: Self):
        self.axis = existing.axis
        self.split_value = existing.split_value
        self.point_ref = existing.point_ref
        self.left = existing.left
        self.right = existing.right
        self.leaf_indices = existing.leaf_indices
        self.leaf_count = existing.leaf_count


struct KDTree:
    """A k-d tree for efficient nearest neighbor search.

    Uses a contiguous coordinate buffer for cache-friendly memory access
    and SIMD-vectorized distance computation. Supports a leaf_size parameter
    (default 30, matching sklearn) to reduce tree depth and improve query
    performance by brute-force scanning at leaf nodes.
    """
    var root: UnsafePointer[KDNode]
    var num_dimensions: Int
    var size: Int
    var coord_buffer: UnsafePointer[KDFloat]
    var original_indices: UnsafePointer[Int]
    var leaf_size: Int

    fn __init__(inout self, owned points: List[Point], leaf_size: Int = 30):
        """Build a k-d tree from a list of points.

        Copies point data into a contiguous buffer for cache-friendly access,
        then builds the tree recursively with the given leaf_size.
        """
        self.size = len(points)
        self.root = UnsafePointer[KDNode]()
        self.leaf_size = leaf_size

        if self.size == 0:
            self.num_dimensions = 0
            self.coord_buffer = UnsafePointer[KDFloat]()
            self.original_indices = UnsafePointer[Int]()
            return

        self.num_dimensions = len(points[0])

        # Copy from List[Point] into contiguous row-major buffer
        self.coord_buffer = UnsafePointer[KDFloat].alloc(self.size * self.num_dimensions)
        self.original_indices = UnsafePointer[Int].alloc(self.size)
        for i in range(self.size):
            self.original_indices[i] = points[i].original_index
            for j in range(self.num_dimensions):
                self.coord_buffer[i * self.num_dimensions + j] = points[i].coords[j]

        # Build tree from buffer indices
        var indices = List[Int]()
        for i in range(self.size):
            indices.append(i)

        self.root = self._build_recursive(indices, 0)

    @always_inline
    fn get_coord(self, buffer_index: Int, dim: Int) -> KDFloat:
        return self.coord_buffer[buffer_index * self.num_dimensions + dim]

    @always_inline
    fn distance_squared_to_buffer_point(
        self, query_ptr: UnsafePointer[KDFloat], buffer_index: Int
    ) -> KDFloat:
        """SIMD-vectorized squared Euclidean distance from query to a buffer point."""
        var point_ptr = self.coord_buffer + buffer_index * self.num_dimensions
        var sum_sq: KDFloat = 0.0
        var k = 0
        while k + KD_SIMD_W <= self.num_dimensions:
            var diff = query_ptr.load[width=KD_SIMD_W](k) - point_ptr.load[width=KD_SIMD_W](k)
            sum_sq += (diff * diff).reduce_add()
            k += KD_SIMD_W
        while k < self.num_dimensions:
            var diff = query_ptr[k] - point_ptr[k]
            sum_sq += diff * diff
            k += 1
        return sum_sq

    fn _build_recursive(
        inout self,
        owned indices: List[Int],
        depth: Int,
    ) -> UnsafePointer[KDNode]:
        """Recursively build the k-d tree."""
        if len(indices) == 0:
            return UnsafePointer[KDNode]()

        # LEAF NODE: brute-force scan at query time
        if len(indices) <= self.leaf_size:
            var leaf_refs = UnsafePointer[PointRef].alloc(len(indices))
            for i in range(len(indices)):
                var buf_idx = indices[i]
                leaf_refs[i] = PointRef(buf_idx, self.original_indices[buf_idx])
            var node_ptr = UnsafePointer[KDNode].alloc(1)
            node_ptr.init_pointee_move(KDNode(
                axis=-1, split_value=0.0, point_ref=PointRef(0, 0),
                left=UnsafePointer[KDNode](), right=UnsafePointer[KDNode](),
                leaf_indices=leaf_refs, leaf_count=len(indices)
            ))
            return node_ptr

        # INTERNAL NODE
        var axis = depth % self.num_dimensions
        self._sort_by_axis(indices, axis)

        var median_pos = len(indices) // 2
        var median_buf_idx = indices[median_pos]

        var node_ptr = UnsafePointer[KDNode].alloc(1)
        node_ptr.init_pointee_move(KDNode(
            axis=axis,
            split_value=self.get_coord(median_buf_idx, axis),
            point_ref=PointRef(median_buf_idx, self.original_indices[median_buf_idx]),
            left=UnsafePointer[KDNode](), right=UnsafePointer[KDNode](),
            leaf_indices=UnsafePointer[PointRef](), leaf_count=0
        ))

        var left_indices = List[Int]()
        var right_indices = List[Int]()

        for i in range(median_pos):
            left_indices.append(indices[i])

        for i in range(median_pos + 1, len(indices)):
            right_indices.append(indices[i])

        node_ptr[].left = self._build_recursive(left_indices^, depth + 1)
        node_ptr[].right = self._build_recursive(right_indices^, depth + 1)

        return node_ptr

    fn _sort_by_axis(self, inout indices: List[Int], axis: Int):
        """Sort indices by coordinate value on given axis (insertion sort)."""
        for i in range(1, len(indices)):
            var key_idx = indices[i]
            var key_val = self.get_coord(key_idx, axis)
            var j = i - 1

            while j >= 0 and self.get_coord(indices[j], axis) > key_val:
                indices[j + 1] = indices[j]
                j -= 1

            indices[j + 1] = key_idx

    fn k_nearest_neighbors(self, query_ptr: UnsafePointer[KDFloat], k: Int) -> List[Neighbor]:
        """Find the k nearest neighbors to the query point.

        Args:
            query_ptr: Pointer to the query point's feature data (length = num_dimensions).
            k: Number of nearest neighbors to find.

        Returns:
            List of Neighbor structs sorted by distance (ascending).
        """
        var heap = MaxHeap(k)
        self._knn_search(self.root, query_ptr, heap)

        var result = heap.to_sorted_list()
        for i in range(len(result)):
            result[i].distance = sqrt(result[i].distance)

        return result

    fn _knn_search(
        self,
        node_ptr: UnsafePointer[KDNode],
        query_ptr: UnsafePointer[KDFloat],
        inout heap: MaxHeap,
    ):
        """Recursive k-nearest neighbor search with pruning."""
        if not node_ptr:
            return

        var node = node_ptr[]

        # LEAF: brute-force scan all points in the leaf
        if node.leaf_count > 0:
            for i in range(node.leaf_count):
                var pt_ref = node.leaf_indices[i]
                var dist_sq = self.distance_squared_to_buffer_point(query_ptr, pt_ref.buffer_index)
                heap.push(Neighbor(pt_ref.original_index, dist_sq))
            return

        # INTERNAL: check this node's point
        var dist_sq = self.distance_squared_to_buffer_point(query_ptr, node.point_ref.buffer_index)
        heap.push(Neighbor(node.point_ref.original_index, dist_sq))

        var diff = query_ptr[node.axis] - node.split_value
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
        self._knn_search(first, query_ptr, heap)

        # Only search the other subtree if:
        # 1. We don't have k neighbors yet, OR
        # 2. The splitting plane is closer than our worst (farthest) neighbor
        if not heap.is_full() or diff_sq < heap.max_distance():
            self._knn_search(second, query_ptr, heap)

    fn nearest_neighbor(self, query_ptr: UnsafePointer[KDFloat]) -> Neighbor:
        """Find the single nearest neighbor to the query point."""
        var best = Neighbor(-1, KDFloat.MAX)
        self._nn_search(self.root, query_ptr, best)
        best.distance = sqrt(best.distance)
        return best

    fn _nn_search(
        self,
        node_ptr: UnsafePointer[KDNode],
        query_ptr: UnsafePointer[KDFloat],
        inout best: Neighbor,
    ):
        """Recursive nearest neighbor search with pruning."""
        if not node_ptr:
            return

        var node = node_ptr[]

        # LEAF: brute-force scan
        if node.leaf_count > 0:
            for i in range(node.leaf_count):
                var pt_ref = node.leaf_indices[i]
                var dist_sq = self.distance_squared_to_buffer_point(query_ptr, pt_ref.buffer_index)
                if dist_sq < best.distance:
                    best = Neighbor(pt_ref.original_index, dist_sq)
            return

        # INTERNAL
        var dist_sq = self.distance_squared_to_buffer_point(query_ptr, node.point_ref.buffer_index)
        if dist_sq < best.distance:
            best = Neighbor(node.point_ref.original_index, dist_sq)

        var diff = query_ptr[node.axis] - node.split_value
        var diff_sq = diff * diff

        var first: UnsafePointer[KDNode]
        var second: UnsafePointer[KDNode]

        if diff < 0:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left

        self._nn_search(first, query_ptr, best)

        if diff_sq < best.distance:
            self._nn_search(second, query_ptr, best)

    fn get_depth(self) -> Int:
        """Get the maximum depth of the tree."""
        return self._get_depth_recursive(self.root)

    fn _get_depth_recursive(self, node_ptr: UnsafePointer[KDNode]) -> Int:
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
        if not node_ptr:
            return 0
        return (
            1
            + self._count_recursive(node_ptr[].left)
            + self._count_recursive(node_ptr[].right)
        )
