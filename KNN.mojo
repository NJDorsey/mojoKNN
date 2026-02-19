from collections import List, Dict
from sys import simdwidthof
from python import Python

from memory import memset_zero, stack_allocation
from sys import info
from random import rand, randint
from memory.unsafe_pointer import UnsafePointer

# import geniris      # I have literally no idea what this is or where it's from, it doesn't seem to be used anywhere, but maybe it's needed for the Iris dataset?
from csv import load_data64, load_data
from split import train_test_split64 , train_test_split

from time import monotonic
from algorithm.functional import vectorize, parallelize

#SQRT
from memory import bitcast

# K-D Tree for fast nearest neighbor search
from kdtree import KDTree, Point, Neighbor, KDFloat


alias T = Float32
alias vtype = DType.int64

# Get optimal number of elements to run with vectorize at compile time.
# 2x or 4x helps with pipelining and running multiple SIMD operations in parallel.
alias nelts = get_simd_width()

@parameter
fn get_simd_width() -> Int:
    return 2 * simdwidthof[T]()

struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[T]

    # Initialize zeroing all values
    fn __init__(out self):
        self.data = UnsafePointer[T].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __del__(owned self):
        self.data.free()

    #Initializes with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[T].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    @staticmethod
    fn randint() -> Self:
        var data = UnsafePointer[T].alloc(rows * cols)
        randint(data, rows * cols, 0, 5)
        return Self(data)

    # Initialize taking a pointer, don't set any elements
    fn __init__(out self, data: UnsafePointer[T]):
        self.data = data

    fn __getitem__(self, y: Int, x: Int) -> T:
        return self.data.load(y * self.cols + x)

    fn __setitem__(mut self, y: Int, x: Int, val: T):
        self.data.store(y * self.cols + x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.store(y * self.cols + x, val)

    
    fn pprint(self):
        print("Matrix size:", self.rows,"x", self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
    
    fn sprint(self, nrows:Int = 5, head: Bool = False):
        if head:
            rstart = 0
            rend = nrows
        else:
            rstart = self.rows - nrows
            rend = self.rows
        print("Matrix size:", self.rows,"x", self.cols)
        print("Showing:", nrows, "rows:")
        for i in range(rstart, rend):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
        
    fn check_validity(self):
        valid = 0
        validbool = 0
        for i in range(self.rows):
            validbool = 0
            for j in range(self.cols):
                if self[i,j]: validbool = 1
            if validbool == 1: valid += 1

        print("Rows with valid data:", valid)


struct Vector[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[vtype]]

    # Initialize zeroing all values
    fn __init__(out self):
        self.data = UnsafePointer[Scalar[vtype]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __del__(owned self):
        self.data.free()

    #Initializes with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[vtype]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    @staticmethod
    fn randint() -> Self:
        var data = UnsafePointer[Scalar[vtype]].alloc(rows * cols)
        randint(data, rows * cols, 0, 5)
        return Self(data)

    # Initialize taking a pointer, don't set any elements
    @implicit
    fn __init__(out self, data: UnsafePointer[Scalar[vtype]]):
        self.data = data

    fn __getitem__(self, y: Int, x: Int) -> Scalar[vtype]:
        return self.load(y, x)

    fn __setitem__(mut self, y: Int, x: Int, val: Scalar[vtype]):
        self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[vtype, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[vtype, nelts]):
        self.data.store(y * self.cols + x, val)

    
    fn pprint(self):
        print("Matrix size:", self.rows,"x", self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
    
    fn sprint(self, nrows:Int = 5, head: Bool = False):
        if head:
            rstart = 0
            rend = nrows
        else:
            rstart = self.rows - nrows
            rend = self.rows
        print("Matrix size:", self.rows,"x", self.cols)
        print("Showing:", nrows, "rows:")
        for i in range(rstart, rend):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
        
    fn check_validity(self):
        valid = 0
        validbool = 0
        for i in range(self.rows):
            validbool = 0
            for j in range(self.cols):
                if self[i,j]: validbool = 1
            if validbool == 1: valid += 1

        print("Rows with valid data:", valid)



struct SQRT:
    # Currently only implements simple approximation using exponential halving (and removing the least important bit in the mantissa.)
    # This works by forcing the binary int to be even, then by cutting it in half.
    # TODO: implement error handling (error is too obvious, negative sqrt, etc.)
    # TODO: Check if always_inline is actually faster
    # NOTE: Is a struct with static methods to maybe eventually implement the more complex look up table method. However, these fast sqrts are much faster. 
    @staticmethod
    @always_inline 
    fn fast_sqrt_float32(n: Float32) -> Float32:
        out = bitcast[DType.uint32, 1](n)
        out = out + (127 << 23)
        out = out >> 1
        return bitcast[DType.float32, 1](out)

    @staticmethod
    fn fast_sqrt_dbl(n: Float64) -> Float64:
        #NOTE: untested, we probably don't need 64s
        out = bitcast[DType.uint64, 1](n)
        out = out+(127<<52)
        out = out>>1
        return bitcast[DType.float64, 1](out)

struct Sorting:
    @staticmethod
    fn simd_sort_quick(mut matrix: Matrix, mut indices: Vector, nrows: Int) -> None:
        """
        Sorts a column vector `matrix` (n x 1) in descending order using Quick Sort.
        Modifies `matrix` and `indices` in place.
        """
        for i in range(nrows):
            indices[i, 0] = i
        Sorting.quick_sort(matrix, indices, 0, nrows - 1)

    @staticmethod
    fn simd_sort_heap(mut matrix: Matrix, mut indices: Vector, nrows: Int) -> None:
        """
        Sorts a column vector `matrix` (n x 1) in descending order using Heap Sort.
        Modifies `matrix` and `indices` in place.
        """
        for i in range(nrows):
            indices[i, 0] = i
        Sorting.heap_sort(matrix, indices)

    @staticmethod
    fn quick_sort(mut matrix: Matrix, mut indices: Vector, low: Int, high: Int) -> None:
        if low < high:
            var pivot_index = Sorting.partition(matrix, indices, low, high)
            Sorting.quick_sort(matrix, indices, low, pivot_index - 1)
            Sorting.quick_sort(matrix, indices, pivot_index + 1, high)

    @staticmethod
    fn partition(mut matrix: Matrix, mut indices: Vector, low: Int, high: Int) -> Int:
        var pivot = matrix[high, 0]
        var i = low - 1

        for j in range(low, high):
            if matrix[j, 0] <= pivot:  # Ascending order (changed this condition)
                i += 1
                var temp_val = matrix[i, 0]
                matrix[i, 0] = matrix[j, 0]
                matrix[j, 0] = temp_val

                var temp_idx = indices[i, 0]
                indices[i, 0] = indices[j, 0]
                indices[j, 0] = temp_idx

        var temp_val = matrix[i + 1, 0]
        matrix[i + 1, 0] = matrix[high, 0]
        matrix[high, 0] = temp_val

        var temp_idx = indices[i + 1, 0]
        indices[i + 1, 0] = indices[high, 0]
        indices[high, 0] = temp_idx

        return i + 1

    @staticmethod
    fn heap_sort(mut matrix: Matrix, mut indices: Vector) -> None:
        """
        Heap Sort for Matrix[n,1] in descending order.
        """
        var n: Int = matrix.rows

        # Build a max heap
        for i in range(n // 2 - 1, -1, -1):
            Sorting.heapify(matrix, indices, n, i)

        # Extract elements one by one
        for i in range(n - 1, 0, -1):
            # Swap values
            var temp_val = matrix[0, 0]
            matrix[0, 0] = matrix[i, 0]
            matrix[i, 0] = temp_val

            # Swap indices
            var temp_idx = indices[0, 0]
            indices[0, 0] = indices[i, 0]
            indices[i, 0] = temp_idx

            # Heapify the reduced heap
            Sorting.heapify(matrix, indices, i, 0)

    @staticmethod
    fn heapify(mut matrix: Matrix, mut indices: Vector, heap_size: Int, root: Int) -> None:
        """
        Heapify a subtree rooted at index `root`, ensuring a max heap.
        """
        var largest: Int = root
        var left: Int = 2 * root + 1
        var right: Int = 2 * root + 2

        if left < heap_size and matrix[left, 0] > matrix[largest, 0]:
            largest = left

        if right < heap_size and matrix[right, 0] > matrix[largest, 0]:
            largest = right

        if largest != root:
            # Swap values
            var temp_val = matrix[root, 0]
            matrix[root, 0] = matrix[largest, 0]
            matrix[largest, 0] = temp_val

            # Swap indices
            var temp_idx = indices[root, 0]
            indices[root, 0] = indices[largest, 0]
            indices[largest, 0] = temp_idx

            # Recursively heapify
            Sorting.heapify(matrix, indices, heap_size, largest)



fn distMatvec(training_data: Matrix, input_pointT: Matrix, mut distmat: Matrix) -> None:
#TESTINGPOINT NEEDS TO BE TRANSPOSED (input_pointT) FOR THIS METHOD TO WORK!
# Calculates euclidean distance using 'matrix multiplication' formula -> except instead of a11b11 + a21b12 + ... it is (a11-b11)^2 + (a21, b12)^2 + ...
# Vectorized!
    for m in range(distmat.rows):
        for k in range(training_data.cols):

            @parameter  
            fn calc_row_euc[nelts: Int](n : Int):   

                distmat.store[nelts](
                    m, n,
                    distmat.load[nelts](m,n) + (training_data.load[nelts](m, k) - input_pointT.load[nelts](k, n)) ** 2
                )

            vectorize[calc_row_euc, nelts, size=distmat.cols]()


fn most_common_item(count_dict: Dict[Int, Int]) -> Float32:
    # Initialize variables to track the most common item
    most_common = -1.0
    highest_count = -1

    # Iterate through the dictionary
    for e in count_dict.items():
        # If this count is higher than the current highest, update the most common item
        if e[].value > highest_count:
            most_common = e[].key
            highest_count = e[].value

    return most_common.cast[DType.float32]()
    #return most_common

fn predict_class(training_classes: Matrix, K: Int, sorted_indices: Vector) raises -> Float32:
    #ASSUMES INCOMING TRAINING CLASS MATRIX ARE ALREADY SORTED!
    counts = Dict[Int, Int]()
    for i in range(K):
        var curidx = int(sorted_indices[i, 0])
        var curvote = int(training_classes[curidx, 0])
        if curvote in counts:
            counts[curvote] += 1
        else:
            counts[curvote] = 1

    #DEFAULTS TO BREAKING TIES BY LOWEST CLASS NUMBER. CAN TOTALLY MAKE THIS RANDOM LATER

    return most_common_item(counts)


fn predict_class_from_neighbors(training_classes: Matrix, neighbors: List[Neighbor]) raises -> Float32:
    """Predict class from k-d tree neighbor results."""
    counts = Dict[Int, Int]()
    for i in range(len(neighbors)):
        var curidx = neighbors[i].index
        var curvote = int(training_classes[curidx, 0])
        if curvote in counts:
            counts[curvote] += 1
        else:
            counts[curvote] = 1

    return most_common_item(counts)


fn matrix_to_points(data: Matrix) -> List[Point]:
    """Convert a Matrix to a List[Point] for k-d tree construction."""
    var points = List[Point]()
    for i in range(data.rows):
        var coords = List[KDFloat]()
        for j in range(data.cols):
            coords.append(KDFloat(data[i, j]))
        points.append(Point(coords, i))  # Store original index
    return points


fn row_to_point(data: Matrix, row: Int) -> Point:
    """Convert a single matrix row to a Point."""
    var coords = List[KDFloat]()
    for j in range(data.cols):
        coords.append(KDFloat(data[row, j]))
    return Point(coords, row)

fn validate_scores(y_pred: Matrix, y_true: Matrix) raises -> None:
    if y_pred.rows != y_true.rows:
        raise "Prediction and True Row number mismatch. Check length of predicted rows."
    else:
        correct = 0
        for i in range(y_pred.rows):
            if y_pred[i, 0] == y_true[i, 0]:
                correct +=1

    print("Correct predictions: ", correct, " / ", y_pred.rows)
    print("Testing Accuracy: ", (correct * 100.0 / y_pred.rows), "%")

from python import Python

# NOTE: I included this when I was trying to implement sklearn metrics for validation, but I couldn't figure out how to get the Python-Mojo 
# interoperability to work for this. I left it here in case I want to try again later, but it's not currently being used.
# I implemented a Mojo-native version instead 

# def validate_with_sklearn(predicted: Matrix, actual: Matrix) -> None:
#     # Import sklearn and numpy
#     sk_metrics = Python.import_module("sklearn.metrics")
#     np = Python.import_module("numpy")

#     # Convert Mojo matrices → Python lists
#     pred_list = List[Float32]()
#     for i in range(predicted.rows):
#         pred_list.append(predicted[i, 0])

#     actual_list = List[Float32]()
#     for i in range(actual.rows):
#         actual_list.append(actual[i, 0])

#     # Convert lists → numpy arrays
#     pred_np = PythonObject(np.array(pred_list))
#     actual_np = PythonObject(np.array(actual_list))

#     # Call sklearn functions
#     cm = PythonObject(sk_metrics.confusion_matrix(actual_np, pred_np))
#     cr = PythonObject(sk_metrics.classification_report(actual_np, pred_np))

#     print("Confusion Matrix:")
#     print(cm)

#     print("\nClassification Report:")
#     print(cr)



@always_inline
fn test_matrix_equal(A: Matrix, B: Matrix) -> Bool:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i, j] != B[i, j]:
                return False
    return True


fn runKNN(mut predictedclasses: Matrix, training: Matrix, testing: Matrix, trainingclasses: Matrix, K: Int) raises:
    """Original brute-force KNN implementation (kept for comparison)."""
    @parameter
    fn predict_one(i: Int):
        # Create local versions inside thread (not shared)
        local_testingpointT = Matrix[testing.cols, 1]()
        local_distmat = Matrix[training.rows, 1]()
        local_sorted_indices = Vector[training.rows, 1]()

        # Copy testing row i into a column vector
        for j in range(testing.cols):
            local_testingpointT[j, 0] = testing[i, j]

        # Compute distances
        try:
            distMatvec(training, local_testingpointT, local_distmat)
        # Sort distances and get indices
            Sorting.simd_sort_quick(local_distmat, local_sorted_indices, local_distmat.rows)
        # Predict class
            predictedclasses[i, 0] = predict_class(trainingclasses, K, local_sorted_indices)
        except:
            print("Running failed.")


    # Run in parallel across testing samples
    parallelize[origins = MutableAnyOrigin, func = predict_one](predictedclasses.rows)


fn runKNN_kdtree(mut predictedclasses: Matrix, training: Matrix, testing: Matrix, trainingclasses: Matrix, K: Int) raises:
    """KNN implementation using k-d tree for fast nearest neighbor search."""
    # Convert training data to points and build the k-d tree once
    print("Building k-d tree from", training.rows, "training points...")
    var train_points = matrix_to_points(training)
    var tree = KDTree(train_points^)
    print("K-D tree built. Depth:", tree.get_depth(), "nodes:", tree.count_nodes())

    # Process each test point in parallel
    @parameter
    fn predict_one_kdtree(i: Int):
        # Convert test row to a Point
        var query = row_to_point(testing, i)

        # Find k nearest neighbors using the k-d tree
        var neighbors = tree.k_nearest_neighbors(query, K)

        # Predict class from neighbors
        try:
            predictedclasses[i, 0] = predict_class_from_neighbors(trainingclasses, neighbors)
        except:
            pass

    parallelize[origins = MutableAnyOrigin, func = predict_one_kdtree](predictedclasses.rows)

fn classification_report(
    y_true: Matrix, 
    y_pred: Matrix, 
    target_names: List[String] = List[String](),
    digits: Int = 2
) raises -> String:
    """
    Build a classification report showing main classification metrics.
    
    Parameters:
    -----------
    y_true : Matrix[n, 1]
        Ground truth labels (column vector)
    y_pred : Matrix[n, 1]
        Predicted labels (column vector)
    target_names : List[String], optional
        Display names for the classes
    digits : Int, default=2
        Number of decimal places for formatting
        
    Returns:
    --------
    report : String
        Text summary of precision, recall, F1-score for each class
    """
    
    if y_true.rows != y_pred.rows:
        raise "y_true and y_pred must have the same number of rows"
    
    # Get unique labels
    var labels = get_unique_labels(y_true, y_pred)
    var n_labels = len(labels)
    
    # Use target_names if provided, otherwise use label values
    var class_names = List[String]()
    if len(target_names) > 0:
        for i in range(len(target_names)):
            class_names.append(target_names[i])
    else:
        for i in range(n_labels):
            class_names.append(str(int(labels[i])))
    
    # Calculate metrics for each class
    var precision_list = List[Float32]()
    var recall_list = List[Float32]()
    var f1_list = List[Float32]()
    var support_list = List[Int]()
    
    for i in range(n_labels):
        var label = labels[i]
        
        # Calculate TP, FP, FN
        var tp: Float32 = 0.0
        var fp: Float32 = 0.0
        var f_n: Float32 = 0.0
        var support: Int = 0
        
        for j in range(y_true.rows):
            var true_val = y_true[j, 0]
            var pred_val = y_pred[j, 0]
            
            if true_val == label:
                support += 1
                if pred_val == label:
                    tp += 1.0
                else:
                    f_n += 1.0
            elif pred_val == label:
                fp += 1.0
        
        # Precision: tp / (tp + fp)
        var precision: Float32 = 0.0
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        
        # Recall: tp / (tp + fn)
        var recall: Float32 = 0.0
        if (tp + f_n) > 0:
            recall = tp / (tp + f_n)
        
        # F1-score: 2 * (precision * recall) / (precision + recall)
        var f1: Float32 = 0.0
        if (precision + recall) > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)
    
    # Calculate averages
    var total_support = y_true.rows
    
    var macro_precision: Float32 = 0.0
    var macro_recall: Float32 = 0.0
    var macro_f1: Float32 = 0.0
    
    for i in range(n_labels):
        macro_precision += precision_list[i]
        macro_recall += recall_list[i]
        macro_f1 += f1_list[i]
    
    macro_precision /= Float32(n_labels)
    macro_recall /= Float32(n_labels)
    macro_f1 /= Float32(n_labels)
    
    var weighted_precision: Float32 = 0.0
    var weighted_recall: Float32 = 0.0
    var weighted_f1: Float32 = 0.0
    
    for i in range(n_labels):
        var weight = Float32(support_list[i]) / Float32(total_support)
        weighted_precision += precision_list[i] * weight
        weighted_recall += recall_list[i] * weight
        weighted_f1 += f1_list[i] * weight
    
    # Calculate accuracy
    var correct: Int = 0
    for i in range(y_true.rows):
        if y_true[i, 0] == y_pred[i, 0]:
            correct += 1
    var accuracy = Float32(correct) / Float32(total_support)
    
    # Build the report string
    var longest_name = 12  # "weighted avg"
    for i in range(len(class_names)):
        if len(class_names[i]) > longest_name:
            longest_name = len(class_names[i])
    
    var report = String("")
    
    # Header
    report += pad_string("", longest_name + 1)
    report += pad_string("precision", 10)
    report += pad_string("recall", 10)
    report += pad_string("f1-score", 10)
    report += pad_string("support", 10)
    report += "\n\n"
    
    # Metrics for each class
    for i in range(n_labels):
        report += pad_string(class_names[i], longest_name)
        report += " "
        report += format_float(precision_list[i], digits, 9)
        report += " "
        report += format_float(recall_list[i], digits, 9)
        report += " "
        report += format_float(f1_list[i], digits, 9)
        report += " "
        report += pad_int(support_list[i], 9)
        report += "\n"
    
    report += "\n"
    
    # Accuracy
    report += pad_string("accuracy", longest_name)
    report += " "
    report += pad_string("", 10)
    report += pad_string("", 10)
    report += format_float(accuracy, digits, 9)
    report += " "
    report += pad_int(total_support, 9)
    report += "\n"
    
    # Macro average
    report += pad_string("macro avg", longest_name)
    report += " "
    report += format_float(macro_precision, digits, 9)
    report += " "
    report += format_float(macro_recall, digits, 9)
    report += " "
    report += format_float(macro_f1, digits, 9)
    report += " "
    report += pad_int(total_support, 9)
    report += "\n"
    
    # Weighted average
    report += pad_string("weighted avg", longest_name)
    report += " "
    report += format_float(weighted_precision, digits, 9)
    report += " "
    report += format_float(weighted_recall, digits, 9)
    report += " "
    report += format_float(weighted_f1, digits, 9)
    report += " "
    report += pad_int(total_support, 9)
    report += "\n"
    
    return report


fn get_unique_labels(y_true: Matrix, y_pred: Matrix) raises -> List[Float32]:
    """Get sorted unique labels from both y_true and y_pred."""
    var seen = Dict[Int, Bool]()

    # Collect unique labels from y_true
    for i in range(y_true.rows):
        var label = int(y_true[i, 0])
        seen[label] = True

    # Collect unique labels from y_pred
    for i in range(y_pred.rows):
        var label = int(y_pred[i, 0])
        seen[label] = True

    # Convert to sorted list
    var labels = List[Float32]()
    for item in seen.items():
        labels.append(Float32(item[].key))

    # Simple bubble sort
    for i in range(len(labels)):
        for j in range(len(labels) - 1 - i):
            if labels[j] > labels[j + 1]:
                var temp = labels[j]
                labels[j] = labels[j + 1]
                labels[j + 1] = temp
    
    return labels^


fn pad_string(s: String, width: Int) -> String:
    """Right-pad a string to the specified width."""
    var result = s
    var current_len = len(s)
    for i in range(width - current_len):
        result += " "
    return result


fn pad_int(value: Int, width: Int) -> String:
    """Format an integer with right padding."""
    var s = str(value)
    var padding = String("")
    for i in range(width - len(s)):
        padding += " "
    return padding + s


fn format_float(value: Float32, decimals: Int, width: Int) -> String:
    """Format a float with specified decimals and width."""
    # Convert to string with basic formatting
    var int_part = int(value)
    var frac_part = value - Float32(int_part)

    # Handle negative numbers
    var is_negative = value < 0
    if is_negative:
        int_part = -int_part
        frac_part = -frac_part

    # Get decimal places
    var multiplier: Float32 = 1.0
    for i in range(decimals):
        multiplier *= 10.0

    var decimal_int = int(frac_part * multiplier + 0.5)  # Round

    # Build the string
    var result = String("")
    if is_negative:
        result += "-"
    result += str(int_part) + "."
    
    # Pad decimal part with leading zeros
    var dec_str = str(decimal_int)
    for i in range(decimals - len(dec_str)):
        result += "0"
    result += dec_str
    
    # Right-pad to width
    var padding = String("")
    for i in range(width - len(result)):
        padding += " "
    
    return padding + result


fn main() raises:
    # np = Python.import_module("numpy") # NOTE: did these to try and use the whole Mojo-Python interoperability thing but I couldn't figure it out.
    # sk_metrics = Python.import_module("sklearn.metrics")
    # sk_metrics.confusion_matrix(actual_np, pred_np)
    # sk_metrics.classification_report(actual_np, pred_np)

    # Load AAPL_LONG dataset (203586 rows, 22 feature columns, no header)
    var dataraw_tuple = load_data(data="HFTData/AAPL_LONG_features_causal.csv")
    # Explicitly move the List[Float32] out of the tuple
    var dataraw = List[Float32]()
    for i in range(len(dataraw_tuple[0])):
        dataraw.append(dataraw_tuple[0][i])
    var cols = dataraw_tuple[1]
    var rows = dataraw_tuple[2]


    var featuresraw_tuple = load_data(data="HFTData/AAPL_LONG_target_causal.csv")
    var featuresraw = List[Float32]()
    for i in range(len(featuresraw_tuple[0])):
        featuresraw.append(featuresraw_tuple[0][i])
    var unused_cols = featuresraw_tuple[1]
    var unused_rows = featuresraw_tuple[2]

    # Ensure train_test_split is called with correct types
    var split_tuple = train_test_split(
        X=dataraw,
        y=featuresraw,
        nfeatures=cols,
        train_split=Float64(0.8)  # 80/20 split for AAPL_LONG dataset
    )
    var X_train = List[Float32]()
    var X_test = List[Float32]()
    var y_train = List[Float32]()
    var y_test = List[Float32]()
    for i in range(len(split_tuple[0])):
        X_train.append(split_tuple[0][i])
    for i in range(len(split_tuple[1])):
        X_test.append(split_tuple[1][i])
    for i in range(len(split_tuple[2])):
        y_train.append(split_tuple[2][i])
    for i in range(len(split_tuple[3])):
        y_test.append(split_tuple[3][i])

    # Hardcoded matrix dimensions for AAPL_LONG dataset (203586 rows, 22 features, 80/20 split)
    # Training: floor(203586 * 0.8) = 162868 rows, Testing: 203586 - 162868 = 40718 rows
    training = Matrix[162868, 22]() # 80% split for AAPL_LONG: 162868 training samples, 22 features
    for i in range(training.rows):
        for j in range(training.cols):
            idx = i*cols + j
            training[i,j] = X_train[idx]

    testing = Matrix[40718, 22]() # 20% split for AAPL_LONG: 40718 testing samples, 22 features
    for i in range(testing.rows):
        for j in range(testing.cols):
            idx = i*cols + j
            testing[i,j] = X_test[idx]

    trainingclasses = Matrix[162868, 1]() # 80% split for AAPL_LONG: 162868 training labels
    for i in range(trainingclasses.rows):
        for j in range(trainingclasses.cols):
            idx = i*1 + j
            trainingclasses[i,j] = y_train[idx]

    testingclasses = Matrix[40718, 1]() # 20% split for AAPL_LONG: 40718 testing labels
    for i in range(testingclasses.rows):
        for j in range(testingclasses.cols):
            idx = i*1 + j
            testingclasses[i,j] = y_test[idx]

    var K: Int = 100
    var NUM_RUNS: Int = 30

    print("=" * 60)
    print("BENCHMARK: K-D Tree vs Brute Force KNN")
    print("Running", NUM_RUNS, "experiments...")
    print("=" * 60)

    # Build the k-d tree once (shared across all runs)
    print("\nBuilding k-d tree from", training.rows, "training points...")
    var train_points = matrix_to_points(training)
    var tree = KDTree(train_points^)
    print("K-D tree built. Depth:", tree.get_depth(), "nodes:", tree.count_nodes())

    # Storage for results
    var kdtree_times = List[Float64]()
    var brute_times = List[Float64]()
    var kdtree_accuracies = List[Float64]()
    var brute_accuracies = List[Float64]()

    # Run experiments
    for run in range(NUM_RUNS):
        print("\rRun", run + 1, "/", NUM_RUNS, end="")

        # ==================== K-D Tree KNN ====================
        var start_time_kdtree = monotonic()
        var predictedclasses_kdtree = Matrix[testingclasses.rows, 1]()

        @parameter
        fn predict_one_kdtree(i: Int):
            var query = row_to_point(testing, i)
            var neighbors = tree.k_nearest_neighbors(query, K)
            try:
                predictedclasses_kdtree[i, 0] = predict_class_from_neighbors(trainingclasses, neighbors)
            except:
                pass

        parallelize[origins = MutableAnyOrigin, func = predict_one_kdtree](predictedclasses_kdtree.rows)
        var end_time_kdtree = monotonic()
        var kdtree_time = Float64(end_time_kdtree - start_time_kdtree) / 1000000000.0

        # Calculate K-D Tree accuracy
        var kdtree_correct = 0
        for i in range(testingclasses.rows):
            if predictedclasses_kdtree[i, 0] == testingclasses[i, 0]:
                kdtree_correct += 1
        var kdtree_acc = Float64(kdtree_correct) / Float64(testingclasses.rows) * 100.0

        # ==================== Brute Force KNN ====================
        var start_time_brute = monotonic()
        var predictedclasses_brute = Matrix[testingclasses.rows, 1]()

        @parameter
        fn predict_one_brute(i: Int):
            local_testingpointT = Matrix[testing.cols, 1]()
            local_distmat = Matrix[training.rows, 1]()
            local_sorted_indices = Vector[training.rows, 1]()
            for j in range(testing.cols):
                local_testingpointT[j, 0] = testing[i, j]
            try:
                distMatvec(training, local_testingpointT, local_distmat)
                Sorting.simd_sort_quick(local_distmat, local_sorted_indices, local_distmat.rows)
                predictedclasses_brute[i, 0] = predict_class(trainingclasses, K, local_sorted_indices)
            except:
                pass

        parallelize[origins = MutableAnyOrigin, func = predict_one_brute](predictedclasses_brute.rows)
        var end_time_brute = monotonic()
        var brute_time = Float64(end_time_brute - start_time_brute) / 1000000000.0

        # Calculate Brute Force accuracy
        var brute_correct = 0
        for i in range(testingclasses.rows):
            if predictedclasses_brute[i, 0] == testingclasses[i, 0]:
                brute_correct += 1
        var brute_acc = Float64(brute_correct) / Float64(testingclasses.rows) * 100.0

        # Store results
        kdtree_times.append(kdtree_time)
        brute_times.append(brute_time)
        kdtree_accuracies.append(kdtree_acc)
        brute_accuracies.append(brute_acc)

    print("\n\nExperiments complete. Saving results to CSV...")

    # Write results to CSV
    with open("benchmark_results.csv", "w") as f:
        f.write("run,kdtree_time,brute_time,speedup,kdtree_accuracy,brute_accuracy\n")
        for i in range(NUM_RUNS):
            var speedup = brute_times[i] / kdtree_times[i]
            var line = str(i + 1) + "," + str(kdtree_times[i]) + "," + str(brute_times[i]) + "," + str(speedup) + "," + str(kdtree_accuracies[i]) + "," + str(brute_accuracies[i]) + "\n"
            f.write(line)

    print("Results saved to benchmark_results.csv")

    # Print summary statistics
    var kdtree_mean: Float64 = 0.0
    var brute_mean: Float64 = 0.0
    for i in range(NUM_RUNS):
        kdtree_mean += kdtree_times[i]
        brute_mean += brute_times[i]
    kdtree_mean /= Float64(NUM_RUNS)
    brute_mean /= Float64(NUM_RUNS)

    print("\n" + "=" * 60)
    print("SUMMARY (", NUM_RUNS, "runs)")
    print("=" * 60)
    print("K-D Tree mean time:    ", kdtree_mean, "s")
    print("Brute Force mean time: ", brute_mean, "s")
    print("Mean speedup:          ", brute_mean / kdtree_mean, "x")
    print("\nRun the analysis notebook for detailed statistics:")
    print("  jupyter notebook benchmark_analysis.ipynb")