from collections import List, Dict
from sys import simdwidthof

from memory import memset_zero, stack_allocation
from sys import info
from random import rand, randint
from memory.unsafe_pointer import UnsafePointer

import geniris
from csv import load_data64, load_data
from split import train_test_split64 , train_test_split

from time import monotonic
from algorithm.functional import vectorize, parallelize

#SQRT
from memory import bitcast


alias type = DType.float32
alias vtype = DType.int64

# Get optimal number of elements to run with vectorize at compile time.
# 2x or 4x helps with pipelining and running multiple SIMD operations in parallel.
alias nelts = get_simd_width()

fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()

struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroing all values
    fn __init__(mut self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    #Initializes with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    @staticmethod
    fn randint() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        randint(data, rows * cols, 0, 5)
        return Self(data)

    # Initialize taking a pointer, don't set any elements
    @implicit
    fn __init__(mut self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load(y, x)

    fn __setitem__(mut self, y: Int, x: Int, val: Scalar[type]):
        self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[type, nelts]):
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
    fn __init__(mut self):
        self.data = UnsafePointer[Scalar[vtype]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

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
    fn __init__(mut self, data: UnsafePointer[Scalar[vtype]]):
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
        curidx = Int(sorted_indices[i, 0])
        curvote = Int(training_classes[curidx, 0])
        if curvote in counts:
            counts[curvote] += 1
        else:
            counts[curvote] = 1
    
    #DEFAULTS TO BREAKING TIES BY LOWEST CLASS NUMBER. CAN TOTALLY MAKE THIS RANDOM LATER

    return most_common_item(counts)


fn validate_scores(y_pred: Matrix, y_true: Matrix) raises -> None:
    if y_pred.rows != y_true.rows:
        raise "Prediction and True Row number mismatch. Check length of predicted rows."
    else:
        correct = 0
        for i in range(y_pred.rows):
            if y_pred[i, 0] == y_true[i, 0]:
                correct +=1
        print("Testing Accuracy: {}%".format((correct/y_pred.rows)*100))

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
    parallelize[origins = MutableAnyOrigin, func = predict_one](predictedclasses.rows, predictedclasses.rows)




fn main() raises:
    # Now with working CSV reader
    ## Import data using csv
    dataraw, cols, rows = load_data(data="mnist.csv")
    featuresraw, _, __ = load_data(data='mnistclasses.csv')

    X_train, X_test, y_train, y_test = train_test_split(X=dataraw, y=featuresraw, nfeatures=cols, train_split=0.9985)

    #69895
    training = Matrix[69895, 784]() # HARDCODE IN HERE
    for i in range(training.rows):
        for j in range(training.cols):
            idx = i*cols + j    
            training[i,j] = X_train[idx]

    testing = Matrix[105, 784]()
    for i in range(testing.rows):
        for j in range(testing.cols):
            idx = i*cols + j    
            testing[i,j] = X_test[idx]

    trainingclasses = Matrix[69895, 1]()
    for i in range(trainingclasses.rows):
        for j in range(trainingclasses.cols):
            idx = i*1 + j    
            trainingclasses[i,j] = y_train[idx]

    testingclasses = Matrix[105, 1]()
    for i in range(testingclasses.rows):
        for j in range(testingclasses.cols):
            idx = i*1 + j    
            testingclasses[i,j] = y_test[idx]


    # SET K FOR KNN
    var K: Int = 100


    ### MNIST DATA
    start_time = monotonic()

    predictedclasses = Matrix[testingclasses.rows, 1]()

    runKNN(predictedclasses, training, testing, trainingclasses, K)


    end_time = monotonic()

    execution_time_nanoseconds = end_time - start_time
    execution_time_seconds = execution_time_nanoseconds / 1000000000

    print("Predicted Classes for MNIST:")
    predictedclasses.sprint()

    validate_scores(predictedclasses, testingclasses)


    print("Execution time: ", execution_time_seconds, "seconds or ", execution_time_nanoseconds, "nanoseconds")