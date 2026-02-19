from csv import load_data

from math import floor
from collections import List, Set
from memory import memset_zero, stack_allocation
from sys import info, simdwidthof
import sys
from random import rand, randint, random_ui64, random_si64
from memory.unsafe_pointer import UnsafePointer
from memory import bitcast

alias T = Float32

# Get optimal number of elements to run with vectorize at compile time.
# 2x or 4x helps with pipelining and running multiple SIMD operations in parallel.
alias nelts = get_simd_width()

@parameter
fn get_simd_width() -> Int:
    # if info.is_apple_silicon():
    #     return 4 * simdwidthof[type]()
    # else:
    return (2 * simdwidthof[T]())


struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[T]

    # Initialize zeroing all values
    fn __init__(out self):
        self.data = UnsafePointer[T].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

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





def train_test_split(X: List[Float32], y: List[Float32], nfeatures: Int, train_split:Float64) -> Tuple[List[Float32], List[Float32], List[Float32], List[Float32]]:
    # X -> List of length (n*m), input data
    # y -> List of length (n*1), input labels
    # train_split -> float from 0.0 - 1.0
    # nfeatures -> int for feature count

    # Useful for later
    var nrows = int(len(X) // nfeatures)
    var split_ = int(floor(nrows * train_split))

    # Pre-generating lists to populate
    X_train = List[Float32]()
    y_train = List[Float32]()
    X_test = List[Float32]()
    y_test = List[Float32]()

    indexes = List[Int]()
    trainindexes = List[Int]()

    # Populating Indices List
    for i in range(nrows):
        indexes.append(i)

    # Creating random indices based on split
    for i in range(split_):
        var idx = int(random_ui64(min=0, max=nrows-1))
        trainindexes.append(indexes.pop(idx))
        nrows -= 1

    # Copy remaining indexes to testindexes
    testindexes = List[Int]()
    for i in range(len(indexes)):
        testindexes.append(indexes[i])

    # iterating over indices
    # grabs data at all columns for x, and just at index for classes
    for i in range(len(trainindexes)):
        idx = trainindexes[i]
        for j in range(nfeatures): # all features
            X_train.append(X[idx*nfeatures + j])
        y_train.append(y[idx])

    for i in range(len(testindexes)):
        idx = testindexes[i]
        for j in range(nfeatures):
            X_test.append(X[idx*nfeatures + j])
        y_test.append(y[idx])

    return X_train^, X_test^, y_train^, y_test^




def train_test_split64(X: List[Float64], y: List[Float64], nfeatures: Int, train_split:Float64) -> Tuple[List[Float64], List[Float64], List[Float64], List[Float64]]:
    # X -> List of length (n*m), input data
    # y -> List of length (n*1), input labels
    # train_split -> float from 0.0 - 1.0
    # nfeatures -> int for feature count

    # Useful for later
    var nrows = int(len(X) // nfeatures)
    var split_ = int(floor(nrows * train_split))

    # Pre-generating lists to populate
    X_train = List[Float64]()
    y_train = List[Float64]()
    X_test = List[Float64]()
    y_test = List[Float64]()

    indexes = List[Int]()
    trainindexes = List[Int]()

    # Populating Indices List
    for i in range(nrows):
        indexes.append(i)

    # Creating random indices based on split
    for i in range(split_):
        var idx = int(random_ui64(min=0, max=nrows-1))
        trainindexes.append(indexes.pop(idx))
        nrows -= 1

    # Copy remaining indexes to testindexes
    testindexes = List[Int]()
    for i in range(len(indexes)):
        testindexes.append(indexes[i])

    # iterating over indices
    # grabs data at all columns for x, and just at index for classes
    for i in range(len(trainindexes)):
        idx = trainindexes[i]
        for j in range(nfeatures): # all features
            X_train.append(X[idx*nfeatures + j])
        y_train.append(y[idx])

    for i in range(len(testindexes)):
        idx = testindexes[i]
        for j in range(nfeatures):
            X_test.append(X[idx*nfeatures + j])
        y_test.append(y[idx])

    return X_train^, X_test^, y_train^, y_test^


fn validate_scores[rows: Int](y_pred: Matrix[rows, 1], y_true: Matrix[rows, 1]) raises:
    if y_pred.rows != y_true.rows:
        raise "Prediction and True Row number mismatch. Check length of predicted rows."
    else:
        matches = 0
        for i in range(y_pred.rows):
            if y_pred[i, 0] == y_true[i, 0]:
                matches +=1
        print("Testing Accuracy: ", (matches * 100.0 / y_pred.rows), "%", sep="")



    


def main():
    # Import data using csv
    dataraw, cols, rows = load_data(data="mnist.csv")
    featuresraw, _, __ = load_data(data='mnistclasses.csv')

    X_train, X_test, y_train, y_test = train_test_split(X=dataraw^, y=featuresraw^, nfeatures=cols, train_split=0.7)

    training = Matrix[49000, 784]() # HARDCODE IN HERE
    for i in range(training.rows):
        for j in range(training.cols):
            idx = i*cols + j    
            training[i,j] = X_train[idx]

    testing = Matrix[21000, 784]()
    for i in range(testing.rows):
        for j in range(testing.cols):
            idx = i*cols + j    
            testing[i,j] = X_test[idx]

    ytraining = Matrix[49000, 1]()
    for i in range(ytraining.rows):
        for j in range(ytraining.cols):
            idx = i*1 + j    
            ytraining[i,j] = y_train[idx]

    ytesting = Matrix[21000, 1]()
    for i in range(ytesting.rows):
        for j in range(ytesting.cols):
            idx = i*1 + j    
            ytesting[i,j] = y_test[idx]


    training.check_validity()
    testing.check_validity()