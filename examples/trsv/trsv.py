import pykokkos as pk

import argparse
import copy
import math
import random
import sys

import kernels 
from kernels import *

DefaultExecSpace = "Kokkos::DefaultExecutionSpace"
matrix_type = CrsMatrix(float, int, DefaultExecSpace, None, int)

def makeCrsMatrix(numRows):
    numCols = numRows;
    nnz     = 2 + 3*(numRows - 2) + 2;

    ptr: pk.View1D = pk.View([numRows + 1], pk.int32, layout=pk.Layout.LayoutLeft)
    ind: pk.View1D = pk.View([nnz], pk.int32, layout=pk.Layout.LayoutLeft)
    val: pk.View1D = pk.View([nnz], pk.double, layout=pk.Layout.LayoutRight)

    ptrIn: pk.View1D = create_mirror_view(ptr)
    indIn: pk.View1D = create_mirror_view(ind) 
    valIn: pk.View1D = create_mirror_view(val) 

    two  =  2.0;
    mone = -1.0;

    # Add rows one-at-a-time
    for i in range(numRows + 1):
        if i==0: 
            ptrIn[0] = 0
            indIn[0] = 0   
            indIn[1] = 1
            valIn[0] = two 
            valIn[1] = mone
      
        elif i==numRows:
            ptrIn[numRows] = nnz

        elif i==(numRows-1):
            ptrIn[i] = 2 + 3*(i-1)
            indIn[2 + 3*(i-1)] = i-1
            indIn[2 + 3*(i-1) + 1] = i
            valIn[2 + 3*(i-1)] = mone
            valIn[2 + 3*(i-1) + 1] = two
        else:
            ptrIn[i] = 2 + 3*(i-1)
            indIn[2 + 3*(i-1)] = i-1
            indIn[2 + 3*(i-1) + 1] = i
            indIn[2 + 3*(i-1) + 2] = i+1
            valIn[2 + 3*(i-1)] = mone
            valIn[2 + 3*(i-1) + 1] = two
            valIn[2 + 3*(i-1) + 2] = mone

    deep_copy(ptr, ptrIn)
    deep_copy(ind, indIn)
    deep_copy(val, valIn)

    return matrix_type("A", numRows, numCols, nnz, val, ptr, ind)


def run() -> None: 

    N = int(sys.argv[1])

    A = makeCrsMatrix(N)
    b = pk.View([N, N], dtype=pk.double)
    x = pk.View([N, N], dtype=pk.double)

    deep_copy(b, -1.0)

    trsv(char_ptr("L"), char_ptr("N"), char_ptr("N"), A, b, x)
 

if __name__ == "__main__":
    run()
