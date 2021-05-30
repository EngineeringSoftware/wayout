import pykokkos as pk

import argparse
import copy
import importlib
import math
import random
import sys

import kernel
from kernel import *

DefaultExecSpace = "Kokkos::DefaultExecutionSpace"
DefaultHostSpace = "Kokkos::DefaultExecutionSpace::memory_space"
# DefaultExecSpace = "Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>"
# DefaultHostSpace = "Kokkos::HostSpace"
    
def makeCrsMatrix(numRows):
    numCols = numRows;
    nnz     = 2 + 3*(numRows - 2) + 2;

    ptr: pk.View1D = pk.View([numRows + 1], pk.int32, layout=pk.Layout.LayoutLeft)
    ind: pk.View1D = pk.View([nnz], pk.int32, layout=pk.Layout.LayoutLeft)
    val: pk.View1D = pk.View([nnz], pk.double, layout=pk.Layout.LayoutRight)

    ptrIn: pk.View1D = Kokkos.create_mirror_view(ptr)
    indIn: pk.View1D = Kokkos.create_mirror_view(ind) 
    valIn: pk.View1D = Kokkos.create_mirror_view(val) 

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

    Kokkos.deep_copy(ptr, ptrIn)
    Kokkos.deep_copy(ind, indIn)
    Kokkos.deep_copy(val, valIn)

    return KokkosSparse.CrsMatrix(float, int, DefaultExecSpace, None, int)(
            "AA", numRows, numCols, nnz, val, ptr, ind)
    
def run() -> None: 
    N = 1024
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, help="Unused")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")

    args = parser.parse_args()
    if args.N:
        N = args.N
    print("Number of Rows N = %d, Number of Cols N = %d, Total nnz = %d\n" % (N, N, 2 + 3*(N - 2) + 2));

    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    one: float = 1.0
    zero: float = 0.0
    tolerance: float = 0.0000000001

    timer = pk.Timer()
    A = makeCrsMatrix(N)

    b: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    x: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    xx: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    p: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    Ap: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    r: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)

    h_xx = Kokkos.create_mirror_view(xx)
    for i in range(len(xx)):
        h_xx[i] = random.random()

    Kokkos.deep_copy(xx, h_xx)

    # b = A*xx
    KokkosSparse.spmv(char_ptr("N"), one, A, xx, zero, b)

    # b -> r
    Kokkos.deep_copy(r, b)

    Kokkos.fence()

    init_time = timer.seconds()
    timer.reset()

    KokkosSparse.spmv(char_ptr("N"), one, A, x, zero, Ap); Kokkos.fence()
    KokkosBlas.axpy(-1.0, Ap, r); Kokkos.fence()

    r_old_dot: float = KokkosBlas.dot(r, r); Kokkos.fence()

    norm_res = math.sqrt(r_old_dot)
    # r -> p
    Kokkos.deep_copy(p, r); Kokkos.fence()

    k = 0

    while tolerance < norm_res and k < N:
        # Ap = A * p
        KokkosSparse.spmv(char_ptr("N"), one, A, p, zero, Ap); Kokkos.fence()
        # pAp_dot = p' * A*p
        pAp_dot: float = KokkosBlas.dot(p, Ap); Kokkos.fence()
        alpha: float = r_old_dot / pAp_dot

        # x = x + alpha*p
        KokkosBlas.axpy(alpha, p, x)
        # r = r + -alpha*A*p
        KokkosBlas.axpy(-alpha, Ap, r)

        r_dot: float = KokkosBlas.dot(r, r)
        beta: float = r_dot / r_old_dot

        # p = r + beta*p
        KokkosBlas.axpby(one, r, beta, p)
        r_old_dot = r_dot
        norm_res = math.sqrt(r_old_dot)

        k += 1

        Kokkos.fence()

    Kokkos.fence()

    time = timer.seconds()

    KokkosBlas.axpby(one, x, -one, xx)
    final_norm_res: float = math.sqrt(KokkosBlas.dot(xx, xx))

    # Print results (problem size, time, number of iterations and final norm residual).
    print("Results: N( %d ), time( %g s), iterations( %d ), final norm_res(%.20lf)" %
            (N, time, k, final_norm_res) );

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("cgsolve", args.cuda, N, 1, init_time, time))

 
if __name__ == "__main__":
    run()
