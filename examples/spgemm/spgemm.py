import pykokkos as pk

import argparse
import copy
import importlib
import math
import random
import sys

import kernels 
from kernels import *

DefaultExecSpace = "Kokkos::DefaultExecutionSpace"
DefaultHostSpace = "Kokkos::DefaultExecutionSpace::memory_space"
# DefaultExecSpace = "Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>"
# DefaultHostSpace = "Kokkos::HostSpace"

crs_matrix_type = CrsMatrix(float, int, DefaultExecSpace, None, int)

def makeCrsMatrix(numRows):
    numCols = numRows;
    nnz     = 2 + 3*(numRows - 2) + 2;
    ptr: pk.View1D = pk.View([numRows + 1], pk.int32, layout=pk.Layout.LayoutRight)
    ind: pk.View1D = pk.View([nnz], pk.int32, layout=pk.Layout.LayoutRight)
    val: pk.View1D = pk.View([nnz], pk.double, layout=pk.Layout.LayoutRight)

    two  =  2.0;
    mone = -1.0;

    # Add rows one-at-a-time
    for i in range(numRows + 1):
        if i==0: 
            ptr[0] = 0
            ind[0] = 0   
            ind[1] = 1
            val[0] = two 
            val[1] = mone
      
        elif i==numRows:
            ptr[numRows] = nnz

        elif i==(numRows-1):
            ptr[i] = 2 + 3*(i-1)
            ind[2 + 3*(i-1)] = i-1
            ind[2 + 3*(i-1) + 1] = i
            val[2 + 3*(i-1)] = mone
            val[2 + 3*(i-1) + 1] = two
        else:
            ptr[i] = 2 + 3*(i-1)
            ind[2 + 3*(i-1)] = i-1
            ind[2 + 3*(i-1) + 1] = i
            ind[2 + 3*(i-1) + 2] = i+1
            val[2 + 3*(i-1)] = mone
            val[2 + 3*(i-1) + 1] = two
            val[2 + 3*(i-1) + 2] = mone

    return crs_matrix_type("AA", numRows, numCols, nnz, val, ptr, ind)
    
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
    print("  Number of Rows N = %d, Number of Cols N = %d, Total nnz = %d" % (N, N, 2 + 3*(N - 2) + 2));

    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    one: float = 1.0
    zero: float = 0.0
    tolerance: float = 0.0000000001

    timer = pk.Timer()
    A = makeCrsMatrix(N)

    kh = KokkosKernelsHandle(int, int, float, DefaultExecSpace, DefaultExecSpace, DefaultExecSpace)()
    kh.set_team_work_size(16)
    kh.set_dynamic_scheduling(True)
    # kh.set_verbose(True)

    myalg = "SPGEMM_KK_MEMORY"
    spgemm_algorithm = StringToSPGEMMAlgorithm(myalg)
    kh.create_spgemm_handle(spgemm_algorithm)

    init_time = timer.seconds()
    timer.reset()
    C = crs_matrix_type()
    s1 = timer.seconds()

    spgemm_symbolic(kh, A, False, A, False, C)
    s2 = timer.seconds()

    spgemm_numeric(kh, A, False, A, False, C)
    end = timer.seconds()

    kh.destroy_spgemm_handle()

    time = end
    symbolic_time = s2 - s1
    numeric_time = end - s2

    # with open("out.txt", "w") as f:
    #     for i in range(C.numRows()):
    #         row = C.row(i)
    #         for j in range(row.length):
    #             f.write("%d %.2f\n" % (row.colidx(j), row.value(j)))

    # Print results (problem size, time, number of iterations and final norm residual).
    print( "    Results: N( %d ), overall spgemm time( %g s ), symbolic time( %g s ), numeric time( %g s )" 
            % (N, time, symbolic_time, numeric_time))

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("spgemm", args.cuda, N, 1, init_time, time))

 
if __name__ == "__main__":
    run()
