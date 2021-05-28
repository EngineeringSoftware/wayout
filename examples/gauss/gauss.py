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

tolerance = 1e-6
one = 1.0

def run() -> None: 
    numRows = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, help="Unused")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file") 

    args = parser.parse_args()
    if args.N:
        numRows = args.N
    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    timer = pk.Timer()
    nnz = numRows * 20
    A = kk_generate_diagonally_dominant_sparse_matrix(numRows, numRows, nnz, 2, 100, 1.05, template_args=[crs_matrix_type])
    print(f"Generated a matrix with {numRows} rows/cols, and {nnz} entries.")

    handle = KokkosKernelsHandle(int, int, float, DefaultExecSpace, DefaultHostSpace, DefaultHostSpace)()
    handle.create_gs_handle(kernels.GSAlgorithm.GS_DEFAULT)
    handle_ptr = ptr(handle)

    init_time = timer.seconds()
    timer.reset()

    row_map = A.graph.row_map
    entries = A.graph.entries
    values = A.values
    gauss_seidel_symbolic(handle_ptr, numRows, numRows, row_map, entries, False)
    gauss_seidel_numeric(handle_ptr, numRows, numRows, row_map, entries, values, False)

    x = pk.View([numRows], pk.double)
    b = pk.View([numRows], pk.double)
    res = pk.View([numRows], pk.double)

    # skip mirror views
    for i in range(numRows):
        b[i] = 3 * random.random()

    initialRes = nrm2(b)
    scaledResNorm = 1
    firstIter = True
    numIters = 0
    while(scaledResNorm > tolerance):
        forward_sweep_gauss_seidel_apply(handle_ptr, numRows, numRows, row_map, 
                entries, values, x, b, firstIter, firstIter, one, 1)
        firstIter = False
        deep_copy(res, b)
        spmv(char_ptr("N"), one, A, x, -one, res)
        scaledResNorm = nrm2(res) / initialRes
        numIters += 1
        print(f"Iteration {numIters} scaled residual norm: {scaledResNorm}")

    time = timer.seconds()
    print(f"SUCCESS: converged in {numIters} iterations.")

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("gauss", args.cuda, numRows, 1, init_time, time))

 
if __name__ == "__main__":
    run()
