import pykokkos as pk

import argparse
import copy
import enum
import importlib
import math
import random
import sys

import kernels 
from kernels import *

EXPAND_FACT = 6

class Algo(enum.Enum):
    DEFAULT = 0
    LVLSCHED_RP = 1
    LVLSCHED_TP1 = 2
    
def run() -> None: 

    scalar_t = float
    lno_t = int
    size_type = int

    execution_space = "Kokkos::DefaultExecutionSpace"
    memory_space = "Kokkos::DefaultExecutionSpace::memory_space"
    crsmat_t = CrsMatrix(scalar_t, lno_t, execution_space, None, size_type)
    KernelHandle = KokkosKernelsHandle(size_type, lno_t, scalar_t, execution_space, memory_space, memory_space)

    nx = 10
    ny = 10
    test_algo = Algo.LVLSCHED_RP
    k = 0
    team_size = -1

    parser = argparse.ArgumentParser("ILU(k) Options: (simple Laplacian matrix on a cartesian grid where nrows = nx * ny)")
    parser.add_argument('-nx', type=int, help="grid points in x direction (default: 10)")
    parser.add_argument('-ny', type=int, help="grid points in y direction (default: 10)")
    parser.add_argument('-N', type=int, help="grid points in x direction (default: 10)")
    parser.add_argument('-M', type=int, help="grid points in y direction (default: 10)")

    parser.add_argument('-k', type=int, help="fill level (default: 0)")
    parser.add_argument('-algo', type=str, help="kernel implementation (default: lvlrp)\n[OPTIONS]: lvlrp, lvltp1")
    parser.add_argument('-ts', type=int, help="team size only when lvltp1 is used (default: -1)")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file") 

    args = parser.parse_args()
    if args.nx:
        nx = args.nx
    if args.ny:
        ny = args.ny
    if args.N:
        nx = args.N
    if args.M:
        ny = args.M

    if args.k:
        k = args.k
    if args.algo:
        if args.algo == "lvlrp":
            test_algo = Algo.LVLSCHED_RP
        if args.algo == "lvltp1":
            test_algo = Algo.LVLSCHED_TP1
    if args.ts:
        team_size = args.ts

    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    print("ILU(k) execution_space: ", pk.get_default_space(), ", memory_space: ", 
            pk.get_default_memory_space(pk.get_default_space()))

    timer = pk.Timer()
    one = scalar_t(1.)
    zero = scalar_t(0.)
    mone = scalar_t(-1.)

    mat_structure = pk.View([2, 3], dtype=lno_t, space=pk.HostSpace)
    mat_structure[0][0] = nx
    mat_structure[0][1] = 0 
    mat_structure[0][2] = 0 
    mat_structure[1][0] = ny 
    mat_structure[1][1] = 0
    mat_structure[1][2] = 0 

    A = generate_structured_matrix2D("FD", mat_structure, template_args=[crsmat_t])

    graph = A.graph
    N = graph.numRows()
    fill_lev = lno_t(k)
    nnzA = A.graph.entries.extent(0)
    print("Matrix size: ", N, "x ", N, ", nnz = ", nnzA)

    kh = KernelHandle()

    print("Create handle ...")
    if test_algo == Algo.LVLSCHED_RP:
        kh.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("Kernel implementation type: ", end="", flush=True)
        kh.get_spiluk_handle().print_algorithm()
    elif test_algo == Algo.LVLSCHED_TP1:
        kh.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_TP1, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("TP1 set team_size = ", team_size)
        if team_size != -1:
            kh.get_spiluk_handle().set_team_size(team_size)
        print("Kernel implementation type: ", end="", flush=True)
        kh.get_spiluk_handle().print_algorithm()
    else:
        kh.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("Kernel implementation type: ", end="", flush=True)
        kh.get_spiluk_handle().print_algorithm()

    spiluk_handle = kh.get_spiluk_handle()

    L_row_map = pk.View([N + 1], dtype = lno_t, layout=pk.LayoutLeft)
    L_entries = pk.View([spiluk_handle.get_nnzL()], dtype = lno_t, layout=pk.LayoutLeft)
    L_values = pk.View([spiluk_handle.get_nnzL()], dtype = scalar_t, layout=pk.LayoutLeft)
    U_row_map = pk.View([N + 1], dtype = lno_t, layout=pk.LayoutLeft)
    U_entries = pk.View([spiluk_handle.get_nnzL()], dtype = lno_t, layout=pk.LayoutLeft)
    U_values = pk.View([spiluk_handle.get_nnzL()], dtype = scalar_t, layout=pk.LayoutLeft)

    print("Expected L_row_map size = ", L_row_map.extent(0))
    print("Expected L_entries size = ", L_entries.extent(0))
    print("Expected L_values size  = ", L_values.extent(0))
    print("Expected U_row_map size = ", U_row_map.extent(0))
    print("Expected U_entries size = ", U_entries.extent(0))
    print("Expected U_values size  = ", U_values.extent(0))

    print("Run symbolic phase ...")
    init_time = timer.seconds()
    timer.reset()
    initial_begin = timer.seconds()
    begin = initial_begin
    spiluk_symbolic(ptr(kh), fill_lev, A.graph.row_map, A.graph.entries, L_row_map, L_entries, U_row_map, U_entries)
    fence()
    end = timer.seconds()
    print("ILU(", fill_lev, ") Symbolic Time: ", end-begin, " seconds")

    resize(L_entries, spiluk_handle.get_nnzL())
    resize(L_values, spiluk_handle.get_nnzL())
    resize(U_entries, spiluk_handle.get_nnzU())
    resize(U_values, spiluk_handle.get_nnzU())

    print("Actual L_row_map size = ", L_row_map.extent(0))
    print("Actual L_entries size = ", L_entries.extent(0))
    print("Actual L_values size  = ", L_values.extent(0))
    print("Actual U_row_map size = ", U_row_map.extent(0))
    print("Actual U_entries size = ", U_entries.extent(0))
    print("Actual U_values size  = ", U_values.extent(0))
    print("ILU(k) fill_level: ", fill_lev)
    print("ILU(k) fill-factor: ", (spiluk_handle.get_nnzL() + spiluk_handle.get_nnzU() - N)/nnzA)
    print("num levels: ", spiluk_handle.get_num_levels())
    print("max num rows levels: ", spiluk_handle.get_level_maxrows(), "\n")

    print("Run ILU(k) numeric phase ...")
    begin = timer.seconds()
    spiluk_numeric(ptr(kh), fill_lev, A.graph.row_map, A.graph.entries, A.values, 
            L_row_map, L_entries, L_values, U_row_map, U_entries, U_values)
    fence()
    end = timer.seconds()
    total_time = end - initial_begin
    print("ILU(", fill_lev, ") Numeric Time: ", end-begin, " seconds")

    L = crsmat_t(char_ptr("L"), N, N, spiluk_handle.get_nnzL(), L_values, L_row_map, L_entries)
    U = crsmat_t(char_ptr("U"), N, N, spiluk_handle.get_nnzU(), U_values, U_row_map, U_entries)
    e_one = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    bb = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    bb_tmp = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)

    deep_copy(e_one, scalar_t(1))

    spmv(char_ptr("N"), one, A, e_one, zero, bb)
    spmv(char_ptr("N"), one, U, e_one, zero, bb_tmp)
    spmv(char_ptr("N"), one, L, bb_tmp, mone, bb)

    bb_nrm = nrm2(bb)

    print("Row-sum difference: nrm2(A*e-L*U*e) = ", bb_nrm)

    kh.destroy_spiluk_handle()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("sparse_iluk", args.cuda, nx, ny, init_time, total_time))


if __name__ == "__main__":
    run()
