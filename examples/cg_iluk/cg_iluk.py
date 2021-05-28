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
    random.seed(12345)

    scalar_t = float
    lno_t = int
    size_type = int

    execution_space = "Kokkos::DefaultExecutionSpace"
    memory_space = "Kokkos::DefaultExecutionSpace::memory_space"
    crsmat_t = CrsMatrix(scalar_t, lno_t, execution_space, None, size_type)
    KernelHandle = KokkosKernelsHandle(size_type, lno_t, scalar_t, execution_space, memory_space, memory_space)

    nx = 10
    ny = 10
    algo_spiluk = Algo.LVLSCHED_RP
    algo_sptrsv = Algo.LVLSCHED_RP
    k = 0
    team_size = -1
    prec = 0

    parser = argparse.ArgumentParser("CGSolve with ILU(k) preconditioner options: (simple Laplacian matrix on a cartesian grid where nrows = nx * ny)")
    parser.add_argument('-nx', type=int, help="grid points in x direction (default: 10)")
    parser.add_argument('-ny', type=int, help="grid points in y direction (default: 10)")
    parser.add_argument('-N', type=int, help="grid points in x direction (default: 10)")
    parser.add_argument('-M', type=int, help="grid points in y direction (default: 10)")

    parser.add_argument('-k', type=int, help="fill level (default: 0)")
    parser.add_argument('-algospiluk', type=str, help="SPILUK kernel implementation (default: lvlrp)")
    parser.add_argument('-algosptrsv', type=str, help="SPTRSV kernel implementation (default: lvlrp)\n[OPTIONS]: lvlrp, lvltp1")
    parser.add_argument('-ts', type=int, help="team size only when lvltp1 is used (default: -1)")
    parser.add_argument('-prec', type=int, help="whether having preconditioner or not (default: 0)")
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
    if args.algospiluk:
        if args.algospliuk == "lvlrp":
            algo_spiluk = Algo.LVLSCHED_RP
        if args.algospliuk == "lvltp1":
            algo_spiluk = Algo.LVLSCHED_TP1
    if args.algosptrsv:
        if args.algosptrsv == "lvlrp":
            algo_sptrsv = Algo.LVLSCHED_RP
        if args.algosptrsv == "lvltp1":
            algo_sptrsv = Algo.LVLSCHED_TP1
    if args.ts:
        team_size = args.ts
    if args.prec:
        prec = args.prec

    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    print("CGSolve execution_space: ", pk.get_default_space(), ", memory_space: ", 
            pk.get_default_memory_space(pk.get_default_space()))

    timer = pk.Timer()
    tolerance = scalar_t(0.0000000001)
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

    kh_spiluk = KernelHandle()
    kh_sptrsv_L = KernelHandle()
    kh_sptrsv_U = KernelHandle()

    print("Create SPILUK handle ...")
    if algo_spiluk == Algo.LVLSCHED_RP:
        kh_spiluk.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("Kernel implementation type: ", end="", flush=True)
        kh_spiluk.get_spiluk_handle().print_algorithm()
    elif algo_spiluk == Algo.LVLSCHED_TP1:
        kh_spiluk.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_TP1, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("TP1 set team_size = ", team_size)
        if team_size != -1:
            kh_spiluk.get_spiluk_handle().set_team_size(team_size)
        print("Kernel implementation type: ", end="", flush=True)
        kh_spiluk.get_spiluk_handle().print_algorithm()
    else:
        kh_spiluk.create_spiluk_handle(SPILUKAlgorithm.SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1))
        print("Kernel implementation type: ", end="", flush=True)
        kh_spiluk.get_spiluk_handle().print_algorithm()

    print("Create SPTRSV handle for L ...")
    is_lower_tri = True
    if algo_sptrsv == Algo.LVLSCHED_RP:
        kh_sptrsv_L.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_RP, N, is_lower_tri)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_L.get_sptrsv_handle().print_algorithm()
    elif algo_sptrsv == Algo.LVLSCHED_TP1:
        kh_sptrsv_L.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_TP1, N, is_lower_tri)
        print("TP1 set team_size = ", team_size)
        if team_size != -1:
            kh_sptrsv_L.get_sptrsv_handle().set_team_size(team_size)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_L.get_sptrsv_handle().print_algorithm()
    else:
        kh_sptrsv_L.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_RP, N, is_lower_tri)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_L.get_sptrsv_handle().print_algorithm()

    print("Create SPTRSV handle for U ...")
    is_lower_tri = False 
    if algo_sptrsv == Algo.LVLSCHED_RP:
        kh_sptrsv_U.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_RP, N, is_lower_tri)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_U.get_sptrsv_handle().print_algorithm()
    elif algo_sptrsv == Algo.LVLSCHED_TP1:
        kh_sptrsv_U.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_TP1, N, is_lower_tri)
        print("TP1 set team_size = ", team_size)
        if team_size != -1:
            kh_sptrsv_U.get_sptrsv_handle().set_team_size(team_size)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_U.get_sptrsv_handle().print_algorithm()
    else:
        kh_sptrsv_U.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_RP, N, is_lower_tri)
        print("Kernel implementation type: ", end="", flush=True)
        kh_sptrsv_U.get_sptrsv_handle().print_algorithm()

    spiluk_handle = kh_spiluk.get_spiluk_handle()
    sptrsvL_handle = kh_sptrsv_L.get_spiluk_handle()
    sptrsvU_handle = kh_sptrsv_U.get_spiluk_handle()

    L_row_map = pk.View([N + 1], dtype = lno_t)
    L_entries = pk.View([spiluk_handle.get_nnzL()], dtype = lno_t)
    L_values = pk.View([spiluk_handle.get_nnzL()], dtype = scalar_t)
    U_row_map = pk.View([N + 1], dtype = lno_t)
    U_entries = pk.View([spiluk_handle.get_nnzL()], dtype = lno_t)
    U_values = pk.View([spiluk_handle.get_nnzL()], dtype = scalar_t)

    print("Run symbolic phase ...")
    init_time = timer.seconds()
    timer.reset()
    initial_begin = timer.seconds()
    begin = initial_begin
    spiluk_symbolic(ptr(kh_spiluk), fill_lev, A.graph.row_map, A.graph.entries, L_row_map, L_entries, U_row_map, U_entries)
    fence()
    end = timer.seconds()
    print("ILU(", fill_lev, ") Symbolic Time: ", end-begin, " seconds")

    resize(L_entries, spiluk_handle.get_nnzL())
    resize(L_values, spiluk_handle.get_nnzL())
    resize(U_entries, spiluk_handle.get_nnzU())
    resize(U_values, spiluk_handle.get_nnzU())

    # NOTE: views object metadata does not update when underlying object updates
    print("L_row_map size = ", L_row_map.extent(0))
    print("L_entries size = ", L_entries.array.shape[0])
    print("L_values size  = ", L_values.array.shape[0])
    print("U_row_map size = ", U_row_map.extent(0))
    print("U_entries size = ", U_entries.array.shape[0])
    print("U_values size  = ", U_values.array.shape[0])
    print("ILU(k) fill_level: ", fill_lev)
    print("ILU(k) fill-factor: ", (spiluk_handle.get_nnzL() + spiluk_handle.get_nnzU() - N)/nnzA)
    print("num levels: ", spiluk_handle.get_num_levels())
    print("max num rows levels: ", spiluk_handle.get_level_maxrows(), "\n")

    begin = timer.seconds()
    spiluk_numeric(ptr(kh_spiluk), fill_lev, A.graph.row_map, A.graph.entries, A.values, 
            L_row_map, L_entries, L_values, U_row_map, U_entries, U_values)
    fence()
    end = timer.seconds()
    print("ILU(", fill_lev, ") Numeric Time: ", end-begin, " seconds")

    sptrsv_symbolic(ptr(kh_sptrsv_L), L_row_map, L_entries)
    sptrsv_symbolic(ptr(kh_sptrsv_U), U_row_map, U_entries)

    b  = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    x  = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    xx = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)

    p  = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    Ap = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    r  = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    z  = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)
    Linvr = pk.View([N], dtype=scalar_t, layout=pk.LayoutLeft)

    h_xx = create_mirror_view(xx)

    # Views don't currently support span
    for i in range(len(xx)):
        h_xx[i] = random.random()
        # h_xx[i] = 1/(i+1)

    deep_copy(xx, h_xx)

    spmv(char_ptr("N"), one, A, xx, zero, b)

    deep_copy(r, b); fence()

    begin = timer.seconds()

    spmv(char_ptr("N"), one, A, x, zero, Ap); fence()
    axpy(mone, Ap, r); fence()

    if prec == 1:
        sptrsv_solve(ptr(kh_sptrsv_L), L_row_map, L_entries, L_values, r, Linvr); fence()
        sptrsv_solve(ptr(kh_sptrsv_U), U_row_map, U_entries, U_values, Linvr, z), fence()
    else:
        deep_copy(z, r)

    r_old_dot = dot(r, z)
    
    norm_res = math.sqrt(r_old_dot)

    deep_copy(p, z)
    fence()

    k = 0

    while tolerance < norm_res and k < N:
        spmv(char_ptr("N"), one, A, p, zero, Ap); fence()
        pAp_dot = dot(p, Ap); fence()
        alpha = r_old_dot / pAp_dot

        axpy(alpha, p, x); fence()
        axpy(-alpha, Ap, r); fence()

        if prec == 1:
            sptrsv_solve(ptr(kh_sptrsv_L), L_row_map, L_entries, L_values, r, Linvr); fence()
            sptrsv_solve(ptr(kh_sptrsv_U), U_row_map, U_entries, U_values, Linvr, z), fence()
        else:
            deep_copy(z, r)

        r_dot = dot(r, z); fence()
        beta = r_dot / r_old_dot

        axpby(one, z, beta, p); fence()

        r_old_dot = r_dot
        norm_res = math.sqrt(r_old_dot)

        k += 1
    
    end = timer.seconds()

    time = end - begin
    total_time = end - initial_begin

    print("    Results: N (%d), time (%g s), iterations (%d), norm_res(%.20lf)" % (N, time, k, norm_res))

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("cg_iluk", args.cuda, nx, ny, init_time, total_time))


if __name__ == "__main__":
    run()
