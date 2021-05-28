import pykokkos as pk

import argparse
import copy
import math
import random
import sys

import kernels 
from kernels import *


def run() -> None: 

    scalar_t = float
    lno_t = int
    size_type = int
    device = "Kokkos::DefaultExecutionSpace::device_type"
    execution_space = "Kokkos::DefaultExecutionSpace"
    memory_space = "Kokkos::DefaultExecutionSpace::memory_space"

    KernelHandle = KokkosKernelsHandle(size_type, lno_t, scalar_t, execution_space,
            memory_space, memory_space)

    ZERO = scalar_t(0)
    ONE = scalar_t(1)

    nrows = 5
    nnz = 10

    # Upper triangular solve
    row_map = pk.View([nrows+1], dtype=size_type)
    entries = pk.View([nnz], dtype=lno_t)
    values = pk.View([nnz], dtype=scalar_t)

    hrow_map = create_mirror_view(row_map)
    hentries = create_mirror_view(entries)
    hvalues = create_mirror_view(values)

    hrow_map[0] = 0;
    hrow_map[1] = 2;
    hrow_map[2] = 4;
    hrow_map[3] = 7;
    hrow_map[4] = 9;
    hrow_map[5] = 10;

    hentries[0] = 0;
    hentries[1] = 2;
    hentries[2] = 1;
    hentries[3] = 4;
    hentries[4] = 2;
    hentries[5] = 3;
    hentries[6] = 4;
    hentries[7] = 3;
    hentries[8] = 4;
    hentries[9] = 4;

    deep_copy(row_map, hrow_map)
    deep_copy(entries, hentries)

    deep_copy(values, ONE)

    x = pk.View([nrows], dtype=scalar_t)

    b = pk.View([nrows], dtype=scalar_t)
    deep_copy(b, ONE)

    kh = KernelHandle()
    is_lower_tri = False
    kh.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_TP1, nrows, is_lower_tri)

    sptrsv_symbolic(ptr(kh), row_map, entries, values)
    fence()

    sptrsv_solve(ptr(kh), row_map, entries, values, b, x)
    fence()
    kh.destroy_sptrsv_handle()

    # Lower triangular solve
    row_map = pk.View([nrows+1], dtype=size_type)
    entries = pk.View([nnz], dtype=lno_t)
    values = pk.View([nnz], dtype=scalar_t)

    hrow_map = create_mirror_view(row_map)
    hentries = create_mirror_view(entries)
    hvalues = create_mirror_view(values)
    
    hrow_map[0] = 0;
    hrow_map[1] = 1;
    hrow_map[2] = 2;
    hrow_map[3] = 4;
    hrow_map[4] = 6;
    hrow_map[5] = 10;

    hentries[0] = 0;
    hentries[1] = 1;
    hentries[2] = 0;
    hentries[3] = 2;
    hentries[4] = 2;
    hentries[5] = 3;
    hentries[6] = 1;
    hentries[7] = 2;
    hentries[8] = 3;
    hentries[9] = 4;

    deep_copy(row_map, hrow_map)
    deep_copy(entries, hentries)

    deep_copy(values, ONE)

    x = pk.View([nrows], dtype=scalar_t)

    b = pk.View([nrows], dtype=scalar_t)
    deep_copy(b, ONE)

    kh = KernelHandle()
    is_lower_tri = True
    kh.create_sptrsv_handle(SPTRSVAlgorithm.SEQLVLSCHD_TP1, nrows, is_lower_tri)

    sptrsv_symbolic(ptr(kh), row_map, entries, values)
    fence()

    sptrsv_solve(ptr(kh), row_map, entries, values, b, x)
    fence()
    kh.destroy_sptrsv_handle()


if __name__ == "__main__":
    run()
