import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    M = int(sys.argv[1])
    N = int(sys.argv[2])

    A = pk.View([M, N], dtype=pk.double)
    B = pk.View([N, M], dtype=pk.double)
    C = pk.View([M, M], dtype=pk.double)

    deep_copy(A, 1.)
    deep_copy(B, 2.)

    alpha = 1.
    beta = 0.

    kernels.gemm(char_ptr("N"), char_ptr("N"), alpha, A, B, beta, C)


if __name__ == "__main__":
    run()

