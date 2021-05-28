import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    M = int(sys.argv[1])
    N = int(sys.argv[2])

    A = pk.View([M, N], dtype=pk.double)
    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)

    deep_copy(A, 1.)
    deep_copy(x, 3.)

    alpha = 1.
    beta = 0.

    kernels.gemv(char_ptr("N"), alpha, A, x, beta, y)


if __name__ == "__main__":
    run()

