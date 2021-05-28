import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = N

    A = pk.View([K, K], dtype=pk.double)
    B = pk.View([M, N], dtype=pk.double)

    rand_pool = Random_XorShift64_Pool("Kokkos::DefaultExecutionSpace")(13718)
    fill_random(A, rand_pool, 10.)
    fill_random(B, rand_pool, 10.)

    alpha = 1.

    kernels.trmm(char_ptr("R"), char_ptr("L"), char_ptr("T"), char_ptr("N"), alpha, A, B)


if __name__ == "__main__":
    run()

