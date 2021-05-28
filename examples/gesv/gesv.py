import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    N = int(sys.argv[1])
    NRHS = int(sys.argv[2])

    A = pk.View([N, N], dtype=pk.double)
    B = pk.View([N, NRHS], dtype=pk.double)
    IPIV = pk.View([N], dtype=pk.double)

    rand_pool = Random_XorShift64_Pool("Kokkos::DefaultExecutionSpace")(13718)
    fill_random(A, rand_pool, 10.)
    fill_random(B, rand_pool, 10.)

    kernels.gesv(A, B, IPIV)


if __name__ == "__main__":
    run()

