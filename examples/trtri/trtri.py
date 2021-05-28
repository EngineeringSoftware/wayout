import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    M = int(sys.argv[1])

    A = pk.View([M, M], dtype=pk.double)

    rand_pool = Random_XorShift64_Pool("Kokkos::DefaultExecutionSpace")(13718)
    fill_random(A, rand_pool, 10.)

    alpha = 1.

    kernels.trtri(char_ptr("L"), char_ptr("N"), A)


if __name__ == "__main__":
    run()

