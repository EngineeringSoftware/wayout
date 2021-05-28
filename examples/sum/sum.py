import pykokkos as pk

import sys

import kernels 
from kernels import *


def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)

    h_x = create_mirror_view(x)

    rand_pool = Random_XorShift64_Pool("Kokkos::DefaultExecutionSpace")(13718)
    fill_random(x, rand_pool, 10.)

    deep_copy(h_x, x)

    sum_ = kernels.sum(x)

    expected_result = 0
    for i in range(N):
        expected_result += h_x[i]

    print("Sum of X: %lf, Expected: %lf\n" % (sum_, expected_result))


if __name__ == "__main__":
    run()

