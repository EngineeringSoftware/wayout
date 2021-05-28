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

    max_loc = iamax(x)

    expected_result = h_x[0]
    expected_max_loc = 0
    for i in range(N):
        val = abs(h_x[i])
        if val > expected_result:
            expected_result = val
            expected_max_loc = i + 1

    print("Iamax of X: %i, Expected: %i\n" % (max_loc, expected_max_loc))


if __name__ == "__main__":
    run()

