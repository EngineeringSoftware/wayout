import pykokkos as pk

import sys

import kernels 
from kernels import *

@pk.workunit
def ErrorCheck(i: int, update: pk.Acc[pk.double], z: pk.View1D[pk.double], solution: pk.double):
    if z[i] != solution:
        update += 1


def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)
    z = pk.View([N], dtype=pk.double)

    deep_copy(x, 6.)
    deep_copy(y, 2.)
    deep_copy(z, 4.)

    alpha = 1.5
    beta = 0. 
    gamma = 1.2

    kernels.update(alpha, x, beta, y, gamma, z)

    solution = gamma * 4. + alpha * 6. + beta * 2.
    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    error_count = pk.parallel_reduce(p, ErrorCheck, z=z, solution=solution)

    if error_count > 0:
        print("Errors: %d\n" % error_count)


if __name__ == "__main__":
    run()

