import pykokkos as pk

import sys

import kernels 
from kernels import *

@pk.workunit
def ErrorCheck(i: int, update: pk.Acc[pk.double], y: pk.View1D[pk.double], solution: pk.double):
    if y[i] != solution:
        update += 1


def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)
    a = pk.View([N], dtype=pk.double)

    deep_copy(x, 3.)
    deep_copy(y, 2.)
    deep_copy(a, 4.)

    alpha = 1.5
    gamma = 1.2

    kernels.mult(gamma, y, alpha, a, x)

    solution = gamma * 2. + alpha * 3 * 4
    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    error_count = pk.parallel_reduce(p, ErrorCheck, y=y, solution=solution)

    if error_count > 0:
        print("Errors: %d\n" % error_count)


if __name__ == "__main__":
    run()

