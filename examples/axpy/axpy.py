import pykokkos as pk

import sys

import kernels 
from kernels import *

@pk.workunit
def CheckValue(i: int, lsum: pk.Acc[pk.double], y: pk.View1D[pk.double]):
    lsum += y[i]
 

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)

    deep_copy(x, 3.)
    deep_copy(y, 2.)

    alpha = 1.5

    kernels.axpy(alpha, x, y)

    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    sum_ = pk.parallel_reduce(p, CheckValue, y=y)

    expected = 1.0*N*(2.+1.5*3)
    print("Sum: %lf Expected: %lf Diff: %e\n" % (sum_, expected, sum_ - expected))


if __name__ == "__main__":
    run()

