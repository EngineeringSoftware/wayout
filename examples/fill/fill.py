import pykokkos as pk

import sys

import kernels 
from kernels import *

@pk.workunit
def CheckValue(i: int, lsum: pk.Acc[pk.double], x: pk.View1D[pk.double]):
    lsum += x[i]
 

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)

    alpha = 1.5

    kernels.fill(x, alpha)

    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    sum_ = pk.parallel_reduce(p, CheckValue, x=x)

    expected = 1.0*N*1.5
    print("Sum: %lf Expected: %lf Diff: %e\n" % (sum_, expected, sum_ - expected))


if __name__ == "__main__":
    run()

