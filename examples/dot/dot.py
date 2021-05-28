import pykokkos as pk

import sys

import kernels 
from kernels import *

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)

    deep_copy(x, 3.)
    deep_copy(y, 2.)

    x_y = kernels.dot(x, y)

    expected = 1.0*N*(3*2)
    print("Sum: %lf Expected: %lf Diff: %e\n" % (x_y, expected, x_y - expected))


if __name__ == "__main__":
    run()

