import pykokkos as pk

import math
import sys

import kernels 
from kernels import *

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    deep_copy(x, 3.)
    # x_5 = x[5:6]
    # deep_copy(x_5, -7.5)
    x[5] = -7.5

    x_nrm = kernels.nrminf(x)

    expected = 7.5
    print("Sum: %lf Expected: %lf Diff: %e\n" % (x_nrm, expected, x_nrm - expected))


if __name__ == "__main__":
    run()

