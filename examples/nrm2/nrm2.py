import pykokkos as pk

import math
import sys

import kernels 
from kernels import *

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    deep_copy(x, 3.)

    x_nrm = kernels.nrm2(x)

    expected = 1.0*math.sqrt(N*3.*3.)
    print("Sum: %lf Expected: %lf Diff: %e\n" % (x_nrm, expected, x_nrm - expected))


if __name__ == "__main__":
    run()

