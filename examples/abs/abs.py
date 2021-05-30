import pykokkos as pk

import sys

import kernel as k

@pk.workunit
def CheckValue(i: int, lsum: pk.Acc[pk.double], y: pk.View1D[pk.double]):
    lsum += y[i]
 

def run() -> None: 
    N = int(sys.argv[1])

    x = pk.View([N], dtype=pk.double)
    y = pk.View([N], dtype=pk.double)

    k.Kokkos.deep_copy(x, -1)
    k.KokkosBlas.abs(y, x)

    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    sum_ = pk.parallel_reduce(p, CheckValue, y=y)

    print("Sum: %lf Expected: %lf Diff: %e\n" % (sum_, 1.0*N, sum_-1.0*N))


if __name__ == "__main__":
    run()

