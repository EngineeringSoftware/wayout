import pykokkos as pk

import sys

import kernels 
from kernels import *

def run() -> None: 
    x = View("int **")("test", 10, 20)

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    run()

