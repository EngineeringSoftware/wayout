import pykokkos as pk

import argparse
import copy
import math
import random
import sys

import kernels 
from kernels import *

Scalar = "default_scalar"
Ordinal = "default_lno_t"
Offset = "default_size_type"
Layout = "default_layout"

def run() -> None: 

    execution_space = "Kokkos::DefaultExecutionSpace"
    memory_space = "Kokkos::DefaultExecutionSpace::memory_space"
    matrix_type = CrsMatrix(Scalar, Ordinal, execution_space, None, Offset)

    return_value = 0

    mat_structure = pk.View([2, 3], dtype=int)
    mat_structure[0][0] = 10
    mat_structure[0][1] = 1
    mat_structure[1][2] = 1
    mat_structure[1][0] = 10
    mat_structure[1][1] = 1
    mat_structure[1][2] = 1

    A = generate_structured_matrix2D("FD", mat_structure, template_args=[matrix_type])
    B = generate_structured_matrix2D("FE", mat_structure, template_args=[matrix_type])
    C = matrix_type()

    KernelHandle = KokkosKernelsHandle(Offset, Ordinal, Scalar, execution_space,
            memory_space, memory_space)

    kh = KernelHandle()
    kh.create_spadd_handle(False)

    alpha = 2.5
    beta = 1.2
    
    spadd_symbolic(ptr(kh), A, B, C, namespace="KokkosSparse")
    spadd_numeric(ptr(kh), alpha, A, beta, B, C, namespace="KokkosSparse")
    kh.destroy_spadd_handle()

    print("spadd was performed correctly!")


if __name__ == "__main__":
    run()
