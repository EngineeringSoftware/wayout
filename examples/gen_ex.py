""" Usage: call with <exp_name>
"""

import sys
import os
import platform

import pykokkos as pk
import wayout.static as wayout 

machine: str = platform.node()

if machine == "xps2" or machine == "seoul":
    # xps2
    LIB_PATH="/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1"
    PROJECT_PATH="/home/jzhu2/Kokkos/"
else:
    # home
    LIB_PATH="/usr/lib/libclang.so"
    PROJECT_PATH="/home/steven/projects/"


KOKKOS_KERNEL_PATH=PROJECT_PATH+"kokkos-kernels/"
INCLUDE_PATH=[PROJECT_PATH+"kokkos/core/", PROJECT_PATH+"kokkos/core/src/", KOKKOS_KERNEL_PATH+"src/", KOKKOS_KERNEL_PATH+"src/blas/impl/"]

THRUST_PATH="/home/jzhu2/thrust/"
THRUST_INCLUDE_PATH=[THRUST_PATH]
THRUST_ITERATOR_PATHS = [
    "thrust/iterator/iterator_facade.h", 
    "thrust/iterator/detail/iterator_adaptor_base.h", 
    "thrust/iterator/iterator_adaptor.h", 
    "thrust/iterator/detail/normal_iterator.h",
]
        

ABS_PATH="src/blas/KokkosBlas1_abs.hpp"
AXPBY_PATH="src/blas/KokkosBlas1_axpby.hpp"
DOT_PATH="src/blas/KokkosBlas1_dot.hpp"
FILL_PATH="src/blas/KokkosBlas1_fill.hpp"
IAMAX_PATH="src/blas/KokkosBlas1_iamax.hpp"
MULT_PATH="src/blas/KokkosBlas1_mult.hpp"
NRM1_PATH="src/blas/KokkosBlas1_nrm1.hpp"
NRM2_PATH="src/blas/KokkosBlas1_nrm2.hpp"
NRM2_SQUARED_PATH="src/blas/KokkosBlas1_nrm2_squared.hpp"
NRMINF_PATH="src/blas/KokkosBlas1_nrminf.hpp"
RECIPROCAL_PATH="src/blas/KokkosBlas1_reciprocal.hpp"
SCAL_PATH="src/blas/KokkosBlas1_scal.hpp"
SUM_PATH="src/blas/KokkosBlas1_sum.hpp"
UPDATE_PATH="src/blas/KokkosBlas1_update.hpp"
GEMV_PATH="src/blas/KokkosBlas2_gemv.hpp"
GEMM_PATH="src/blas/KokkosBlas3_gemm.hpp"
TRSM_PATH="src/blas/KokkosBlas3_trsm.hpp"
TRMM_PATH="src/blas/KokkosBlas3_trmm.hpp"
GESV_PATH="src/blas/KokkosBlas_gesv.hpp"
TRTRI_PATH="src/blas/KokkosBlas_trtri.hpp"

SPMV_PATH="src/sparse/KokkosSparse_spmv.hpp"
CRS_PATH="src/sparse/KokkosSparse_CrsMatrix.hpp"
SPGEMM_PATH="src/sparse/KokkosSparse_spgemm.hpp"
SPGEMM_HANDLE_PATH="src/sparse/KokkosSparse_spgemm_handle.hpp"
GAUSS_PATH="src/sparse/KokkosSparse_gauss_seidel.hpp"
GS_HANDLE_PATH="src/sparse/KokkosSparse_gauss_seidel_handle.hpp"
SPILUK_PATH="src/sparse/KokkosSparse_spiluk.hpp"
SPILUK_HANDLE_PATH="src/sparse/KokkosSparse_spiluk_handle.hpp"
SPTRSV_PATH="src/sparse/KokkosSparse_sptrsv.hpp"
SPTRSV_HANDLE_PATH="src/sparse/KokkosSparse_sptrsv_handle.hpp"
TRSV_PATH="src/sparse/KokkosSparse_trsv.hpp"
SPADD_PATH="src/sparse/KokkosSparse_spadd.hpp"
SPADD_HANDLE_PATH="src/sparse/KokkosSparse_spadd_handle.hpp"

COLOR1_PATH="src/graph/KokkosGraph_Distance1Color.hpp"
COLOR1_HANDLE_PATH="src/graph/KokkosGraph_Distance1ColorHandle.hpp"
COLOR2_PATH="src/graph/KokkosGraph_Distance2Color.hpp"
COLOR2_HANDLE_PATH="src/graph/KokkosGraph_Distance2ColorHandle.hpp"

DEFAULT_TYPES_PATH="src/common/KokkosKernels_default_types.hpp"
HANDLE_PATH="src/common/KokkosKernels_Handle.hpp"
IO_UTILS_PATH="src/common/KokkosKernels_IOUtils.hpp"
SPARSE_UTILS_PATH="src/sparse/common/KokkosKernels_SparseUtils.hpp"

STRUCT_MAT_PATH="test_common/KokkosKernels_Test_Structured_Matrix.hpp"

CORE_PATH=PROJECT_PATH+"kokkos/core/src/Kokkos_Core.hpp"
VIEW_PATH=PROJECT_PATH+"kokkos/core/src/Kokkos_View.hpp"
COPY_VIEW_PATH=PROJECT_PATH+"kokkos/core/src/Kokkos_CopyViews.hpp"
CRS_GRAPH_PATH=PROJECT_PATH+"kokkos/containers/src/Kokkos_StaticCrsGraph.hpp"

RANDOM_PATH=PROJECT_PATH+"kokkos/algorithms/src/Kokkos_Random.hpp"

# emulate user interaction
_examples = {
    "cgsolve": [KOKKOS_KERNEL_PATH + path for path in [AXPBY_PATH, DOT_PATH, SPMV_PATH, CRS_PATH]],
    "inner": [KOKKOS_KERNEL_PATH + path for path in [DOT_PATH, GEMV_PATH]], 
    "spgemm":[KOKKOS_KERNEL_PATH + path for path in [CRS_PATH, SPGEMM_PATH, HANDLE_PATH, SPGEMM_HANDLE_PATH]], 
    "gauss": [KOKKOS_KERNEL_PATH + path for path in [HANDLE_PATH, IO_UTILS_PATH, SPMV_PATH, CRS_PATH, GAUSS_PATH, NRM2_PATH, GS_HANDLE_PATH]] + \
            [CRS_GRAPH_PATH], 
    "color": [KOKKOS_KERNEL_PATH + path for path in [DEFAULT_TYPES_PATH, HANDLE_PATH, COLOR1_PATH, COLOR1_HANDLE_PATH,
            COLOR2_PATH, COLOR2_HANDLE_PATH]],
    "sparse_iluk": [KOKKOS_KERNEL_PATH + path for path in [NRM2_PATH, CRS_PATH, SPMV_PATH, SPILUK_PATH, IO_UTILS_PATH, \
            STRUCT_MAT_PATH, HANDLE_PATH, SPILUK_HANDLE_PATH]] + \
            [CRS_GRAPH_PATH],
    "cg_iluk": [KOKKOS_KERNEL_PATH + path for path in [DOT_PATH, AXPBY_PATH, CRS_PATH, SPMV_PATH, SPILUK_PATH, SPTRSV_PATH, IO_UTILS_PATH, \
            STRUCT_MAT_PATH, HANDLE_PATH, SPILUK_HANDLE_PATH, SPTRSV_HANDLE_PATH]] + \
            [CRS_GRAPH_PATH],
    # trivial exp
    "abs": [KOKKOS_KERNEL_PATH + path for path in [ABS_PATH]], 
    "axpy": [KOKKOS_KERNEL_PATH + path for path in [AXPBY_PATH]],
    "axpby": [KOKKOS_KERNEL_PATH + path for path in [AXPBY_PATH]],
    "dot": [KOKKOS_KERNEL_PATH + path for path in [DOT_PATH]],
    "fill": [KOKKOS_KERNEL_PATH + path for path in [FILL_PATH]],
    "iamax": [KOKKOS_KERNEL_PATH + path for path in [IAMAX_PATH]] + [RANDOM_PATH],
    "mult": [KOKKOS_KERNEL_PATH + path for path in [MULT_PATH]],
    "nrm1": [KOKKOS_KERNEL_PATH + path for path in [NRM1_PATH]],
    "nrm2": [KOKKOS_KERNEL_PATH + path for path in [NRM2_PATH]],
    "nrm2_squared": [KOKKOS_KERNEL_PATH + path for path in [NRM2_SQUARED_PATH]],
    "nrminf": [KOKKOS_KERNEL_PATH + path for path in [NRMINF_PATH]],
    "reciprocal": [KOKKOS_KERNEL_PATH + path for path in [RECIPROCAL_PATH]],
    "scal": [KOKKOS_KERNEL_PATH + path for path in [SCAL_PATH]],
    "sum": [KOKKOS_KERNEL_PATH + path for path in [SUM_PATH]] + [RANDOM_PATH],
    "update": [KOKKOS_KERNEL_PATH + path for path in [UPDATE_PATH]],
    "gemv": [KOKKOS_KERNEL_PATH + path for path in [GEMV_PATH]],
    "gemm": [KOKKOS_KERNEL_PATH + path for path in [GEMM_PATH]],
    "trsm": [KOKKOS_KERNEL_PATH + path for path in [TRSM_PATH]] + [RANDOM_PATH],
    "trmm": [KOKKOS_KERNEL_PATH + path for path in [TRMM_PATH]] + [RANDOM_PATH],
    "gesv": [KOKKOS_KERNEL_PATH + path for path in [GESV_PATH]] + [RANDOM_PATH],
    "trtri": [KOKKOS_KERNEL_PATH + path for path in [TRTRI_PATH]] + [RANDOM_PATH],
    "spmv_struct": [KOKKOS_KERNEL_PATH + path for path in [SPMV_PATH, CRS_PATH]],
    "trsv": [KOKKOS_KERNEL_PATH + path for path in [TRSV_PATH, CRS_PATH]],
    "sptrsv": [KOKKOS_KERNEL_PATH + path for path in [SPTRSV_PATH, CRS_PATH, IO_UTILS_PATH, HANDLE_PATH, SPTRSV_HANDLE_PATH]],
    "spadd": [KOKKOS_KERNEL_PATH + path for path in [DEFAULT_TYPES_PATH, SPADD_PATH, STRUCT_MAT_PATH, CRS_PATH, HANDLE_PATH, SPADD_HANDLE_PATH]] + 
            [CRS_GRAPH_PATH],
    "view_test": [],

    # thrust
    "thrust_sort": \
        [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/generate.h", "thrust/sort.h", "thrust/copy.h"]],
    #     [THRUST_PATH + path for path in ["thrust/host_vector.h", "thrust/device_vector.h", "thrust/generate.h", "thrust/sort.h", "thrust/copy.h"]],
    "thrust_sum": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/generate.h", "thrust/reduce.h", "thrust/functional.h"]],
    "thrust_saxpy": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/transform.h", "thrust/functional.h",
        "thrust/fill.h", "thrust/copy.h"]],
    "thrust_sparse": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/functional.h", "thrust/merge.h",
        "thrust/reduce.h", "thrust/inner_product.h", "thrust/pair.h", "thrust/copy.h"]],
    "thrust_mode": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/sort.h", 
        "thrust/pair.h", "thrust/reduce.h",
        "thrust/inner_product.h", "thrust/extrema.h", "thrust/functional.h", "thrust/copy.h", 
        "thrust/iterator/detail/zip_iterator_base.h", "thrust/iterator/zip_iterator.h", 
        "thrust/iterator/detail/constant_iterator_base.h", "thrust/iterator/constant_iterator.h", 
        "thrust/random/linear_congruential_engine.h", "thrust/random/uniform_int_distribution.h", "thrust/random.h"]],
    "thrust_set": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/merge.h", "thrust/set_operations.h",
        "thrust/iterator/detail/discard_iterator_base.h", "thrust/iterator/discard_iterator.h", "thrust/copy.h"]],
    "thrust_histogram": [THRUST_PATH + path for path in THRUST_ITERATOR_PATHS + \
        ["thrust/detail/vector_base.h", "thrust/host_vector.h", "thrust/device_vector.h", "thrust/sort.h", "thrust/copy.h", 
        "thrust/random/linear_congruential_engine.h", "thrust/random/uniform_int_distribution.h", "thrust/random.h",
        "thrust/functional.h", "thrust/reduce.h", "thrust/pair.h", "thrust/inner_product.h", "thrust/binary_search.h", "thrust/adjacent_difference.h",
        "thrust/iterator/detail/constant_iterator_base.h", "thrust/iterator/constant_iterator.h",
        "thrust/iterator/detail/counting_iterator.inl", "thrust/iterator/counting_iterator.h"]],

}
# return static build time
def driver(exp_name, clean, target=wayout.Target.kokkos_omp):
    paths = []

    if exp_name in _examples:
        paths.extend(_examples[exp_name])
    else:
        exit(f"no matching example in {_examples.keys()}")

    exp_dir = exp_name + "/"
    build_dir = exp_dir + "/build/"
    # clean up build directory
    if clean:
        os.system(f"rm -rf {exp_dir}/kernels.py")
        os.system(f"rm -rf {build_dir}")

    timer = pk.Timer()
    if exp_name.startswith("thrust_"):
        if target == wayout.Target.kokkos_omp:
            target = wayout.Target.thrust_omp
        wayout.generate_wrapper(exp_dir, paths, ["-I"+p for p in THRUST_INCLUDE_PATH] + ["-xc++"], target)
    else:
        paths.append(CORE_PATH)
        paths.append(VIEW_PATH)
        paths.append(COPY_VIEW_PATH)

        if exp_name == "view_test":
            paths = [CORE_PATH]

        wayout.generate_wrapper(exp_dir, paths, ["-I"+p for p in INCLUDE_PATH]+["-std=c++17"], target)
    return timer.seconds()


# initial param tune to take around ~100 sec
kokkos_benchmarks = {
    "cgsolve": [2**15, 1],
    "inner": [2**20, 2**10],
    "spgemm": [2**26, 1], 
    "gauss": [2**24, 1],
    "color": [2**15, 2**10],
    "sparse_iluk": [2**11, 2**10],
    "cg_iluk": [2**10, 2**10]
}

iter = 100
thrust_benchmarks = {
    "thrust_sort": [2**27, iter],
    "thrust_sum": [2**30, iter],
    "thrust_saxpy": [2**28, iter], 
    "thrust_sparse": [2**25, iter],
    "thrust_mode": [2**26, iter],
    "thrust_set": [2**26, iter],
    "thrust_histogram": [2**26, iter]
}

import re
import subprocess
num_trials = 6
# builds both omp and cuda version
def build_kokkos_benchmarks():
    for bench in kokkos_benchmarks:
        print(f"\tbuilding kokkos {bench} benchmark...")

        build_dir = f"kokkos/{bench}/" 
        # OpenMP version
        os.system(f"cp kokkos/Makefile.omp {build_dir}/Makefile.inc")
        os.system(f"make clean -s -C {build_dir}")
        os.system(f"make -s -C {build_dir}")

        # Cuda version
        os.system(f"cp kokkos/Makefile.cuda {build_dir}/Makefile.inc")
        os.system(f"make clean -s -C {build_dir}")
        os.system(f"make -s -C {build_dir}")


def run_kokkos_benchmarks(outfile):
    abs_path = os.path.abspath(outfile)

    for bench in kokkos_benchmarks:
        for i in range(num_trials-1, -1, -1):
            param = kokkos_benchmarks[bench]
            N = int(param[0] / (2**i))
            print(f"\trunning kokkos {bench} benchmark (N={N})...")

            build_dir = f"kokkos/{bench}/" 
            # OpenMP
            command = [f'./{bench}.host', f"{N}", f"{param[1]}", abs_path]
            subprocess.run(command, cwd=build_dir+"/", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            # subprocess.run(command, cwd=build_dir+"/", check=True)

            # Cuda 
            command = [f'./{bench}.cuda', f"{N}", f"{param[1]}", abs_path]
            subprocess.run(command, cwd=build_dir+"/", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            # subprocess.run(command, cwd=build_dir+"/", check=True)


def build_thrust_benchmarks():
    for bench in thrust_benchmarks:
        print(f"\tbuilding thrust {bench} benchmark...")

        build_dir = f"thrust/{bench}/" 
        # OpenMP version
        os.system(f"cp thrust/Makefile.omp {build_dir}/Makefile.inc")
        os.system(f"make clean -s -C {build_dir}")
        os.system(f"make -s -C {build_dir}")

        # Cuda version
        os.system(f"cp thrust/Makefile.cuda {build_dir}/Makefile.inc")
        os.system(f"make clean -s -C {build_dir}")
        os.system(f"make -s -C {build_dir}")


def run_thrust_benchmarks(outfile):
    abs_path = os.path.abspath(outfile)

    for bench in thrust_benchmarks:
        for i in range(num_trials-1, -1, -1):
            param = thrust_benchmarks[bench]
            N = int(param[0] / (2**i))
            print(f"\trunning thrust {bench} benchmark (N={N})...")

            build_dir = f"thrust/{bench}/" 
            # OpenMP
            command = [f'./{bench}.host', f"{N}", f"{param[1]}", abs_path]
            subprocess.run(command, cwd=build_dir+"/", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            # subprocess.run(command, cwd=build_dir+"/", check=True)

            # Cuda 
            command = [f'./{bench}.cuda', f"{N}", f"{param[1]}", abs_path]
            subprocess.run(command, cwd=build_dir+"/", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            # subprocess.run(command, cwd=build_dir+"/", check=True)       


def build_benchmarks(use_cuda, build_info, benchmarks):
    for bench in benchmarks:
        print(f"\tbuilding {bench} benchmark...")
        bench_dir = bench+ "/"
        build_dir = bench_dir + "build/"
        command = ['python', f'{bench}.py']
        if use_cuda:
            command.append("--cuda")
        subprocess.run(['make', 'clean', '-C', build_dir], stdout=subprocess.DEVNULL)
        res = subprocess.run(command, cwd=bench_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        res_str = res.stdout.decode('utf-8')

        t= re.search(r"dynamic_compile_time=\[(.*)\]", res_str).group(1)

        if use_cuda:
            build_info[bench]['nvcc_time'] = t
        else:
            build_info[bench]['gcc_time'] = t
            build_info[bench]['num_ctors'] = re.search(r"num_ctors=\[(.*)\]", res_str).group(1)
            build_info[bench]['num_kernels'] = re.search(r"num_kernels=\[(.*)\]", res_str).group(1)
            build_info[bench]['num_mod'] = os.popen(f'ls -l {build_dir}/f_*.so | wc -l').read().strip()


def run_benchmarks(outfile, use_cuda, benchmarks):
    abs_path = os.path.abspath(outfile)
    file = f"--file={abs_path}"

    for bench in benchmarks:
        for i in range(num_trials-1, -1, -1):
            param = benchmarks[bench]
            N = int(param[0] / (2**i))
            print(f"\trunning {bench} benchmark (N={N})...")
            command = ['python', f'{bench}.py', f"-N={N}", f"-M={param[1]}", file]
            if use_cuda:
                command.append("--cuda")
            subprocess.run(command, cwd=bench+"/", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            # subprocess.run(command, cwd=bench+"/")


def benchmark(outfile, kokkos_outfile, build_info_outfile, build_only = False):
        if not build_only:
            header = "workload, cuda, size1, size2, init_time, kernel_time"
            os.system(f"rm -f {outfile}")
            os.system(f"echo {header} > {outfile}")
            os.system(f"rm -f {kokkos_outfile}")
            os.system(f"echo {header} > {kokkos_outfile}")

        # build and run kokkos
        if not build_only:
            print("Kokkos:")
            build_kokkos_benchmarks()
            run_kokkos_benchmarks(kokkos_outfile)

        # clean build
        build_info = {}
        print("cleaning and running static...")
        for bench in kokkos_benchmarks:
            t = driver(bench, False, wayout.Target.kokkos_omp)
            build_info[bench] = {"workload": bench}
            build_info[bench]['static_time'] = t
                
        # OpenMP
        print("OpenMP:")
        build_benchmarks(False, build_info, kokkos_benchmarks)
        if not build_only:
            run_benchmarks(outfile, False, kokkos_benchmarks)

        # update makefile
        for bench in kokkos_benchmarks:
            t = driver(bench, True, wayout.Target.kokkos_cuda)

        # Cuda
        print("Cuda:")
        build_benchmarks(True, build_info, kokkos_benchmarks)
        if not build_only:
            run_benchmarks(outfile, True, kokkos_benchmarks)

        if build_only:
            return

        build_info_header = "workload, static_time, gcc_time, nvcc_time, num_mod, num_ctors, num_kernels"
        with open(build_info_outfile, "w") as f:
            f.write(build_info_header + "\n")
            for bench in kokkos_benchmarks:
                info = build_info[bench]
                f.write(f"{bench},{info['static_time']},{info['gcc_time']},{info['nvcc_time']},"
                    f"{info['num_mod']},{info['num_ctors']},{info['num_kernels']}\n")


def thrust_benchmark(outfile, thrust_outfile, build_info_outfile, build_only = False):
        if not build_only:
            header = "workload, cuda, size1, size2, init_time, kernel_time"
            os.system(f"rm -f {outfile}")
            os.system(f"echo {header} > {outfile}")
            os.system(f"rm -f {thrust_outfile}")
            os.system(f"echo {header} > {thrust_outfile}")

        # build and run thrust
        if not build_only:
            print("Thrust:")
            build_thrust_benchmarks()
            run_thrust_benchmarks(thrust_outfile)

        # clean build
        build_info = {}
        print("cleaning and running static...")
        for bench in thrust_benchmarks:
            t = driver(bench, False, wayout.Target.thrust_omp)
            build_info[bench] = {"workload": bench}
            build_info[bench]['static_time'] = t
                
        # OpenMP
        print("OpenMP:")
        build_benchmarks(False, build_info, thrust_benchmarks)
        if not build_only:
            run_benchmarks(outfile, False, thrust_benchmarks)

        # update makefile
        for bench in thrust_benchmarks:
            t = driver(bench, True, wayout.Target.thrust_cuda)

        # Cuda
        print("Cuda:")
        build_benchmarks(True, build_info, thrust_benchmarks)
        if not build_only:
            run_benchmarks(outfile, True, thrust_benchmarks)

        if build_only:
            return

        build_info_header = "workload, static_time, gcc_time, nvcc_time, num_mod, num_ctors, num_kernels"
        with open(build_info_outfile, "w") as f:
            f.write(build_info_header + "\n")
            for bench in thrust_benchmarks:
                info = build_info[bench]
                f.write(f"{bench},{info['static_time']},{info['gcc_time']},{info['nvcc_time']},"
                    f"{info['num_mod']},{info['num_ctors']},{info['num_kernels']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(f"provide name of example to translate ({_examples.keys()}) or 'all' to run all examples")

    exp_name = sys.argv[1]
    clean = False
    if len(sys.argv) > 2 and sys.argv[2] == "clean":
        clean = True

    if exp_name == "bench":
        outfile = 'data/bench_res.csv'
        kokkos_outfile = 'data/kokkos_bench_res.csv'
        build_info_outfile = 'data/build_info.csv'

        # 3 runs 
        num_runs = 3
        for i in range(num_runs):
            print(f"==================== Run {i} ====================")
            i_ = f".{i}"
            benchmark(outfile+i_, kokkos_outfile+i_, build_info_outfile+i_)
    elif exp_name == "build":
        benchmark(None, None, None, build_only = True)

    elif exp_name == "thrust_bench":
        outfile = 'thrust_data/bench_res.csv'
        thrust_outfile = 'thrust_data/thrust_bench_res.csv'
        build_info_outfile = 'thrust_data/build_info.csv'

        # 3 runs 
        num_runs = 3
        for i in range(num_runs):
            print(f"==================== Run {i} ====================")
            i_ = f".{i}"
            thrust_benchmark(outfile+i_, thrust_outfile+i_, build_info_outfile+i_)
    elif exp_name == "thrust_build":
        thrust_benchmark(None, None, None, build_only = True)

    elif exp_name == "all":
        for exp in _examples:
            driver(exp, True)

    else:
        driver(exp_name, clean)

