# gen\_ex.py
gen\_ex.py takes an example name as an argument and calls the tool to generate wrapper classes

ex: gen\_ex.py cgsolve

# kokkos kernels
To build kokkos kernels, create a build/install directory in ~/Kokkos and in build/ run

```
cmake ../kokkos-kernels -DCMAKE_CXX_COMPILER=<gxx> -DCMAKE_INSTALL_PREFIX=<prefix> -DCMAKE_CXX_FLAGS="-fPIC" -DKokkos_DIR=<kokkos_install>/lib/cmake/Kokkos -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF
make -j install
```
* disable cublas and cusparse because it has link issue on seoul for some reason
* libclang==11.0.1

The following environment variables need to be set to point to the path
to the Kokkos installation directories:
PK_KOKKOS_KERNELS_LIB_PATH_CUDA: this is the path to the lib/ directory in your Kokkos Kernels CUDA install
PK_KOKKOS_KERNELS_INCLUDE_PATH_CUDA: this is the path to the include/ directory in your Kokkos Kernels CUDA install
PK_KOKKOS_KERNELS_LIB_PATH_OMP: same as above for openmp
PK_KOKKOS_KERNELS_INCLUDE_PATH_OMP: same as above for openmp


