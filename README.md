# nvortexKokkos
Simple direct n-body solvers using Kokkos for parallelization and performance

## Compile and run

    git clone --recursive https://github.com/markstock/nvortexKokkos.git
    mkdir build
    cd build
    ccmake ..
    make
    ./nvKok01.bin -n=100000

## Description

The first code, `nvKok01.cpp` is designed to be the minimal n-body code for Kokkos. When compiler with the OPENMP driver, the performance should be similar to the basic "host" OpenMP operation.

Future versions will be designed to make calls to accelerators (via HIP or CUDA backends).


