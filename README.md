# nvortexKokkos
Simple direct n-body solvers using [Kokkos](https://github.com/kokkos/kokkos) for parallelization and performance

## Compile and run

    git clone --recursive https://github.com/markstock/nvortexKokkos.git
    mkdir build
    cd build
    ccmake ..
    make
    ./nvKok01.bin -n=100000

## Description

The first code, `nvKok01.cpp` is designed to be the minimal n-body code for Kokkos. When compiled with the OPENMP driver, the performance should be similar to the basic "host" OpenMP operation.

Future versions will be designed to make calls to accelerators (via HIP or CUDA backends).

## Other notes

Launch a CPU-only batch job with Slurm on a Bard Peak (64-core Epyc) with these commands:

    OMP_PROC_BIND=spread OMP_PLACES=threads srun -N1 -n1 -pbardpeak --exclusive --threads-per-core=1 --cpus-per-task=64 -t0:00:30 ./nvKok01.bin -n=300000
    OMP_PROC_BIND=spread OMP_PLACES=threads srun -N1 -n1 -pbardpeak --exclusive --threads-per-core=2 --cpus-per-task=128 -t0:00:30 ./nvKok01.bin -n=300000

