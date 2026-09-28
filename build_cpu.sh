#!/bin/bash
# KWQFT - Kokkos Ken Wilson Quantum Field Theory
# Build script for CPU (MPI + OpenMP backend)

# setting up compiler environments
# e.g. module load lqcd/gpu/mpi/openmpi/5.0.10-ucx-cuda12.4-gcc11
NCOLORS=4
NDIMS=4
DIR=build_omp_nc${NCOLORS}_nd${NDIMS}

rm -rf ${DIR}
mkdir -p ${DIR} && cd ${DIR}

cmake .. -DKWQFT_NCOLORS=${NCOLORS} -DKWQFT_NDIMS=${NDIMS} \
	-DCMAKE_BUILD_TYPE=Release \
	-DKWQFT_ENABLE_OPENMP=ON \
	-DKWQFT_USE_MPI=ON \
	-DKokkos_ARCH_NATIVE=ON \
	-DKWQFT_ENABLE_HOST_SIMD=ON \
	-DKOKKOS_SOURCE_DIR="../../kokkos/kokkos" # path to Kokkos source directory

make -j
