#!/bin/bash
# KWQFT - Kokkos Ken Wilson Quantum Field Theory
# Build script for GPU (CUDA backend, e.g. for Nvidia A100)

# setting up compiler environments
# e.g. module load lqcd/gpu/mpi/openmpi/5.0.10-ucx-cuda12.4-gcc11
NCOLORS=3
NDIMS=4
DIR=build_cuda_nc${NCOLORS}_nd${NDIMS}

rm -rf ${DIR}
mkdir -p ${DIR} && cd ${DIR}

cmake .. -DKWQFT_NCOLORS=${NCOLORS} -DKWQFT_NDIMS=${NDIMS} \
	-DCMAKE_BUILD_TYPE=Release \
	-DKWQFT_ENABLE_CUDA=ON \
	-DKWQFT_USE_MPI=ON \
	-DKokkos_ARCH_AMPERE80=ON \
	-DKOKKOS_SOURCE_DIR="../../kokkos/kokkos" # path to Kokkos source directory

make -j
