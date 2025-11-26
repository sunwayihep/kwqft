#!/bin/bash
# KWQFT - Kokkos Ken Wilson Quantum Field Theory
# Build script for GPU (CUDA backend)

module load lqcd/gpu/cuda/11.8

NCOLORS=3
NDIMS=4
DIR=build_cuda_nc${NCOLORS}_nd${NDIMS}

rm -rf ${DIR}
mkdir -p ${DIR} && cd ${DIR}

cmake .. -DKWQFT_NCOLORS=${NCOLORS} -DKWQFT_NDIMS=${NDIMS} \
	-DCMAKE_BUILD_TYPE=Release -DKWQFT_ENABLE_CUDA=ON \
	-DKokkos_ARCH_AMPERE80=ON \
	-DKOKKOS_SOURCE_DIR="../kokkos"

make -j
