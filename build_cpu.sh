#!/bin/bash
# KWQFT - Kokkos Ken Wilson Quantum Field Theory
# Build script for CPU (OpenMP backend)

NCOLORS=3
NDIMS=4
DIR=build_omp_nc${NCOLORS}_nd${NDIMS}

rm -rf ${DIR}
mkdir -p ${DIR} && cd ${DIR}

cmake .. -DKWQFT_NCOLORS=${NCOLORS} -DKWQFT_NDIMS=${NDIMS} \
	-DCMAKE_BUILD_TYPE=Release -DKWQFT_ENABLE_OPENMP=ON \
	-DKOKKOS_SOURCE_DIR="../kokkos"

make -j
