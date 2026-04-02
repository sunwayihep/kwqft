# KWQFT - **K**okkos **K**en **W**ilson **Q**uantum **F**ield **T**heory

KWQFT is a lattice gauge theory library based on the Kokkos programming model, inspired by the CUDA version [sunw](https://github.com/sunwayihep/sunw.git). Using the Kokkos programming model allows the same source code to run efficiently on both CPUs and GPUs without modification.

## Features

- **Performance Portability**: The same code compiles and runs on:
  - CPU (using Serial or OpenMP backend)
  - NVIDIA GPU (using CUDA backend)
  - AMD GPU (using HIP backend)
  - Intel GPU (using SYCL backend)

- **SU(N) Gauge Theory**: Supports SU(N) theory with arbitrary spacetime dimensions
- **Pseudo Heatbath Algorithm**: Efficient Monte Carlo updates
- **Overrelaxation**: Accelerated thermalization with overrelaxation updates
- **Physical Observables**: Plaquette and Polyakov loop measurements
- **Configuration I/O**: Binary format save and load

## Building

### Dependencies

- CMake >= 3.16
- C++17 compatible compiler
- Kokkos >= 4.0

### Obtaining Kokkos Source Code (Offline Environment)

If the server cannot access GitHub, you need to download the Kokkos source code on another machine first:

```bash
# Download on a machine with network access
git clone https://github.com/kokkos/kokkos.git
cd kokkos && git checkout 4.7.01

# Optional: Download KokkosKernels
git clone https://github.com/kokkos/kokkos-kernels.git
cd kokkos-kernels && git checkout 4.7.01

# Then copy the directories to the target server
```

### CPU Version (Serial)

```bash
mkdir build_cpu && cd build_cpu

# Using local Kokkos source code
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos

make -j
```

### CPU Version (OpenMP Multi-threaded)

```bash
mkdir build_omp && cd build_omp
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKWQFT_ENABLE_OPENMP=ON \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos
make -j
```

### NVIDIA GPU Version (CUDA)

```bash
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKWQFT_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos
make -j
```


The full list of `Kokkos_ARCH_*` flags for **CPU** (AMD, ARM, IBM, Intel, RISC-V) and **GPU** (NVIDIA, AMD, Intel, etc.) is maintained in the Kokkos documentation — see the **Architectures** section of the [Kokkos Configuration Guide](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html). Choose the flag that matches your hardware and pass e.g. `-DKokkos_ARCH_AMPERE86=ON` to CMake.

Common NVIDIA GPU examples (not exhaustive):

- `-DKokkos_ARCH_VOLTA70=ON` (V100)
- `-DKokkos_ARCH_TURING75=ON` (T4, RTX 20xx)
- `-DKokkos_ARCH_AMPERE80=ON` (A100)
- `-DKokkos_ARCH_AMPERE86=ON` (A40, RTX 30xx)
- `-DKokkos_ARCH_HOPPER90=ON` (H100)
- `-DKokkos_ARCH_BLACKWELL100=ON` (B100)
- `-DKokkos_ARCH_BLACKWELL120=ON` (RTX 50xx)

If you enable `KWQFT_ENABLE_CUDA=ON` but do not set any `Kokkos_ARCH_*` option, KWQFT’s CMake defaults to `Kokkos_ARCH_AMPERE80`; override using the guide above.

### AMD GPU Version (HIP)

```bash
mkdir build_hip && cd build_hip
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKWQFT_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos
make -j
```

### Custom SU(N) Parameters

```bash
cmake .. -DKWQFT_NCOLORS=4 -DKWQFT_NDIMS=4  # SU(4) in 4D
```

### CMake Options Summary

| Option | Description | Default |
|--------|-------------|---------|
| `KOKKOS_SOURCE_DIR` | Local Kokkos source directory | Empty (downloads from GitHub) |
| `KOKKOS_KERNELS_SOURCE_DIR` | Local KokkosKernels source directory (optional) | Empty |
| `KWQFT_ENABLE_OPENMP` | Enable OpenMP backend | OFF |
| `KWQFT_ENABLE_CUDA` | Enable CUDA backend | OFF |
| `KWQFT_ENABLE_HIP` | Enable HIP backend | OFF |
| `KWQFT_ENABLE_SYCL` | Enable SYCL backend | OFF |
| `KWQFT_USE_MPI` | Enable MPI build and `-geom` domain decomposition | OFF |
| `KWQFT_NCOLORS` | N value for SU(N) | 3 |
| `KWQFT_NDIMS` | Spacetime dimensions | 4 |

## Running

### Generating Gauge Field Configurations

```bash
# Format: ./heatbath L1 L2 L3 L4 beta ntraj [xi0]
./heatbath 8 8 8 16 6.0 1000
./heatbath 8 8 8 16 6.0 1000 2.0
```

Parameter description:
- `L1 L2 L3 L4`: Lattice dimensions (x, y, z, t)
- `beta`: Gauge coupling constant
- `ntraj`: Number of trajectories
- `xi0`: Bare anisotropy (optional, default `1.0`)

When `xi0 != 1`, the code uses anisotropic Wilson plaquette weights:
- spatial-spatial plaquettes: `beta / xi0`
- spatial-temporal plaquettes: `beta * xi0`

### MPI Support (`-geom`)

Build with MPI enabled:

```bash
mkdir build_mpi && cd build_mpi
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKWQFT_USE_MPI=ON \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos
make -j
```

MPI run format:

```bash
# mpirun -np P ./heatbath -geom p0 p1 ... p{NDIMS-1} L0 L1 ... L{NDIMS-1} beta ntraj [xi0]
```

Rules:
- `p0 * p1 * ... * p{NDIMS-1} == P`
- each global size `L[d]` must be divisible by `p[d]`
- local subdomain on each rank is `L[d] / p[d]`

Example (4D):

```bash
mpirun -np 8 ./heatbath -geom 2 2 2 1 4 4 4 8 6.0 10
```

This means global lattice `4x4x4x8`, process grid `2x2x2x1`, and each rank owns local lattice `2x2x2x8`.

### Hybrid MPI + OpenMP

Build OpenMP + MPI:

```bash
mkdir build_mpi_omp && cd build_mpi_omp
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DKWQFT_ENABLE_OPENMP=ON \
      -DKWQFT_USE_MPI=ON \
      -DKOKKOS_SOURCE_DIR=/path/to/kokkos
make -j
```

Recommended runtime binding:

```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

Use `PE=<threads-per-rank>` so each MPI rank gets enough CPU cores:

```bash
# 4 MPI ranks x 2 OpenMP threads = 8 cores
OMP_NUM_THREADS=2 mpirun -np 4 \
  --map-by slot:PE=2 --bind-to core --report-bindings \
  ./heatbath -geom 4 1 1 1 24 24 24 96 6.0 10 5.0

# 2 MPI ranks x 4 OpenMP threads = 8 cores
OMP_NUM_THREADS=4 mpirun -np 2 \
  --map-by slot:PE=4 --bind-to core --report-bindings \
  ./heatbath -geom 2 1 1 1 24 24 24 96 6.0 10 5.0
```

Notes:
- `--map-by slot:PE=n` asks Open MPI to reserve `n` processing elements per rank.
- `--bind-to core` binds each rank to cores; OpenMP threads then run inside that core set.
- `--report-bindings` prints actual CPU binding and is useful for checking whether threads are spread as expected.

### Running Tests

```bash
./test_kwqft
```

## Code Structure

```
kokkos_src/
├── CMakeLists.txt          # CMake build configuration
├── include/
│   ├── kwqft.hpp           # Main header file
│   ├── kwqft_common.hpp    # Common definitions and Kokkos type aliases
│   ├── complex.hpp         # Complex number class
│   ├── msu2.hpp            # SU(2) subgroup representation
│   ├── matrixsun.hpp       # SU(N) matrix class
│   ├── constants.hpp       # Lattice parameters
│   ├── index.hpp           # Lattice indexing functions
│   ├── gauge_array.hpp     # Gauge field array container
│   ├── random.hpp          # Random number generation
│   ├── monte.hpp           # Monte Carlo algorithms
│   └── measurements.hpp    # Physical measurements
├── src/
│   ├── constants.cpp       # Constants implementation
│   ├── gauge_array.cpp     # Gauge field implementation
│   ├── random.cpp          # Random number implementation
│   ├── monte.cpp           # Monte Carlo implementation
│   ├── plaquette.cpp       # Plaquette measurement
│   ├── polyakov.cpp        # Polyakov loop measurement
│   ├── reunitarize.cpp     # Reunitarization
│   ├── io_gauge.cpp        # Configuration I/O
│   └── heatbath_main.cpp   # Main program
└── test/
    └── test_main.cpp       # Test program
```

## Performance Portability Notes

This code achieves performance portability using the Kokkos programming model:

1. **Execution Space Abstraction**: Uses `Kokkos::parallel_for` and `Kokkos::parallel_reduce` instead of CUDA kernels
2. **Memory Space Abstraction**: Uses `Kokkos::View` instead of explicit CUDA memory management
3. **Atomic Operations**: Uses `Kokkos::atomic_*` functions
4. **Random Number Generation**: Uses `Kokkos::Random_XorShift64_Pool`

## References

- [Kokkos Documentation](https://kokkos.org/kokkos-core-wiki/)
- [Kokkos Tutorials](https://github.com/kokkos/kokkos-tutorials)
