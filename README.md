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

Supported GPU architecture options:
- `-DKokkos_ARCH_VOLTA70=ON` (V100)
- `-DKokkos_ARCH_TURING75=ON` (T4, RTX 20xx)
- `-DKokkos_ARCH_AMPERE80=ON` (A100)
- `-DKokkos_ARCH_AMPERE86=ON` (RTX 30xx)
- `-DKokkos_ARCH_HOPPER90=ON` (H100)

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
| `KWQFT_NCOLORS` | N value for SU(N) | 3 |
| `KWQFT_NDIMS` | Spacetime dimensions | 4 |

## Running

### Generating Gauge Field Configurations

```bash
# Format: ./heatbath L1 L2 L3 L4 beta ntraj
./heatbath 8 8 8 16 6.0 1000
```

Parameter description:
- `L1 L2 L3 L4`: Lattice dimensions (x, y, z, t)
- `beta`: Gauge coupling constant
- `ntraj`: Number of trajectories

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
