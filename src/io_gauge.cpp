/**
 * @file io_gauge.cpp
 * @brief I/O functions for gauge field configurations
 */

#include "constants.hpp"
#include "index.hpp"
#include "io_gauge.hpp"
#include "mpi_layout.hpp"
#include "shift.hpp"

#include <fstream>
#include <string>

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

namespace {

template <typename Real, typename RealSaveConf>
void cast_links_to_save(const MatrixSun<Real, NCOLORS> *src,
                        MatrixSun<RealSaveConf, NCOLORS> *dst, int ndirs) {
  for (int dir = 0; dir < ndirs; ++dir) {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        dst[dir].e[i][j].real() =
            static_cast<RealSaveConf>(src[dir].e[i][j].real());
        dst[dir].e[i][j].imag() =
            static_cast<RealSaveConf>(src[dir].e[i][j].imag());
      }
    }
  }
}

int global_grid_dim(const LatticeParams &p, int dim) {
  return p.mpi ? p.global_grid[dim] : p.grid[dim];
}

int64_t global_volume_total(const LatticeParams &p) {
  int64_t vol = 1;
  for (int d = 0; d < NDIMS; ++d) {
    vol *= static_cast<int64_t>(global_grid_dim(p, d));
  }
  return vol;
}

template <typename Real, typename RealSaveConf>
void cast_links_from_file(const MatrixSun<RealSaveConf, NCOLORS> *src,
                          MatrixSun<Real, NCOLORS> *dst, int ndirs) {
  for (int dir = 0; dir < ndirs; ++dir) {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        dst[dir].e[i][j].real() =
            static_cast<Real>(src[dir].e[i][j].real());
        dst[dir].e[i][j].imag() =
            static_cast<Real>(src[dir].e[i][j].imag());
      }
    }
  }
}

template <typename Real>
void store_gauge_link_soa(Complex<Real> *gauge_ptr, int64_t idx_eo, int dir,
                          int64_t soa_stride, const LatticeParams &p,
                          const MatrixSun<Real, NCOLORS> &u) {
  const int64_t base =
      idx_eo + dir * p.volume;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      gauge_ptr[base + (j + i * NCOLORS) * soa_stride] = u.e[i][j];
    }
  }
}

bool config_params_mismatch(const LatticeParams &p, const int grid_dim[NDIMS],
                            double beta) {
  for (int d = 0; d < NDIMS; ++d) {
    if (grid_dim[d] != global_grid_dim(p, d)) {
      return true;
    }
  }
  return p.beta != beta;
}

} // namespace

template <typename Real, typename RealSaveConf>
void save_gauge_binary(const GaugeArray<Real> &gauge, const std::string &filename,
                       bool withheader) {
  const LatticeParams &p = PARAMS::params;

  if (!gauge.even_odd()) {
    KWQFT_ERROR("save_gauge_binary requires even/odd gauge storage");
    return;
  }
  if (gauge.type() != ArrayType::SOA) {
    KWQFT_ERROR("save_gauge_binary requires SOA gauge storage");
    return;
  }

  Kokkos::fence();
  auto host_view = Kokkos::create_mirror_view(gauge.getView());
  Kokkos::deep_copy(host_view, gauge.getView());
  const Complex<Real> *gauge_ptr = host_view.data();
  const int64_t soa_stride = gauge.size();

  const int rank = mpi_comm_rank();
  const int master = 0;

#ifdef KWQFT_USE_MPI
  const bool mpi_save = p.mpi && p.nproc > 1;
#endif

  std::ofstream fileout;
  if (rank == master) {
    fileout.open(filename, std::ios::binary | std::ios::out);
    if (!fileout.is_open()) {
      KWQFT_ERROR("Error saving configuration: cannot open file");
      return;
    }
    printf("Saving configuration %s\n", filename.c_str());

    if (withheader) {
      int grid_out[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        grid_out[d] = global_grid_dim(p, d);
      }
      fileout.write(reinterpret_cast<const char *>(grid_out),
                    sizeof(int) * NDIMS);
      fileout.write(reinterpret_cast<const char *>(&p.beta), sizeof(Real));
      const size_t confprec = sizeof(RealSaveConf);
      fileout.write(reinterpret_cast<const char *>(&confprec), sizeof(size_t));
    }
  }

  int global_grid[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_grid[d] = global_grid_dim(p, d);
  }

  const int64_t global_volume = global_volume_total(p);
  MatrixSun<Real, NCOLORS> links[NDIMS];
  MatrixSun<RealSaveConf, NCOLORS> links_save[NDIMS];

  const int link_bytes =
      static_cast<int>(sizeof(MatrixSun<Real, NCOLORS>) * NDIMS);
  const int link_save_bytes =
      static_cast<int>(sizeof(MatrixSun<RealSaveConf, NCOLORS>) * NDIMS);

  for (int64_t site = 0; site < global_volume; ++site) {
    int gx[NDIMS];
    indexNdNm(site, gx, global_grid);

    int owner_rank = master;
#ifdef KWQFT_USE_MPI
    if (mpi_save) {
      int cart[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        cart[d] = gx[d] / p.grid[d];
      }
      MPI_Cart_rank(kwqft_mpi_cart_comm(), cart, &owner_rank);
    }
#endif

    if (rank == owner_rank) {
      int lx[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        lx[d] = gx[d] % p.grid[d];
      }
      const int64_t idx_eo = coords_to_eo_idx(lx, p);
      for (int dir = 0; dir < NDIMS; ++dir) {
        loadGaugeLinkSoa(gauge_ptr, idx_eo, dir, soa_stride, p, links[dir]);
      }
    }

#ifdef KWQFT_USE_MPI
    if (mpi_save) {
      if (rank == owner_rank && rank != master) {
        MPI_Send(links, link_bytes, MPI_BYTE, master, owner_rank,
                 MPI_COMM_WORLD);
      }
      if (rank != owner_rank && rank == master) {
        MPI_Status status;
        MPI_Recv(links, link_bytes, MPI_BYTE, owner_rank, owner_rank,
                 MPI_COMM_WORLD, &status);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    if (rank == master) {
      if (sizeof(Real) != sizeof(RealSaveConf)) {
        cast_links_to_save(links, links_save, NDIMS);
        fileout.write(reinterpret_cast<const char *>(links_save),
                      link_save_bytes);
      } else {
        fileout.write(reinterpret_cast<const char *>(links), link_bytes);
      }
      if (fileout.fail()) {
        KWQFT_ERROR("ERROR: Unable to save to file");
        return;
      }
    }
  }

  if (rank == master) {
    fileout.close();
    printf("Finished saving configuration %s\n", filename.c_str());
  }
}

template <typename Real, typename RealSaveConf>
void load_gauge_binary(GaugeArray<Real> &gauge, const std::string &filename,
                       bool withheader) {
  const LatticeParams &p = PARAMS::params;

  if (!gauge.even_odd()) {
    KWQFT_ERROR("load_gauge_binary requires even/odd gauge storage");
    return;
  }
  if (gauge.type() != ArrayType::SOA) {
    KWQFT_ERROR("load_gauge_binary requires SOA gauge storage");
    return;
  }

  const int rank = mpi_comm_rank();
  const int master = 0;

#ifdef KWQFT_USE_MPI
  const bool mpi_load = p.mpi && p.nproc > 1;
#endif

  const int link_bytes =
      static_cast<int>(sizeof(MatrixSun<Real, NCOLORS>) * NDIMS);
  const int link_file_bytes =
      static_cast<int>(sizeof(MatrixSun<RealSaveConf, NCOLORS>) * NDIMS);

  int global_grid[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_grid[d] = global_grid_dim(p, d);
  }
  const int64_t global_volume = global_volume_total(p);
  const int64_t expected_body_bytes =
      global_volume * static_cast<int64_t>(
                          (sizeof(Real) != sizeof(RealSaveConf))
                              ? link_file_bytes
                              : link_bytes);

  std::ifstream filein;
  if (rank == master) {
    filein.open(filename, std::ios::binary | std::ios::in);
    if (!filein.is_open()) {
      KWQFT_ERROR("Error reading configuration: cannot open file");
      return;
    }
    printf("Reading configuration %s\n", filename.c_str());

    if (withheader) {
      int grid_dim[NDIMS];
      filein.read(reinterpret_cast<char *>(grid_dim), sizeof(int) * NDIMS);
      Real beta_file = Real(0);
      filein.read(reinterpret_cast<char *>(&beta_file), sizeof(Real));
      size_t confprec = 0;
      filein.read(reinterpret_cast<char *>(&confprec), sizeof(size_t));

      if (filein.fail()) {
        KWQFT_ERROR("ERROR: Unable to read configuration header");
        return;
      }
      if (confprec != sizeof(RealSaveConf)) {
        KWQFT_ERROR("Input lattice precision does not match file");
        return;
      }
      if (config_params_mismatch(p, grid_dim, static_cast<double>(beta_file))) {
        KWQFT_ERROR("Input lattice parameters do not match configuration file");
        return;
      }
    } else {
      filein.seekg(0, std::ios::end);
      const std::streamoff file_size = filein.tellg();
      filein.seekg(0, std::ios::beg);
      if (file_size != expected_body_bytes) {
        KWQFT_ERROR("Configuration file size does not match lattice volume");
        return;
      }
    }
  }

  auto host_view = Kokkos::create_mirror_view(gauge.getView());
  Complex<Real> *gauge_ptr = host_view.data();
  const int64_t soa_stride = gauge.size();

  MatrixSun<Real, NCOLORS> links[NDIMS];
  MatrixSun<RealSaveConf, NCOLORS> links_file[NDIMS];

  for (int64_t site = 0; site < global_volume; ++site) {
    int gx[NDIMS];
    indexNdNm(site, gx, global_grid);

    int owner_rank = master;
#ifdef KWQFT_USE_MPI
    if (mpi_load) {
      int cart[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        cart[d] = gx[d] / p.grid[d];
      }
      MPI_Cart_rank(kwqft_mpi_cart_comm(), cart, &owner_rank);
    }
#endif

    if (rank == master) {
      if (sizeof(Real) != sizeof(RealSaveConf)) {
        filein.read(reinterpret_cast<char *>(links_file), link_file_bytes);
        if (filein.fail()) {
          KWQFT_ERROR("ERROR: Unable to read configuration file");
          return;
        }
        cast_links_from_file(links_file, links, NDIMS);
      } else {
        filein.read(reinterpret_cast<char *>(links), link_bytes);
        if (filein.fail()) {
          KWQFT_ERROR("ERROR: Unable to read configuration file");
          return;
        }
      }
    }

#ifdef KWQFT_USE_MPI
    if (mpi_load) {
      if (rank == master && rank != owner_rank) {
        MPI_Send(links, link_bytes, MPI_BYTE, owner_rank, owner_rank,
                 MPI_COMM_WORLD);
      }
      if (rank == owner_rank && rank != master) {
        MPI_Status status;
        MPI_Recv(links, link_bytes, MPI_BYTE, master, owner_rank,
                 MPI_COMM_WORLD, &status);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    if (rank == owner_rank) {
      int lx[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        lx[d] = gx[d] % p.grid[d];
      }
      const int64_t idx_eo = coords_to_eo_idx(lx, p);
      for (int dir = 0; dir < NDIMS; ++dir) {
        store_gauge_link_soa(gauge_ptr, idx_eo, dir, soa_stride, p,
                             links[dir]);
      }
    }
  }

  if (rank == master) {
    filein.close();
    printf("Finished reading configuration %s\n", filename.c_str());
  }

  Kokkos::deep_copy(gauge.getView(), host_view);
  Kokkos::fence();
}

template void save_gauge_binary<float, float>(const GaugeArray<float> &,
                                              const std::string &, bool);
template void save_gauge_binary<float, double>(const GaugeArray<float> &,
                                               const std::string &, bool);
template void save_gauge_binary<double, float>(const GaugeArray<double> &,
                                               const std::string &, bool);
template void save_gauge_binary<double, double>(const GaugeArray<double> &,
                                                const std::string &, bool);

template void load_gauge_binary<float, float>(GaugeArray<float> &,
                                              const std::string &, bool);
template void load_gauge_binary<float, double>(GaugeArray<float> &,
                                               const std::string &, bool);
template void load_gauge_binary<double, float>(GaugeArray<double> &,
                                               const std::string &, bool);
template void load_gauge_binary<double, double>(GaugeArray<double> &,
                                                const std::string &, bool);

} // namespace kwqft
