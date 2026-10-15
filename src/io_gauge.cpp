/**
 * @file io_gauge.cpp
 * @brief I/O functions for gauge field configurations
 *
 * On-disk layout: sites in global lexicographic order (x fastest);
 * per site NDIMS full Nc x Nc complex matrices in MatrixSun memory order, at
 * RealSaveConf precision. MPI builds use collective MPI-IO with a subarray
 * file view so every rank reads/writes only its local sub-block.
 */

#include "io_gauge.hpp"
#include "constants.hpp"
#include "index.hpp"
#include "mpi_layout.hpp"
#include "shift.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

namespace {

template <typename Real, typename RealSaveConf>
void castLinksToSave(const MatrixSun<Real, NCOLORS> *src,
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

int globalGridDim(const LatticeParams &p, int dim) {
  return p.mpi ? p.global_grid[dim] : p.grid[dim];
}

int64_t globalVolumeTotal(const LatticeParams &p) {
  int64_t vol = 1;
  for (int d = 0; d < NDIMS; ++d) {
    vol *= static_cast<int64_t>(globalGridDim(p, d));
  }
  return vol;
}

template <typename Real, typename RealSaveConf>
void castLinksFromFile(const MatrixSun<RealSaveConf, NCOLORS> *src,
                          MatrixSun<Real, NCOLORS> *dst, int ndirs) {
  for (int dir = 0; dir < ndirs; ++dir) {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        dst[dir].e[i][j].real() = static_cast<Real>(src[dir].e[i][j].real());
        dst[dir].e[i][j].imag() = static_cast<Real>(src[dir].e[i][j].imag());
      }
    }
  }
}

/// Bytes per site in the file (NDIMS full Nc x Nc matrices at save precision).
template <typename RealSaveConf> constexpr int siteFileBytes() {
  return static_cast<int>(sizeof(MatrixSun<RealSaveConf, NCOLORS>) * NDIMS);
}

/**
 * @brief Pack local sites [site_lo, site_lo + nsites) into file order.
 *
 * Sites are traversed in local lexicographic order (x fastest), matching the
 * subarray file view; each site contributes NDIMS full Nc x Nc matrices
 * (third row reconstructed when storage is SOA12).
 */
template <typename Real, typename RealSaveConf>
void packSitesForFile(const Complex<Real> *gauge_ptr, const LatticeParams &p,
                      int64_t soa_stride, ArrayType atype, int64_t site_lo,
                      int64_t nsites, char *buf) {
  MatrixSun<Real, NCOLORS> links[NDIMS];
  MatrixSun<RealSaveConf, NCOLORS> links_save[NDIMS];
  const int site_bytes = siteFileBytes<RealSaveConf>();
  for (int64_t s = 0; s < nsites; ++s) {
    int lx[NDIMS];
    indexNdNm(site_lo + s, lx, p);
    const int64_t idx_eo = coordsToEoIdx(lx, p);
    for (int dir = 0; dir < NDIMS; ++dir) {
      loadGaugeLinkSoa(gauge_ptr, idx_eo, dir, soa_stride, p, links[dir],
                       atype);
    }
    castLinksToSave(links, links_save, NDIMS);
    std::memcpy(buf + s * site_bytes, links_save, site_bytes);
  }
}

/// Inverse of \ref packSitesForFile.
template <typename Real, typename RealSaveConf>
void unpackSitesFromFile(const char *buf, Complex<Real> *gauge_ptr,
                         const LatticeParams &p, int64_t soa_stride,
                         ArrayType atype, int64_t site_lo, int64_t nsites) {
  MatrixSun<RealSaveConf, NCOLORS> links_file[NDIMS];
  MatrixSun<Real, NCOLORS> links[NDIMS];
  const int site_bytes = siteFileBytes<RealSaveConf>();
  for (int64_t s = 0; s < nsites; ++s) {
    std::memcpy(links_file, buf + s * site_bytes, site_bytes);
    castLinksFromFile(links_file, links, NDIMS);
    int lx[NDIMS];
    indexNdNm(site_lo + s, lx, p);
    const int64_t idx_eo = coordsToEoIdx(lx, p);
    for (int dir = 0; dir < NDIMS; ++dir) {
      // File has full SU(N); pack into SOA or SOA12 device storage.
      storeGaugeLinkSoa(gauge_ptr, idx_eo, dir, soa_stride, p, links[dir],
                        atype);
    }
  }
}

#ifdef KWQFT_USE_MPI

/// Upper bound for the per-rank staging buffer used by MPI-IO chunks.
constexpr int64_t kIoChunkBytes = static_cast<int64_t>(256) * 1024 * 1024;

/// Duplicated Cartesian communicator with errors returned (not fatal), so I/O
/// failures can be reported consistently on every rank.
MPI_Comm mpiIoComm() {
  MPI_Comm base = kwqftMpiCartComm();
  if (base == MPI_COMM_NULL) {
    base = MPI_COMM_WORLD;
  }
  MPI_Comm dup = MPI_COMM_NULL;
  MPI_Comm_dup(base, &dup);
  MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN);
  return dup;
}

/**
 * @brief Build the etype (one site of file data) and the filetype describing
 *        this rank's sub-block of the global lattice.
 *
 * MPI_ORDER_FORTRAN makes dimension 0 the fastest, matching the on-disk site
 * order produced by \ref indexNdNm.
 */
template <typename RealSaveConf>
void createIoFileTypes(const LatticeParams &p, MPI_Datatype *site_type,
                       MPI_Datatype *file_type) {
  MPI_Type_contiguous(siteFileBytes<RealSaveConf>(), MPI_BYTE, site_type);
  MPI_Type_commit(site_type);

  int coord[NDIMS];
  mpiCartGetCoords(coord);
  int sizes[NDIMS], subsizes[NDIMS], starts[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    sizes[d] = globalGridDim(p, d);
    subsizes[d] = p.grid[d];
    starts[d] = coord[d] * p.grid[d];
  }
  MPI_Type_create_subarray(NDIMS, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                           *site_type, file_type);
  MPI_Type_commit(file_type);
}

/// Chunk size in sites for the MPI-IO staging buffer.
template <typename RealSaveConf>
int64_t ioChunkSites(int64_t local_volume) {
  int64_t chunk = kIoChunkBytes / siteFileBytes<RealSaveConf>();
  if (chunk < 1) {
    chunk = 1;
  }
  if (chunk > local_volume) {
    chunk = local_volume;
  }
  return chunk;
}

template <typename Real, typename RealSaveConf>
void saveGaugeBinaryMpi(const std::string &filename,
                        const Complex<Real> *gauge_ptr, int64_t soa_stride,
                        ArrayType atype) {
  const LatticeParams &p = PARAMS::params;
  const int rank = mpiCommRank();
  if (rank == 0) {
    printf("Saving configuration %s\n", filename.c_str());
  }

  MPI_Comm comm = mpiIoComm();
  MPI_File fh;
  int err = MPI_File_open(comm, filename.c_str(),
                          MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                          &fh);
  if (err != MPI_SUCCESS) {
    KWQFT_ERROR("Error saving configuration: cannot open file");
    return;
  }
  MPI_File_set_errhandler(fh, MPI_ERRORS_RETURN);

  MPI_Datatype site_type, file_type;
  createIoFileTypes<RealSaveConf>(p, &site_type, &file_type);
  err = MPI_File_set_view(fh, 0, site_type, file_type, "native",
                          MPI_INFO_NULL);
  if (err != MPI_SUCCESS) {
    KWQFT_ERROR("ERROR: Unable to set MPI-IO file view");
    return;
  }

  const int64_t local_volume = p.volume;
  const int site_bytes = siteFileBytes<RealSaveConf>();
  const int64_t chunk_sites = ioChunkSites<RealSaveConf>(local_volume);
  std::vector<char> buf(static_cast<size_t>(chunk_sites) * site_bytes);

  for (int64_t lo = 0; lo < local_volume; lo += chunk_sites) {
    const int64_t nsites =
        (lo + chunk_sites <= local_volume) ? chunk_sites : local_volume - lo;
    packSitesForFile<Real, RealSaveConf>(gauge_ptr, p, soa_stride, atype, lo,
                                         nsites, buf.data());
    err = MPI_File_write_at_all(fh, lo, buf.data(), static_cast<int>(nsites),
                                site_type, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
      KWQFT_ERROR("ERROR: Unable to save to file");
      return;
    }
  }

  MPI_File_close(&fh);
  MPI_Type_free(&file_type);
  MPI_Type_free(&site_type);
  MPI_Comm_free(&comm);

  if (rank == 0) {
    printf("Finished saving configuration %s\n", filename.c_str());
  }
}

template <typename Real, typename RealSaveConf>
void loadGaugeBinaryMpi(const std::string &filename,
                        Complex<Real> *gauge_ptr, int64_t soa_stride,
                        ArrayType atype) {
  const LatticeParams &p = PARAMS::params;
  const int rank = mpiCommRank();
  if (rank == 0) {
    printf("Reading configuration %s\n", filename.c_str());
  }

  MPI_Comm comm = mpiIoComm();
  MPI_File fh;
  int err = MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY,
                          MPI_INFO_NULL, &fh);
  if (err != MPI_SUCCESS) {
    KWQFT_ERROR("Error reading configuration: cannot open file");
    return;
  }
  MPI_File_set_errhandler(fh, MPI_ERRORS_RETURN);

  MPI_Offset file_size = 0;
  MPI_File_get_size(fh, &file_size);
  const int64_t expected_body_bytes =
      globalVolumeTotal(p) * siteFileBytes<RealSaveConf>();
  if (file_size != expected_body_bytes) {
    KWQFT_ERROR("Configuration file size does not match lattice volume");
    return;
  }

  MPI_Datatype site_type, file_type;
  createIoFileTypes<RealSaveConf>(p, &site_type, &file_type);
  err = MPI_File_set_view(fh, 0, site_type, file_type, "native",
                          MPI_INFO_NULL);
  if (err != MPI_SUCCESS) {
    KWQFT_ERROR("ERROR: Unable to set MPI-IO file view");
    return;
  }

  const int64_t local_volume = p.volume;
  const int site_bytes = siteFileBytes<RealSaveConf>();
  const int64_t chunk_sites = ioChunkSites<RealSaveConf>(local_volume);
  std::vector<char> buf(static_cast<size_t>(chunk_sites) * site_bytes);

  for (int64_t lo = 0; lo < local_volume; lo += chunk_sites) {
    const int64_t nsites =
        (lo + chunk_sites <= local_volume) ? chunk_sites : local_volume - lo;
    err = MPI_File_read_at_all(fh, lo, buf.data(), static_cast<int>(nsites),
                               site_type, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
      KWQFT_ERROR("ERROR: Unable to read configuration file");
      return;
    }
    unpackSitesFromFile<Real, RealSaveConf>(buf.data(), gauge_ptr, p,
                                            soa_stride, atype, lo, nsites);
  }

  MPI_File_close(&fh);
  MPI_Type_free(&file_type);
  MPI_Type_free(&site_type);
  MPI_Comm_free(&comm);

  if (rank == 0) {
    printf("Finished reading configuration %s\n", filename.c_str());
  }
}

#endif // KWQFT_USE_MPI

} // namespace

template <typename Real, typename RealSaveConf>
void saveGaugeBinary(const GaugeArray<Real> &gauge,
                       const std::string &filename) {
  const LatticeParams &p = PARAMS::params;

  if (!gauge.evenOdd()) {
    KWQFT_ERROR("save_gauge_binary requires even/odd gauge storage");
    return;
  }

  Kokkos::fence();
  auto host_view = Kokkos::create_mirror_view(gauge.getView());
  Kokkos::deep_copy(host_view, gauge.getView());
  const Complex<Real> *gauge_ptr = host_view.data();
  const int64_t soa_stride = gauge.size();
  const ArrayType atype = gauge.type();

#ifdef KWQFT_USE_MPI
  if (p.mpi && p.nproc > 1) {
    saveGaugeBinaryMpi<Real, RealSaveConf>(filename, gauge_ptr, soa_stride,
                                           atype);
    return;
  }
#endif

  std::ofstream fileout(filename, std::ios::binary | std::ios::out);
  if (!fileout.is_open()) {
    KWQFT_ERROR("Error saving configuration: cannot open file");
    return;
  }
  printf("Saving configuration %s\n", filename.c_str());

  int global_grid[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_grid[d] = globalGridDim(p, d);
  }

  const int64_t global_volume = globalVolumeTotal(p);
  MatrixSun<Real, NCOLORS> links[NDIMS];
  MatrixSun<RealSaveConf, NCOLORS> links_save[NDIMS];
  const int site_bytes = siteFileBytes<RealSaveConf>();

  for (int64_t site = 0; site < global_volume; ++site) {
    int gx[NDIMS];
    indexNdNm(site, gx, global_grid);

    int lx[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      lx[d] = gx[d] % p.grid[d];
    }
    const int64_t idx_eo = coordsToEoIdx(lx, p);
    for (int dir = 0; dir < NDIMS; ++dir) {
      // Reconstruct full SU(N) if storage is SOA12; file always stores Nc×Nc.
      loadGaugeLinkSoa(gauge_ptr, idx_eo, dir, soa_stride, p, links[dir],
                       atype);
    }
    castLinksToSave(links, links_save, NDIMS);
    fileout.write(reinterpret_cast<const char *>(links_save), site_bytes);
    if (fileout.fail()) {
      KWQFT_ERROR("ERROR: Unable to save to file");
      return;
    }
  }

  fileout.close();
  printf("Finished saving configuration %s\n", filename.c_str());
}

template <typename Real, typename RealSaveConf>
void loadGaugeBinary(GaugeArray<Real> &gauge, const std::string &filename) {
  const LatticeParams &p = PARAMS::params;

  if (!gauge.evenOdd()) {
    KWQFT_ERROR("load_gauge_binary requires even/odd gauge storage");
    return;
  }

  auto host_view = Kokkos::create_mirror_view(gauge.getView());
  Complex<Real> *gauge_ptr = host_view.data();
  const int64_t soa_stride = gauge.size();
  const ArrayType atype = gauge.type();

#ifdef KWQFT_USE_MPI
  if (p.mpi && p.nproc > 1) {
    loadGaugeBinaryMpi<Real, RealSaveConf>(filename, gauge_ptr, soa_stride,
                                           atype);
    Kokkos::deep_copy(gauge.getView(), host_view);
    Kokkos::fence();
    gauge.invalidateHalo();
    return;
  }
#endif

  std::ifstream filein(filename, std::ios::binary | std::ios::in);
  if (!filein.is_open()) {
    KWQFT_ERROR("Error reading configuration: cannot open file");
    return;
  }
  printf("Reading configuration %s\n", filename.c_str());

  filein.seekg(0, std::ios::end);
  const std::streamoff file_size = filein.tellg();
  filein.seekg(0, std::ios::beg);
  const int64_t expected_body_bytes =
      globalVolumeTotal(p) * siteFileBytes<RealSaveConf>();
  if (file_size != expected_body_bytes) {
    KWQFT_ERROR("Configuration file size does not match lattice volume");
    return;
  }

  int global_grid[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_grid[d] = globalGridDim(p, d);
  }
  const int64_t global_volume = globalVolumeTotal(p);

  MatrixSun<Real, NCOLORS> links[NDIMS];
  MatrixSun<RealSaveConf, NCOLORS> links_file[NDIMS];
  const int site_bytes = siteFileBytes<RealSaveConf>();

  for (int64_t site = 0; site < global_volume; ++site) {
    int gx[NDIMS];
    indexNdNm(site, gx, global_grid);

    filein.read(reinterpret_cast<char *>(links_file), site_bytes);
    if (filein.fail()) {
      KWQFT_ERROR("ERROR: Unable to read configuration file");
      return;
    }
    castLinksFromFile(links_file, links, NDIMS);

    int lx[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      lx[d] = gx[d] % p.grid[d];
    }
    const int64_t idx_eo = coordsToEoIdx(lx, p);
    for (int dir = 0; dir < NDIMS; ++dir) {
      // File has full SU(N); pack into SOA or SOA12 device storage.
      storeGaugeLinkSoa(gauge_ptr, idx_eo, dir, soa_stride, p, links[dir],
                        atype);
    }
  }

  filein.close();
  printf("Finished reading configuration %s\n", filename.c_str());

  Kokkos::deep_copy(gauge.getView(), host_view);
  Kokkos::fence();
  gauge.invalidateHalo();
}

template void saveGaugeBinary<float, float>(const GaugeArray<float> &,
                                              const std::string &);
template void saveGaugeBinary<float, double>(const GaugeArray<float> &,
                                               const std::string &);
template void saveGaugeBinary<double, float>(const GaugeArray<double> &,
                                               const std::string &);
template void saveGaugeBinary<double, double>(const GaugeArray<double> &,
                                                const std::string &);

template void loadGaugeBinary<float, float>(GaugeArray<float> &,
                                              const std::string &);
template void loadGaugeBinary<float, double>(GaugeArray<float> &,
                                               const std::string &);
template void loadGaugeBinary<double, float>(GaugeArray<double> &,
                                                const std::string &);
template void loadGaugeBinary<double, double>(GaugeArray<double> &,
                                                const std::string &);

} // namespace kwqft
