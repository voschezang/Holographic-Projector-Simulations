// #define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h> // unused in this file but causes error if omitted
#include <iostream>

#include "macros.h"
#include "hyper_params.h"
#include "kernel.cu"
#include "init.h"
#include "util.h"
#include "functions.h"

/**
 * Input x,u is splitted over GPU cores/threads
 * Output y,v is streamed (send in batches).
 *
 * It is assumed that x,u all fit in GPU memory, but not necessarily in cache
 * Batches containing parts of y,v are send back to CPU immediately
 *
 * Naming convention
 * i,j,k = indices in flattened arrays
 * n,m = counters
 * N,M = sizes
 *
 * e.g. n = [0,..,N-1]
 */


int main() {
  const size_t Nx = 1, Ny = N, Nz = N;
  const Geometry p = init::geometry(N); // TODO for both y,z
  const Params params = init::params();
  printf("\nHyperparams:");
  printf("\n CUDA geometry: <<<%i,%i>>>", p.gridSize, p.blockSize);
  printf("\t(%fk threads)", p.gridSize * p.blockSize * 1e-3);

  printf("\n Input size (datapoints): x: %i, y: %i, z: %i", Nx, Ny, Nz);
  printf("\n E[N_x / thread]: %6fk", Nx / (double) p.gridSize * p.blockSize * 1e-3);
  printf("\tE[N_y / thread]: %6fk", Ny / (double) p.gridSize * p.blockSize * 1e-3);

  printf("\n n streams: %4i", p.n_streams);
  printf("\tbatch size: \t%6i", p.stream_size);
  printf("\tkernel size: \t%4i", p.kernel_size);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB());
  {
    size_t n = SHARED_MEMORY_SIZE(p.blockSize);
    double m = n * sizeof(WTYPE) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
  struct timespec t0, t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // TODO use cmd arg for x length
  // TODO rm old lowercase vars and replace them by current uppercase vars
  auto
    x = std::vector<WTYPE>(Nx, {1.0});

  auto
    u = init::sparse_plane(x.size(), params.input.width),
    v = init::plane(Ny, params.projector);

  summarize_double('u', u);
  summarize_double('v', v);
  write_arrays<FileType::TXT>(x, u, "xu", true);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("Runtime init: \t%0.3f\n", dt(t0, t1));
  cudaProfilerStart();
  printf("--- --- ---   --- --- ---  --- --- --- \n");

  // The projector distribution is obtained by doing a single backwards transformation
  // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x
  auto y = time_transform<Direction::Backward>(x, u, v, p, &t1, &t2, 1);
  check_cvector(y);
  summarize_c('y', y);
  write_arrays<FileType::TXT>(y, v, "yv", false);

  // The projection distributions at various locations are obtained using forward transformations
  const int n_planes = 2;
  for (unsigned int i = 1; i < n_planes; ++i) {
    auto p = init::geometry(Nz);
    auto w = init::plane(Nz, params.projection);
    auto z = time_transform<Direction::Forward>(x, u, v, p, &t1, &t2, 1);
    check_cvector(z);
    summarize_c('z', z);
    // TODO do this async
    write_arrays<FileType::TXT>(z, w, "zw", false);
  }

  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("done\n");
  cudaProfilerStop();
	return 0;
}
