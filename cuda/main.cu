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
    v = init::plane(Ny, params.projector),
    w = init::plane(Nz, params.projection);

  summarize_double('u', u);
  summarize_double('v', v);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("runtime init: \t%0.3f\n", dt(t0, t1));
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  cudaProfilerStart();
  // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x
  auto y = transform<Direction::Backward>(x, u, v, p);

#ifdef Z_TRANSFORM
  printf("\nSecond transform:\n");
  auto z = transform<Direction::Forward>(y, v, w, init::geometry(Nz));
#else
  auto z = std::vector<WTYPE>(1);
#endif

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &t2);
  cudaProfilerStop();
  printf("done\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  double time = dt(t1, t2);
  printf("runtime init: \t%0.3f\n", time);

#ifdef Z_TRANSFORM
  // TODO allow smaller datasize for N in flops computation
  printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n",  \
         flops(time), FLOP_PER_POINT);
  printf("Bandwidth: \t%0.5f MB/s (excl. shared memory)\n", bandwidth(time, 2, 0));
  printf("Bandwidth: \t%0.5f MB/s (incl. shared memory)\n", bandwidth(time, 2, 1));
#else
  printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n",  \
         2*flops(time), 2*FLOP_PER_POINT);
  printf("Bandwidth: \t%0.5f Mb/s (excl. shared memory)\n", bandwidth(time, 1, 0));
  printf("Bandwidth: \t%0.5f MB/s (incl. shared memory)\n", bandwidth(time, 1, 1));
#endif

  check_cvector(y);
  check_cvector(z);

  summarize_c('y', y);
#ifdef Z_TRANSFORM
  summarize_c('z', z);
#endif

  write_arrays<FileType::TXT>(x,y,z, u,v,w);
	return 0;
}
