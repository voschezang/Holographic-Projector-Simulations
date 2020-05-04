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
  size_t Nx = 1, Ny = N, Nz = N;
  Geometry p = init::params(N); // TODO for both y,z
  printf("\nHyperparams:");
  printf("\n CUDA geometry: <<<%i,%i>>>", p.gridSize, p.blockSize);
  printf("\t(%ik threads)", p.gridSize * p.blockSize * 1e-3);

  printf("\n Input size (datapoints): x: %i, y: %i, z: %i", Nx, Ny, Nz);
  printf("\n E[N_x / thread]: %6fk", Nx / (double) p.gridSize * p.blockSize * 1e-3);
  printf("\tE[N_y / thread]: %6fk", Ny / (double) p.gridSize * p.blockSize * 1e-3);

  printf("\n n streams: %4i", p.n_streams);
  printf("\tbatch size: \t%6i", p.stream_size);
  printf("\tkernel size: \t%4i", p.kernel_size);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB());

  {
    // auto n = double{BLOCKDIM * BATCH_SIZE};
    // auto m = double{n * sizeof(WTYPE) * 1e-3};
    double n = BLOCKDIM * STREAM_BATCH_SIZE;
    double m = n * sizeof(WTYPE) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
  struct timespec t0, t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // TODO use cmd arg for x length
  auto
    X = std::vector<WTYPE>(Nx, {1.0});

  auto
    U = std::vector<STYPE>(X.size() * DIMS),
    V = std::vector<STYPE>(Ny * DIMS),
    W = std::vector<STYPE>(Nz * DIMS);

  init::planes(U, V, W);
  summarize_double('u', U);
  summarize_double('v', V);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("runtime init: \t%0.3f\n", dt(t0, t1));
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  cudaProfilerStart();
#ifdef Y_TRANSFORM
  // if X does not fit on GPU then do y += transform(x') for each subset x' in X
  auto Y = transform<Direction::Backward>(X, U, V);
#endif

#ifdef Z_TRANSFORM
  printf("\nSecond transform:\n");
  auto Z = transform<Direction::Forward>(Y, V, W);
#else
  auto Z = std::vector<WTYPE>(1);
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

  check_cvector(Y);
  check_cvector(Z);

#ifdef Y_TRANSFORM
  summarize_c('y', Y);
#endif
#ifdef Z_TRANSFORM
  summarize_c('z', Z);
#endif

  // write_arrays<FileType::TXT>(x,y,z, u,v,w, N);
  write_arrays<FileType::TXT>(X,Y,Z, U,V,W);
	return 0;
}
