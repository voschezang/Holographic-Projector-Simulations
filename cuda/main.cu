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
  printf("\nHyperparams:");
  printf("\n"); printf(" N: %4i^2 =%6i", N_sqrt, N);
  printf("\t"); printf("BATCH_SIZE:\t%8i", BATCH_SIZE);
  printf("\t"); printf("N_BATCHES: %8i", N_BATCHES);

  printf("\n"); printf(" GRIDDIM: %8i", GRIDDIM);
  printf("\t"); printf("BLOCKDIM: %8i", BLOCKDIM);
  printf("\t"); printf("E[tasks] = %0.3fk", GRIDDIM * BLOCKDIM * 1e-3);
  printf("\t"); printf("\tN/thread: %i", N_PER_THREAD);
  printf("\n"); printf(" N_STREAMS %3i \t\tSTREAM SIZE: %i (x3)", N_STREAMS, STREAM_SIZE);
  printf("\t"); printf("\tBATCHES_PER_STREAM (x BATCH_SIZE = N): %i (x %i = %i)\n", BATCHES_PER_STREAM, BATCH_SIZE, BATCHES_PER_STREAM * BATCH_SIZE);
  printf("KERNELS_PER_BATCH %3i \t\tKERNEL BATCH SIZE: %i\n", KERNELS_PER_BATCH, KERNEL_BATCH_SIZE);
  // if (BATCHES_PER_STREAM < BATCH_SIZE)
  //   printf("BATCHES_PER_STREAM (%i) < BATCH_SIZE (%i)\n", BATCHES_PER_STREAM, BATCH_SIZE);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB());
  {
    // auto n = double{BLOCKDIM * BATCH_SIZE};
    // auto m = double{n * sizeof(WTYPE) * 1e-3};
    double n = BLOCKDIM * BATCH_SIZE;
    double m = n * sizeof(WTYPE) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
  check_params();
  struct timespec t0, t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // use "complex" datatypes, overhead on CPU side can be ignored
  auto
    X = std::vector<WTYPE>(N),
    Y = std::vector<WTYPE>(N),
    Z = std::vector<WTYPE>(N);

  auto
    U = std::vector<STYPE>(N * DIMS),
    V = std::vector<STYPE>(N * DIMS),
    W = std::vector<STYPE>(N * DIMS);

  // use C-style pointers for backwards compatibility
  WTYPE
    *x = &X[0],
    *y = &Y[0],
    *z = &Z[0];

  STYPE
    *u = &U[0],
    *v = &V[0],
    *w = &W[0];

  init_planes(x, u, v, w);
  summarize_double('u', u, N * DIMS);
  summarize_double('v', v, N * DIMS);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("runtime init: \t%0.3f\n", dt(t0, t1));
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  cudaProfilerStart();
  if (Y_TRANSFORM) {
    transform<Direction::Backward>(X, y, U, v);
  } else {
    printf("skipping y\n");
  }
  if (Z_TRANSFORM) {
    printf("\nSecond transform:\n");
    transform<Direction::Forward>(Y, z, V, w);
    // transform(x, z, u, v, 1);
  }
  cudaProfilerStop();

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &t2);
  printf("done\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  double time = dt(t1, t2);
  printf("runtime init: \t%0.3f\n", time);

  if (Z_TRANSFORM) {
    printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n",  \
           flops(time), FLOP_PER_POINT);
    printf("Bandwidth: \t%0.5f MB/s (excl. shared memory)\n", bandwidth(time, 2, 0));
    printf("Bandwidth: \t%0.5f MB/s (incl. shared memory)\n", bandwidth(time, 2, 1));
  }
  else {
    printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n",  \
         2*flops(time), 2*FLOP_PER_POINT);
    printf("Bandwidth: \t%0.5f Mb/s (excl. shared memory)\n", bandwidth(time, 1, 0));
    printf("Bandwidth: \t%0.5f MB/s (incl. shared memory)\n", bandwidth(time, 1, 1));
  }
#ifdef DEBUG
  for (size_t i = 0; i < N2; ++i) {
    assert(cuCabs(y[i]) < DBL_MAX);
    assert(cuCabs(z[i]) < DBL_MAX);
  }
#endif

  if (Y_TRANSFORM) summarize_c('y', y, N);
  if (Z_TRANSFORM) summarize_c('z', z, N);

#ifdef DEBUG
  printf("save results\n");
#endif
  write_arrays<FileType::TXT>(x,y,z, u,v,w, N);
	return 0;
}