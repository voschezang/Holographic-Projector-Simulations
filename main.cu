//#define _POSIX_C_SOURCE 199309L

#include <assert.h>
// #include <complex.h> // TODO use cpp cmath
#include <cuComplex.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_profiler_api.h>

#include <thrust/host_vector.h> // unused in this file but causes error if omitted

#include "macros.h"
#include "kernel.cu"
#include "functions.h"

/**
 * Input x,u is splitted over GPU cores/threads
 * Output y,v is streamed (send in batches).
 * The arrays y,v live on CPU, the batches d_y, d_v live on GPU.
 *
 * It is assumed that x,y,u,v all fit in GPU memory, but not necessarily in cache
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

  printf("\n"); printf(" BLOCKDIM: %8i", BLOCKDIM);
  printf("\t"); printf("THREADS_PER_BLOCK: %8i", THREADS_PER_BLOCK);
  printf("\t"); printf("E[tasks] = %0.3fk", BLOCKDIM * THREADS_PER_BLOCK * 1e-3);
  printf("\t"); printf("\tN/thread: %i", N_PER_THREAD);
  printf("\n"); printf(" N_STREAMS %3i \t\tSTREAM SIZE: %i (x3)", N_STREAMS, STREAM_SIZE);
  printf("\t"); printf("\tBATCHES_PER_STREAM (x BATCH_SIZE = N): %i (x %i = %i)\n", BATCHES_PER_STREAM, BATCH_SIZE, BATCHES_PER_STREAM * BATCH_SIZE);
  // if (BATCHES_PER_STREAM < BATCH_SIZE)
  //   printf("BATCHES_PER_STREAM (%i) < BATCH_SIZE (%i)\n", BATCHES_PER_STREAM, BATCH_SIZE);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB());
  {
    double n = THREADS_PER_BLOCK * BATCH_SIZE;
    double m = n * sizeof(WTYPE_cuda) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
// #ifdef CACHE_BATCH
//   assert(BATCH_SIZE < THREADS_PER_BLOCK);
// #endif
  check_params();

  struct timespec t0, t1, t2;
	const size_t size = N * sizeof( WTYPE );
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // host
  // problem with cudaMallocManaged: cuda complex dtypes differ from normal
  WTYPE
    *x = (WTYPE *) malloc(size),
    *y = (WTYPE *) malloc(size),
    *z = (WTYPE *) malloc(size);

  STYPE
    *u = (STYPE *) malloc(DIMS * N * sizeof(STYPE)),
    *v = (STYPE *) malloc(DIMS * N * sizeof(STYPE)),
    *w = (STYPE *) malloc(DIMS * N * sizeof(STYPE));

  {
    const double width = 0.0005; // m
    const double dS = width * SCALE / N_sqrt; // actually dS^(1/DIMS)
    const double offset = 0.5 * N_sqrt * dS;
#if defined(RANDOM_X_SPACE) || defined(RANDOM_Y_SPACE) || defined(RANDOM_Z_SPACE)
    const double margin = 0.;
    const double random_range = dS - 0.5 * margin;
#endif
    printf("Domain: X : %f x %f, dS: %f\n", width, width, dS);
    for(unsigned int i = 0; i < N_sqrt; ++i) {
      for(unsigned int j = 0; j < N_sqrt; ++j) {
        size_t idx = i * N_sqrt + j;
        x[idx].x = 0;
        x[idx].y = 0;
        if (i == N_sqrt * 1/2 && j == N_sqrt / 2) x[idx].x = 1;
        // if (i == N_sqrt * 1/3 && j == N_sqrt / 2) x[idx] = 1;
        // if (i == N_sqrt * 2/3 && j == N_sqrt / 2) x[idx] = 1;
        // if (i == N_sqrt * 1/4 && j == N_sqrt / 4) x[idx] = 1;
        // if (i == N_sqrt * 3/4 && j == N_sqrt / 4) x[idx] = 1;

        u[Ix(i,j,0)] = i * dS - offset;
        u[Ix(i,j,1)] = j * dS - offset;
        u[Ix(i,j,2)] = 0;

        v[Ix(i,j,0)] = i * dS - offset;
        v[Ix(i,j,1)] = j * dS - offset;
        v[Ix(i,j,2)] = -0.02;
        // if (i == 1 && j == 1) {
        if (i == 2 && j == 2) {
          // printf("random: %f\n", rand() / (double) RAND_MAX - 0.5);
          // printf("random: %f\n", rand() / (double) RAND_MAX - 0.5);
          // printf("random: %f\n", rand() / (double) RAND_MAX - 0.5);
          printf("i,j %i,%i\n", i,j);
          // printf("rand: %f, %f\n", v[Ix(i,j,0)], v[Ix(i,j-1,0)]);
          printf("rand: %f, %f; %f, %f\n", v[Ix(i,j,0)], v[Ix(i-1,j,0)], v[Ix(i,j-1,0)], v[Ix(i-1,j-1,0)]);
        }
#ifdef RANDOM_Y_SPACE
        v[Ix(i,j,0)] += random_range * (rand() / (double) RAND_MAX - 0.5);
        v[Ix(i,j,1)] += random_range * (rand() / (double) RAND_MAX - 0.5);
        if (i == 2 && j == 2) {
          printf("rand: %f, %f; %f, %f\n", v[Ix(i,j,0)], v[Ix(i-1,j,0)], v[Ix(i,j-1,0)], v[Ix(i-1,j-1,0)]);
        }
#endif

        w[Ix(i,j,0)] = i * dS - offset;
        w[Ix(i,j,1)] = j * dS - offset;
        w[Ix(i,j,2)] = 0;
#ifdef RANDOM_Z_SPACE
        w[Ix(i,j,0)] += random_range * (rand() / (double) RAND_MAX - 0.5);
        w[Ix(i,j,1)] += random_range * (rand() / (double) RAND_MAX - 0.5);
#endif
      } }
  }

  summarize_double('u', u, N * DIMS);
  summarize_double('v', v, N * DIMS);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  {
    double dt = (double) (t1.tv_sec - t0.tv_sec) + ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
    printf("runtime init: \t%0.3f\n", dt);
  }
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");

  transform(x, y, u, v, -1);
#ifdef Z
  printf("\nSecond transform:\n");
  transform(y, z, v, w, 1);
#endif

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &t2);
  double dt = (double) (t2.tv_sec - t1.tv_sec) + ((double) (t2.tv_nsec - t1.tv_nsec)) * 1e-9;
  printf("done\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("runtime loop: \t%0.3f\n", dt);
#ifndef Z
  printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n", flops(dt), FLOP_PER_POINT);
#else
  printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n", 2*flops(dt), 2*FLOP_PER_POINT);
#endif
  // printf("FLOP_PER_POINT: %i\n", FLOP_PER_POINT);

#ifdef DEBUG
  for (size_t i = 0; i < N2; ++i) {
    assert(ABS(y[i]) < DBL_MAX);
    assert(ABS(z[i]) < DBL_MAX);
  }
#endif

  summarize_c('y', y, N);
#ifdef Z
  summarize_c('z', z, N);
#endif

  cudaProfilerStop();

  write_arrays(x,y,z, u,v,w, N, TXT);
  write_arrays(x,y,z, u,v,w, N, GRID);
  // write_arrays(x,y,z, u,v,w, N, DAT);
  // write_arrays(x,y,z, u,v,w, 100, DAT);
  free(x);
  free(y);
  free(z);
  free(u);
  free(v);
  free(w);

	return 0;
}
