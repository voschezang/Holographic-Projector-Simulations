//#define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <complex.h>
#include <cuComplex.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "macros.h"
#include "kernel.cu"
#include "functions.h"

/**
 * Input x,u is splitted over GPU cores/threads
 * Output y,v is streamed (send in batches).
 * The arrays y,v live on CPU, the batches d_y, d_v live on GPU.
 *
 * It is assumed that x,y,u,v all fit in GPU memory, but not necessarily in cache
 * It is assumed that z,w does not necessarily fit in GPU memory.
 *
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
  printf("\n"); printf(" N: %i^2 = %i", N_sqrt, N);
  printf("\t"); printf("\tBATCH_SIZE: %i", BATCH_SIZE);

  printf("\n"); printf(" BLOCKDIM: %i\t\t", BLOCKDIM);
  printf("\t"); printf("THREADS_PER_BLOCK: %i", THREADS_PER_BLOCK);
  printf("\t"); printf("E[cores] = %0.3fk", BLOCKDIM * THREADS_PER_BLOCK * 1e-3);
  printf("\t"); printf("\tN/thread: %i", N_PER_THREAD);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB());
  {
    double n = THREADS_PER_BLOCK * BATCH_SIZE;
    double m = n * sizeof(WTYPE_cuda) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
  assert(N_PER_THREAD > 0);
  assert(N_PER_THREAD * THREADS_PER_BLOCK * BLOCKDIM == N);
  assert(sizeof(WTYPE) == sizeof(WTYPE_cuda));

  struct timespec t0, t1, t2;
	const size_t size = N * sizeof( WTYPE );
	const size_t b_size = BATCH_SIZE * sizeof( WTYPE );
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // host
  // problem with cudaMallocManaged: cuda complex dtypes differ from normal
  WTYPE *x = (WTYPE *) malloc(size);
  WTYPE *y = (WTYPE *) malloc(size);
  WTYPE *z = (WTYPE *) malloc(size);

  STYPE *u = (STYPE *) malloc(DIMS * N * sizeof(STYPE));
  STYPE *v = (STYPE *) malloc(DIMS * N * sizeof(STYPE));
  STYPE *w = (STYPE *) malloc(DIMS * N * sizeof(STYPE));

  {
    const double width = 0.0005; // m
    const double dS = width * SCALE / N_sqrt; // actually dS^1/DIMS
    const double offset = 0.5 * N_sqrt * dS;
    for(unsigned int i = 0; i < N_sqrt; ++i) {
      for(unsigned int j = 0; j < N_sqrt; ++j) {
        size_t idx = i * N_sqrt + j;
        x[idx] = 0;
        y[idx] = 0;
        z[idx] = 0;
        if (i == N_sqrt * 1/2 && j == N_sqrt / 2) x[idx] = 1;
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

        w[Ix(i,j,0)] = i * dS - offset;
        w[Ix(i,j,1)] = j * dS - offset;
        w[Ix(i,j,2)] = 0;
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

  transform(x, y, u, v, 1);
#ifdef Z
  printf("\nSecond transform:\n");
  transform(y, z, v, w, 0);
#endif

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &t2);
  double dt = (double) (t2.tv_sec - t1.tv_sec) + ((double) (t2.tv_nsec - t1.tv_nsec)) * 1e-9;
  printf("done\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("runtime loop: \t%0.3f\n", dt);
#ifndef Z
  printf("TFLOPS:   \t%0.5f \t (%i FLOPS_PER_POINT)\n", flops(dt), FLOPS_PER_POINT);
#else
  printf("TFLOPS:   \t%0.5f \t (%i FLOPS_PER_POINT)\n", flops(dt), 2*FLOPS_PER_POINT);
#endif
  // printf("FLOPS_PER_POINT: %i\n", FLOPS_PER_POINT);

  summarize_c('y', y, N);
#ifdef Z
  summarize_c('z', z, N);
#endif

  printf("Save results\n");

  // TODO use csv for i/o, read python generated x
  FILE *out;
  out = fopen("tmp/out.txt", "wb");
  write_carray('x', x, N, out); free(x);
  write_carray('y', y, N, out); free(y);
  write_carray('z', z, N, out); free(z);
  write_array('u', u, N*DIMS, out); free(u);
  free(v);
  fclose(out);

	return 0;
}
