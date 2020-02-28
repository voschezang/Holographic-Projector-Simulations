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
#include "functions.h"
#include "kernel.cu"

/**
 * Input x,w is splitted over GPU cores/threads
 * Output y,v is streamed (send in batches)
 *
 * It is assumed that x,y,v,w all fit in GPU memory, but not necessarily in cache
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

  // printf("N: %i^2 = %i, \t THREADS_PER_BLOCK: %i, \t BLOCKDIM: %i, \n", N_sqrt, N, THREADS_PER_BLOCK, BLOCKDIM );
  // printf(": %0.2f MB/s\n", 2 * memory_in_MB() / dt);
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
  // TODO align?
  // problem with cudaMallocManaged: cuda complex dtypes differ from normal
  // cudaError_t cudaMallocManaged(void** ptr, size_t size);
  WTYPE *x = (WTYPE *) malloc(size);
  WTYPE *y_block = (WTYPE *) malloc(BLOCKDIM * BATCH_SIZE * sizeof(WTYPE));
  WTYPE *y = (WTYPE *) malloc(size);
  // WTYPE *y_batch = (WTYPE *) malloc(G_BATCH_SIZE * sizeof( WTYPE ));

  STYPE *u = (STYPE *) malloc(DIMS * N * sizeof(STYPE));
  STYPE *v = (STYPE *) malloc(DIMS * N * sizeof(STYPE));
  // STYPE *v_block = (STYPE *) malloc(DIMS * BLOCKDIM * Y_BATCH_SIZE * sizeof(STYPE));
  // STYPE *v_batch = (STYPE *) malloc(DIMS * BLOCKDIM * G_BATCH_SIZE * sizeof(STYPE));

  // device
	WTYPE_cuda *d_x, *d_y, *d_y_block;
	STYPE *d_u, *d_v;
	cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) );
	cu( cudaMalloc( (void **) &d_y_block, BLOCKDIM * b_size ) );
  cu( cudaMalloc( (void **) &d_u, DIMS * N * sizeof(STYPE) ) );
  cu( cudaMalloc( (void **) &d_v, DIMS * N * sizeof(STYPE) ) );
  {
    const double width = 0.0005; // m
    // const double dS = SCALE * 7 * 1e-6; // actually dS^1/DIMS
    const double dS = width * SCALE / N_sqrt; // actually dS^1/DIMS
    const double offset = 0.5 * N_sqrt * dS;
    // size_t sum = 0;
    // TODO use #pragma unroll?
    for(unsigned int i = 0; i < N_sqrt; ++i) {
      for(unsigned int j = 0; j < N_sqrt; ++j) {
        size_t idx = i * N_sqrt + j;
        x[idx] = 0;
        y[idx] = 0;
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
      } }
  }

  summarize_double('u', u, N * DIMS);
  summarize_double('v', v, N * DIMS);

  int k = N / 2 + N_sqrt / 2;
	printf( "|x_i| = %0.2f, |y_i| = %0.2f \ti=%i\n", cabs(x[k]), cabs(y[k]), k);

  // TODO use M_BATCH_SIZE to async stream data
	cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
  // TODO cp only batch part: d_v_block
	cudaMemcpy( d_v, v, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &t1);
  {
    double dt = (double) (t1.tv_sec - t0.tv_sec) + ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
    printf("runtime init: \t%0.3f\n", dt);
  }
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");

  // for each batch
  for (size_t i_batch = 0; i_batch < N_BATCHES; ++i_batch) {
    if (i_batch % (int) (N_BATCHES * 0.1) == 0)
      printf("batch %0.1fk\t / %0.3fk\n", i_batch * 1e-3, N_BATCHES * 1e-3);

    cu( cudaMemcpy( d_y_block, y_block, BLOCKDIM * b_size, cudaMemcpyHostToDevice ) );
    cudaDeviceSynchronize();
    // alt, dynamic: k<<<N,M,batch_size>>>
    kernel3<<< BLOCKDIM, THREADS_PER_BLOCK >>>( d_x, d_u, d_y_block, d_v, i_batch, 1 );
    // TODO recursive reduce (e.g. for each 64 elements): use kernels for global sync
    // reduce<<< , >>>( ); // or use thrust::reduce<>()

    cudaDeviceSynchronize();
    // TODO use maxBlocks, maxSize
    cu( cudaMemcpy( y_block, d_y_block, BLOCKDIM * b_size, cudaMemcpyDeviceToHost ) );
    // for each block, add block results to global y
    // n,m = counter, not an index
    for (size_t n = 0; n < BLOCKDIM; ++n) {
      for (size_t m = 0; m < BATCH_SIZE; ++m) {
        // use full y-array in agg
        size_t i = m + i_batch * BATCH_SIZE;
        size_t i_block = m + n * BATCH_SIZE;
        // add block results
        y[i] += y_block[i_block];

#ifdef DEBUG
        if (i >= N) assert(0); // TODO rm after testing
        if (i >= N) break;
#endif
      }
    }
#ifdef DEBUG
    for (size_t m = 0; m < BATCH_SIZE; ++m) {
      // use full y-array in agg
      size_t i = m + i_batch * BATCH_SIZE;
      assert(cabs(y[i]) > 0);
    }
#endif
  }
#ifdef DEBUG
  for(size_t i = 0; i < N; ++i) assert(cabs(y[i]) > 0);
#endif

	cu( cudaFree( d_x ) );
	cu( cudaFree( d_y_block ) );
	cu( cudaFree( d_u ) );
	cu( cudaFree( d_v ) );
	free(y_block);

  normalize_amp(y, N, 0);

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &t2);
  double dt = (double) (t2.tv_sec - t1.tv_sec) + ((double) (t2.tv_nsec - t1.tv_nsec)) * 1e-9;
  printf("done\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("runtime loop: \t%0.3f\n", dt);
  printf("TFLOPS:   \t%0.5f \t (%i FLOPS_PER_POINT)\n", flops(dt), FLOPS_PER_POINT);
  // printf("FLOPS_PER_POINT: %i\n", FLOPS_PER_POINT);

	printf( "|x_i| = %0.2f, |y_i| = %0.2f \ti=%i\n", cabs(x[k]), cabs(y[k]),k );
  k = 3;
	printf( "|x_i| = %0.2f, |y_i| = %0.2f \ti=%i\n", cabs(x[k]), cabs(y[k]),k );

  // summarize_c('x', x, N);
  summarize_c('y', y, N);

  printf("Save results\n");

  // TODO use csv for i/o, read python generated x
  FILE *out;
  out = fopen("tmp/out.txt", "wb");
  write_carray('x', x, N, out); free(x);
  write_carray('y', y, N, out); free(y);
  write_array('u', u, N*DIMS, out); free(u);
  free(v);
  fclose(out);

	return 0;
}
