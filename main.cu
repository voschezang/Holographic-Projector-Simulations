//#define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <complex.h>
#include <cuComplex.h>

#include "macros.h"
#include "kernel.cu"

int main() {
  printf("Init\n");
  printf("N: %i^2 = %i, \t THREADS_PER_BLOCK: %i, \t BLOCKDIM: %i, \n", N_sqrt, N, THREADS_PER_BLOCK, BLOCKDIM );
  printf("N: %i\n", N_PER_THREAD );
  assert(N_PER_THREAD * THREADS_PER_BLOCK * BLOCKDIM == N);

  struct timespec t0, dt;
	size_t size = N * sizeof( WTYPE );

  // host
  // TODO align?
  // problem with cudaMallocManaged: cuda complex dtypes differ from normal
  // cudaError_t cudaMallocManaged(void** ptr, size_t size);
  WTYPE *x = (WTYPE *) malloc(size);
  WTYPE *y_block = (WTYPE *) malloc(BLOCKDIM * BATCH_SIZE);
  WTYPE *y = (WTYPE *) malloc(size);

  STYPE *u = (STYPE *) malloc(DIMS  * N * sizeof(WTYPE));
  STYPE *v = (STYPE *) malloc(DIMS  * N * sizeof(STYPE));
  // STYPE *u[DIMS  * N * sizeof(WTYPE)]
  // STYPE *v[DIMS  * N * sizeof(WTYPE)]

  // device
	WTYPE_cuda *d_x, *d_y, *d_y_block;
	STYPE *d_u, *d_v;
	cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) );
	cudaMalloc( (void **) &d_y, N * sizeof(WTYPE_cuda) );
	cudaMalloc( (void **) &d_y_block, BLOCKDIM * BATCH_SIZE * sizeof(WTYPE_cuda) );

	cudaMalloc( (void **) &d_u, N * sizeof(STYPE) );

  cudaMalloc( (void **) &d_v, N * sizeof(STYPE) );

  const double width = 0.005;
  // const double dS = SCALE * 7 * 1e-6; // actually dS^1/DIMS
  const double dS = width * SCALE / N_sqrt; // actually dS^1/DIMS
  const double offset = 0.5 * N_sqrt * dS;
	for(int i = 0; i < N_sqrt; ++i) {
    for(int j = 0; j < N_sqrt; ++j) {
      size_t idx = i + j * N_sqrt;
      x[idx] = 1;
      if (i == N_sqrt / 2) {
        x[idx] = 1;
      }

      u[Ix(i,j,0)] = i * dS - offset;
      u[Ix(i,j,1)] = j * dS - offset;
      u[Ix(i,j,2)] = 0;

      v[Ix(i,j,0)] = i * dS - offset;
      v[Ix(i,j,1)] = j * dS - offset;
      v[Ix(i,j,2)] = -0.02;
    }
  }

  printf("loop\n");
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // // cudaMemcpy()
  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
  //     // split x over threads
  //   }
  // }

  // for (int i = 0; i < N; i++) {
  //   kernel2<<< m, THREADS_PER_BLOCK >>>( y[i], v[i], v[i+1], v[i+2] );
  // }

	cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u, size, cudaMemcpyHostToDevice );
  // TODO mv cp inside batch loop
	cudaMemcpy( d_v, v, size, cudaMemcpyHostToDevice );

  // for each batch
  for (size_t i_batch = 0; i_batch < N_BATCHES; ++i_batch) {
    kernel3<<< BLOCKDIM, THREADS_PER_BLOCK >>>( d_x, d_u, d_y_block, d_v, i_batch );
    // TODO recursive reduce (e.g. for each 64 elements): use kernels for global sync
    // reduce<<< , >>>( );
    cudaMemcpy( y_block, d_y_block, size, cudaMemcpyDeviceToHost );

    // for each y-datapoint in current batch
    // m = counter, not and index
    for (size_t m = 0; m < BATCH_SIZE; ++m) {
      // agg block results y_block (iter over each block result)
      // note that i_tmp is reset at each batch
      size_t i = m + i_batch * BATCH_SIZE;

      // for each block result
      size_t i_block = m * BLOCKDIM; // == i_block_offset
      size_t i_block_end = i_block + BLOCKDIM;
      y[i] = y_block[i_block];
      for (i_block++; i_block < i_block_end; ++i_block)
        y[i] += y_block[i_block];

    }
  }
	// kernel1<<< m, THREADS_PER_BLOCK >>>( d_x,d_u, d_y,d_v );
	free(y_block);
	cudaMemcpy( y, d_y, size, cudaMemcpyDeviceToHost );
	cudaFree( d_x );
	// cudaFree( d_y );
	cudaFree( d_y_block );
	cudaFree( d_u );
	cudaFree( d_v );

  // end loop
  clock_gettime(CLOCK_MONOTONIC, &dt);
  double t1 = (double) (dt.tv_sec - t0.tv_sec) + ((double) (dt.tv_nsec - t0.tv_nsec)) * 1e-9;
  printf("dt: %0.3f\n", t1);

  int k = N / 2;
	printf( "|x_i| = %0.2f, |y_i| = %0.2f\n", cabs(x[k]), cabs(y[k]) );
	/* printf( "c[%d] = %d\n",N-1, c[N-1] ); */

	free(x);
	free(y);

	return 0;
}
