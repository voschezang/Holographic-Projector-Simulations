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

double flops(double t) {
  // Tera or Giga FLOP/s
  // :lscpu: 6 cores, 2x32K L1 cache, 15MB L3 cache
  // Quadro GV100: peak 7.4 TFLOPS, bandwidth 870 GB/s
  //  cores: 5120, tensor cores 640, memory: 32 GB
  // printf("fpp %i, t %0.4f, N*N %0.4f\n", FLOPS_PER_POINT, t, N*N * 1e-9);
  return 1e-12 * N * N * (double) FLOPS_PER_POINT / t;
}

void check(double complex  z) {
  double a = creal(z), b = cimag(z);
  if (isnan(a)) printf("found nan re\n");
  if (isinf(a)) printf("found inf re\n");
  if (isnan(b)) printf("found nan I\n");
  if (isinf(b)) printf("found inf I\n");
  if (isinf(a)) exit(1);
}

double memory_in_MB() {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = 2 * N * sizeof(WTYPE) + 2 * DIMS * N * sizeof(STYPE);
  return bytes * 1e-6;
}

void summarize(char name, WTYPE *x, size_t n) {
  double max_amp = 0, max_phase = 0;
  for (size_t i = 0; i < n; ++i) {
    max_amp = fmax(max_amp, cabs(x[i]));
    max_phase = fmax(max_phase , carg(x[i]));
  }
  printf("%c)  max amp: %0.6f, max phase: %0.3f\n", name, max_amp, max_phase);
}

void normalize_amp(WTYPE *x, size_t n, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < n; ++i)
    max_amp = fmax(max_amp, cabs(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < n; ++i)
      x[i] /= max_amp;

  if (log_normalize)
    for (size_t i = 0; i < n; ++i)
      if (cabs(x[i]) > 0)
        x[i] = -clog(x[i]);
}

void summarize_double(char name, double *x, size_t n) {
  double max = DBL_MIN, min = DBL_MAX;
  for (size_t i = 0; i < n; ++i) {
    max = fmax(max, x[i]);
    min = fmin(min , x[i]);
  }
  printf("%c)  range: [%0.3f , %0.3f]\n", name, min, max);
}

void print_c(WTYPE x, FILE *out) {
  check(x);
  if (cimag(x) >= 0) {
    fprintf(out, "%f+%fj", creal(x), cimag(x));
  } else {
    fprintf(out, "%f%fj", creal(x), cimag(x));
  }
}
void write_carray(char c, WTYPE *x, size_t len, FILE *out) {
  unsigned int i = 0;
  // key
  fprintf(out, "%c:", c);
  // first value
  print_c(x[0], out);
  // other values, prefixed by ','
  for (i = 1; i < N; ++i) {
    fprintf(out, ",");
    print_c(x[i], out);
  }
  // newline / return
  fprintf(out, "\n");
}

double norm3(double *u, double *v, size_t n, size_t m) {
  double x0 = v[m] - u[n], x1 = v[m+1] - u[n+1], x2 = v[m+2] - u[n+2];
  return sqrt(x0*x0 + x1*x1 + x2*x2);
}

int main() {
  printf("Init\n");
  printf("N: %i^2 = %i, \t THREADS_PER_BLOCK: %i, \t BLOCKDIM: %i, \n", N_sqrt, N, THREADS_PER_BLOCK, BLOCKDIM );
  printf("E[cores] = %0.3fk\n", BLOCKDIM * THREADS_PER_BLOCK * 1e-3);
  printf("N/thread: %i\n", N_PER_THREAD );
  printf("Memory lb: %0.2f MB\n", memory_in_MB());
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
  clock_gettime(CLOCK_MONOTONIC, &t0);
	const size_t size = N * sizeof( WTYPE );
	const size_t b_size = BATCH_SIZE * sizeof( WTYPE );

  // host
  // TODO align?
  // problem with cudaMallocManaged: cuda complex dtypes differ from normal
  // cudaError_t cudaMallocManaged(void** ptr, size_t size);
  WTYPE *x = (WTYPE *) malloc(size);
  WTYPE *y_block = (WTYPE *) malloc(BLOCKDIM * BATCH_SIZE * sizeof(WTYPE));
  WTYPE *y = (WTYPE *) malloc(size);

  STYPE *u = (STYPE *) malloc(DIMS * N * sizeof(STYPE));
  STYPE *v = (STYPE *) malloc(DIMS * N * sizeof(STYPE));

  // device
	WTYPE_cuda *d_x, *d_y, *d_y_block;
	STYPE *d_u, *d_v;
	cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) );
	cu( cudaMalloc( (void **) &d_y_block, BLOCKDIM * b_size ) );
  cu( cudaMalloc( (void **) &d_u, DIMS * N * sizeof(STYPE) ) );
  cu( cudaMalloc( (void **) &d_v, DIMS * N * sizeof(STYPE) ) );
  {
    const double width = 0.005;
    // const double dS = SCALE * 7 * 1e-6; // actually dS^1/DIMS
    const double dS = width * SCALE / N_sqrt; // actually dS^1/DIMS
    const double offset = 0.5 * N_sqrt * dS;
    // size_t sum = 0;
    for(int i = 0; i < N_sqrt; ++i) {
      for(int j = 0; j < N_sqrt; ++j) {
        size_t idx = i * N_sqrt + j;
        x[idx] = 0;
        if (i == N_sqrt / 2 && j == N_sqrt / 2) x[idx] = 1;
        if (i == N_sqrt / 3 && j == N_sqrt / 2) x[idx] = 1;
        if (i == N_sqrt * 2/3 && j == N_sqrt / 2) x[idx] = 1;
        if (i == N_sqrt / 4 && j == N_sqrt / 4) x[idx] = 1;
        if (i == N_sqrt * 3/4 && j == N_sqrt * 3/4) x[idx] = 1;
        if (i == N_sqrt / 5) x[idx] = 1;

        u[Ix(i,j,0)] = i * dS - offset;
        u[Ix(i,j,1)] = j * dS - offset;
        u[Ix(i,j,2)] = 0;
        u[Ix(i,j,1)] = 0;
        u[Ix(i,j,0)] = 0;

        v[Ix(i,j,0)] = i * dS - offset;
        v[Ix(i,j,1)] = j * dS - offset;
        v[Ix(i,j,2)] = -0.02;
      } }
  }

  summarize_double('u', u, N * DIMS);
  summarize_double('v', v, N * DIMS);

  int k = N / 2 + N_sqrt / 2;
	printf( "|x_i| = %0.2f, |y_i| = %0.2f \ti=%i\n", cabs(x[k]), cabs(y[k]), k);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  {
    double dt = (double) (t1.tv_sec - t0.tv_sec) + ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
    printf("runtime init: \t%0.3f\n", dt);
  }
  printf("loop\n");
  printf("--- --- ---   --- --- ---  --- --- --- \n");

	cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
  // TODO cp only batch part: d_v_block
	cudaMemcpy( d_v, v, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );

  // for each batch
  for (size_t i_batch = 0; i_batch < N_BATCHES; ++i_batch) {
    if (i_batch % (int) (N_BATCHES * 0.1) == 0)
      printf("batch %0.1fk\t / %0.3fk\n", i_batch * 1e-3, N_BATCHES * 1e-3);

    cu( cudaMemcpy( d_y_block, y_block, BLOCKDIM * b_size, cudaMemcpyHostToDevice ) );
    // alt, dynamic: k<<<N,M,batch_size>>>
    kernel3<<< BLOCKDIM, THREADS_PER_BLOCK >>>( d_x, d_u, d_y_block, d_v, i_batch, 1 );
    // TODO recursive reduce (e.g. for each 64 elements): use kernels for global sync
    // reduce<<< , >>>( );
    // or use thrust::reduce<>()
    cu( cudaMemcpy( y_block, d_y_block, BLOCKDIM * b_size, cudaMemcpyDeviceToHost ) );
    // for each block, add block results to global y
    // n,m = counter, not an index
    for (size_t n = 0; n < BLOCKDIM; ++n) {
      for (size_t m = 0; m < BATCH_SIZE; ++m) {
        // use full y-array in agg
        size_t i = m + i_batch * BATCH_SIZE;
        size_t i_block = m + n * BATCH_SIZE;
        if (i >= N) assert(0); // TODO rm after testing
        if (i >= N) break;
        // set y to zero
        if (n == 0 && m == 0) y[i] = 0;
        // add block results
        y[i] += y_block[i_block];
        // printf("ybl: %f\n", cabs(y_block[i_block]));
#ifdef DEBUG
        assert(cabs(y_block[i_block]) > 0);
#endif
      }
    }
  }
  normalize_amp(y, N, 0);
	free(y_block);
	cu( cudaFree( d_x ) );
	cu( cudaFree( d_y_block ) );
	cu( cudaFree( d_u ) );
	cu( cudaFree( d_v ) );

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

  summarize('x', x, N);
  summarize('y', y, N);

  printf("Save results\n");

  // TODO use csv for i/o, read python generated x
  FILE *out;
  out = fopen("tmp/out.txt", "wb");
  write_carray('x', x, N, out); free(x);
  write_carray('y', y, N, out); free(y);
  fclose(out);

	return 0;
}