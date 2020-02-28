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
  // max size: 49 152
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

void summarize_c(char name, WTYPE *x, size_t len) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0;
  for (size_t i = 0; i < len; ++i) {
    max_amp = fmax(max_amp, cabs(x[i]));
    min_amp = fmin(min_amp, cabs(x[i]));
    max_phase = fmax(max_phase , carg(x[i]));
  }
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f\n", name, min_amp, max_amp, max_phase);
}

void normalize_amp(WTYPE *x, size_t len, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < len; ++i)
    max_amp = fmax(max_amp, cabs(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < len; ++i)
      x[i] /= max_amp;

  if (log_normalize)
    for (size_t i = 0; i < len; ++i)
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

void write_array(char c, STYPE *x, size_t len, FILE *out) {
  unsigned int i = 0;
  // key
  fprintf(out, "%c:", c);
  // first value
  fprintf(out, "%f", x[0]);
  // other values, prefixed by ','
  for (i = 1; i < len; ++i) {
    fprintf(out, ",%f", x[i]);
  }
  // newline / return
  fprintf(out, "\n");
}

void write_carray(char c, WTYPE *x, size_t len, FILE *out) {
  unsigned int i = 0;
  // key
  fprintf(out, "%c:", c);
  // first value
  print_c(x[0], out);
  // other values, prefixed by ','
  for (i = 1; i < len; ++i) {
    fprintf(out, ",");
    print_c(x[i], out);
  }
  // newline / return
  fprintf(out, "\n");
}

inline void transform_batch(size_t i_batch, WTYPE *x, WTYPE *y, WTYPE *y_block,
                            WTYPE_cuda *d_x, WTYPE_cuda *d_y_block,
                            STYPE *u, STYPE *v, STYPE *d_u, STYPE *d_v_block,
                            const char inverse)
{
  if (i_batch % (int) (N_BATCHES * 0.1) == 0)
    printf("batch %0.1fk\t / %0.3fk\n", i_batch * 1e-3, N_BATCHES * 1e-3);

  cudaMemcpy( d_v_block, &v[DIMS * BATCH_SIZE * i_batch],
              DIMS * BATCH_SIZE * sizeof(STYPE),
              cudaMemcpyHostToDevice );
  cudaDeviceSynchronize();
  // alt, dynamic: k<<<N,M,batch_size>>>
  kernel3<<< BLOCKDIM, THREADS_PER_BLOCK >>>( d_x, d_u, d_y_block, d_v_block,
                                              i_batch, inverse );
  // TODO recursive reduce (e.g. for each 64 elements): use kernels for global sync
  // reduce<<< , >>>( ); // or use thrust::reduce<>()

  cudaDeviceSynchronize();
  // TODO use maxBlocks, maxSize
  cu( cudaMemcpy( y_block, d_y_block,
                  BLOCKDIM * BATCH_SIZE * sizeof( WTYPE ),
                  cudaMemcpyDeviceToHost ) );
  cudaDeviceSynchronize();
  // for each block, add block results to global y
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
}

inline void transform(WTYPE *x, WTYPE *y, STYPE *u, STYPE *v, const char inverse)
{
  // TODO use M_BATCH_SIZE to async stream data


  WTYPE *y_block = (WTYPE *) malloc(BLOCKDIM * BATCH_SIZE * sizeof(WTYPE));
	WTYPE_cuda *d_x, *d_y_block;
	STYPE *d_u, *d_v_block;
	const size_t size = N * sizeof( WTYPE );
	cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) );
	cu( cudaMalloc( (void **) &d_y_block,
                  BLOCKDIM * BATCH_SIZE * sizeof( WTYPE ) ) );
  cu( cudaMalloc( (void **) &d_u, DIMS * N * sizeof(STYPE) ) );
  cu( cudaMalloc( (void **) &d_v_block, DIMS * BLOCKDIM * BATCH_SIZE * sizeof(STYPE) ) );

  // Init memory
	cudaMemcpy( d_x, x, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
  // TODO cp only batch part: d_v_block
  cudaDeviceSynchronize();

  // Loop
  for (size_t i_batch = 0; i_batch < N_BATCHES; ++i_batch) {
    transform_batch(i_batch,
                    x, y, y_block, d_x, d_y_block,
                    u, v, d_u, d_v_block, 1);
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
	cu( cudaFree( d_u ) );
	cu( cudaFree( d_y_block ) );
	free(y_block);
  normalize_amp(y, N, 0);
}
