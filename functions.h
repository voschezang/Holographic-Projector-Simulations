#include <assert.h>
/* #include <complex.h> */
/* #include <cuComplex.h> */
/* #include <float.h> */
/* #include <limits.h> */
/* #include <math.h> */
#include <stdio.h>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "macros.h"
#include "kernel.cu"

enum FileType {TXT, DAT, GRID};

double flops(double runtime) {
  // Tera or Giga FLOP/s
  // :lscpu: 6 cores, 2x32K L1 cache, 15MB L3 cache
  // Quadro GV100: peak 7.4 TFLOPS (dp), 14.8 (sp), 59.3 (int)
  // bandwidth 870 GB/s
  //  cores: 5120, tensor cores 640, memory: 32 GB
  // generation Volta, compute capability 7.0
  // max size: 49 152
  // printf("fpp %i, t %0.4f, N*N %0.4f\n", FLOPS_PER_POINT, t, N*N * 1e-9);
  return 1e-12 * N * N * (double) FLOP_PER_POINT / runtime;
}

void check(WTYPE  z) {
  /* double a = creal(z), b = cimag(z); */
  if (isnan(z.x)) printf("found nan re\n");
  if (isinf(z.x)) printf("found inf re\n");
  if (isnan(z.y)) printf("found nan I\n");
  if (isinf(z.y)) printf("found inf I\n");
  if (isinf(z.x)) exit(1);
}

void check_params() {
#if (N_STREAMS < 1)
  printf("Invalid param: N_STREAMS < 1\n"); assert(0);
#elif (BATCHES_PER_STREAM < 1)
  printf("Invalid param: BATCHES_PER_STREAM < 1\n"); assert(0);
#elif (N_STREAMS * BATCHES_PER_STREAM != N_BATCHES)
  printf("Invalid param: incompatible N_STREAMS and N\n"); assert(0);
#endif
  assert(N_PER_THREAD > 0);
  assert(N == N_STREAMS * STREAM_SIZE);
  assert(N == BATCH_SIZE * BATCHES_PER_STREAM * N_STREAMS);
  assert(N_PER_THREAD * THREADS_PER_BLOCK * BLOCKDIM == N);
  assert(sizeof(WTYPE) == sizeof(WTYPE_cuda));
}

double memory_in_MB() {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = 2 * N * sizeof(WTYPE) + 2 * DIMS * N * sizeof(STYPE);
  return bytes * 1e-6;
}

void summarize_c(char name, WTYPE *x, size_t len) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0, sum = 0;
  for (size_t i = 0; i < len; ++i) {
    max_amp = fmax(max_amp, ABS(x[i]));
    min_amp = fmin(min_amp, ABS(x[i]));
    max_phase = fmax(max_phase , angle(x[i]));
    sum += ABS(x[i]);
  }
  double mean = sum / (double) N;
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f, mean: %f\n", name, min_amp, max_amp, max_phase, mean);
}

void normalize_amp(WTYPE *x, size_t len, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < len; ++i)
    max_amp = fmax(max_amp, ABS(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < len; ++i) {
      x[i].x /= max_amp;
      x[i].y /= max_amp;
    }

  if (log_normalize)
    for (size_t i = 0; i < len; ++i) {
      if (x[i].x > 0) x[i].x = -log(x[i].x);
      if (x[i].y > 0) x[i].y = -log(x[i].y);
    }
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
  if (x.y >= 0) {
    fprintf(out, "%f+%fj", x.x, x.y);
  } else {
    fprintf(out, "%f%fj", x.x, x.y);
  }
}


void write_array(char c, STYPE *x, size_t len, FILE *out, char print_key) {
  // key
  if (print_key == 1)
    fprintf(out, "%c:", c);

  // first value
  fprintf(out, "%f", x[0]);
  // other values, prefixed by ','
  for (size_t i = 1; i < len; ++i) {
    fprintf(out, ",%f", x[i]);
  }
  // newline / return
  fprintf(out, "\n");
}

void write_carray(char c, WTYPE *x, size_t len, FILE *out) {
  // key
  fprintf(out, "%c:", c);
  // first value
  print_c(x[0], out);
  // other values, prefixed by ','
  for (size_t i = 1; i < len; ++i) {
    fprintf(out, ",");
    print_c(x[i], out);
  }
  // newline / return
  fprintf(out, "\n");
}

void write_dot(char name, WTYPE *x, STYPE *u, size_t len) {
  char fn[] = "tmp/out-x.dat";
  fn[8] = name;
  FILE *out = fopen(fn, "wb");
  fprintf(out, "#dim 1 - dim 2 - dim 3 - Amplitude - Phase\n");
  for (size_t i = 0; i < len; ++i) {
    size_t j = i * DIMS;
    fprintf(out, "%f %f %f %f %f\n", u[j], u[j+1], u[j+2], ABS(x[i]), angle(x[i]));
  }
  fclose(out);
}

void write_arrays(WTYPE *x, WTYPE *y, WTYPE *z, STYPE *u, STYPE *v, STYPE *w, size_t len, enum FileType type) {
  printf("Save results as ");
  if (type == TXT) {
    printf(".txt\n");
    // TODO use csv for i/o, read python generated x
    FILE *out = fopen("tmp/out.txt", "wb");
    write_carray('x', x, len, out);
    write_carray('y', y, len, out);
    write_carray('z', z, len, out);
    // ignore 'u'
    write_array('v', v, len*DIMS, out, 1);
    write_array('w', w, len*DIMS, out, 1);
    fclose(out);
  }
  else if (type == DAT) {
    printf(".dat\n");
    write_dot('x', x, u, len);
    write_dot('y', y, v, len);
    write_dot('z', z, w, len);
  }
  else if (type == GRID) {
    printf(".grid\n");
    FILE *out = fopen("tmp/out-y.grid", "wb");
    assert(len == N2);
    double img[N2];
    // ignore borders
    /* img[I_(0,0)] = 0; */
    /* img[I_(0,N_sqrt-1)] = 0; */
    /* img[I_(N_sqrt-1, 0)] = 0; */
    /* img[I_(N_sqrt-1, N_sqrt-1)] = 0; */

    /* for (size_t i = 0; i < N2; ++i) */
    /*   img[i] = 12; */
    for (size_t i = 1; i < N_sqrt-1; ++i) {
      for (size_t j = 1; j < N_sqrt-1; ++j) {
        /* assert(img[I_(i,j)] == 12); */
        /* img[I_(i,j)] = 31.0; */
        /* printf("i: %i, j: %i, img[%4i]: %f\n", i,j, I_(i,j), img[I_(i,j)]); */
        img[I_(i,j)] = ABS(y[I_(i,j)]);// + ABS(y[I_(i+1,j)]); // + y[I_(i-1,j)] + y[I_(i,j+1)] + y[I_(i,j-1)];
#ifdef DEBUG
        assert(ABS(y[I_(i,j)]) < DBL_MAX);
#endif
        /* img[i][j] *= 1./5.; */
        /* printf("i: %i, j: %i, img[%4i]: %f\n", i,j, I_(i,j), img[I_(i,j)]); */
      }
    }
    /* for (size_t i = 1; i < N_sqrt-1; ++i) */
    /*   for (size_t j = 1; j < N_sqrt-1; ++j) { */
    /*     printf("i: %i, j: %i, img[%4i]: %f\n", i,j, I_(i,j), img[I_(i,j)]); */
    /*     assert(img[I_(i,j)] == 31.0); */
    /*   } */

    write_array('y', img, N2, out, 0);
    fclose(out);
  }
}


inline void transform_batch(const size_t i_batch, const WTYPE *x, WTYPE *y,
                            WTYPE_cuda *d_x,
                            const STYPE *u, const STYPE *v,
                            STYPE *d_u, STYPE *d_v,
                            const char direction, cudaStream_t stream,
                            WTYPE_cuda *d_y_stream,
                            /* WTYPE *y_block, */
                            /* WTYPE_cuda */
                            thrust::device_vector<double> d_y_batch,
                            double *d_y_block
                            /* , double *d_y_batch_ptr */
                            )
{
  // TODO most data transfer is actually (BLOCKDIM * STREAM_SIZE),
  // thus, aggregate on gpu
  printf("batch %3i kernel \n", i_batch);
  // start kernel
#ifdef MEMCPY_ASYNC
  kernel3<<< BLOCKDIM, THREADS_PER_BLOCK, 0, stream >>>
    (d_x, d_u, d_y_block, d_v, i_batch, direction );
  // TODO launch thrust agg kernel
#else
  // alt, dynamic: k<<<N,M,batch_size>>>
  kernel3<<< BLOCKDIM, THREADS_PER_BLOCK >>>
    ( d_x, d_u, d_y_block, d_v, i_batch, direction );
#endif

  // TODO recursive reduce (e.g. for each 64 elements): use kernels for global sync
  // reduce<<< , >>>( ); // or use thrust::reduce<>()

  // TODO use maxBlocks, maxSize
  // TODO use cudaMemcpyAsync? otherwise this is a barrier
  printf("batch %3i memcpy \n", i_batch);

  /* thrust::device_vector<double> d_y_batch(BATCH_SIZE * 2); */
  for (unsigned int m = 0; m < BATCH_SIZE; ++m) {

    /* double *d_y_block; */
    /* cu( cudaMalloc( (void **) &d_y_block, BLOCKDIM * BATCH_SIZE * sizeof(double) ) ); */

    // assume two independent reductions are faster or equal to a large reduction
    thrust::device_ptr<double> ptr(d_y_block + m * BLOCKDIM);
    d_y_batch[m] = thrust::reduce(ptr, ptr + BLOCKDIM, 0.0, thrust::plus<double>());
    ptr += BLOCKDIM * BATCH_SIZE;
    d_y_batch[m + BATCH_SIZE] = thrust::reduce(ptr, ptr + BLOCKDIM, 0.0, thrust::plus<double>());
  }

  // TODO do cpu aggregation in a separate loop?, make thrust calls async

  const size_t i = i_batch * BATCH_SIZE;
  double *ptr = thrust::raw_pointer_cast(d_y_batch.data());
  zip_arrays<<< 1,1 >>>(ptr, &ptr[BATCH_SIZE], BATCH_SIZE, d_y_stream);
  // TODO if async..
	cu( cudaMemcpy(&y[i], d_y_stream, BATCH_SIZE * sizeof(WTYPE_cuda), cudaMemcpyDeviceToHost ) );

  printf("batch %3i done \n", i_batch);
}

inline void transform(const WTYPE *x, WTYPE *y,
                      const STYPE *u, const STYPE *v,
                      const char direction) {
  // TODO use M_BATCH_SIZE to async stream data
  /**
   d_y_stream = reserved memory consisting of batch for each stream
   d_y_batch  = batch results, using doubles because thrust doesn't support cuComplexDouble
   d_y_block  = block results (because blocks cannot sync), agg by thrust
   */
  WTYPE_cuda *d_y_stream[N_STREAMS];
  thrust::device_vector<double> d_y_batch[N_STREAMS];
  double *d_y_block[N_STREAMS];

  cudaStream_t streams[N_STREAMS];
	WTYPE_cuda *d_x;
	STYPE *d_u, *d_v;

  // Malloc memory
  cu( cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) ) );
  cu( cudaMalloc( (void **) &d_u, DIMS * N * sizeof(STYPE) ) );

  // TODO cp d_v in batches (size = BATCH_SIZE * N_STREAMS)
/* #ifdef PINNED_MEM */
/*   // use pinned memory for data that is updated for every batch */
/*   // TODO decrease number of copies (i.e. make copy size independent of batch size) */
/*   cu( cudaMallocHost( (void **) &d_v, DIMS * N * sizeof(STYPE) ) ); */
/* #else */
  cu( cudaMalloc( (void **) &d_v, DIMS * N * sizeof(STYPE) ) );
/* #endif */

  /* STYPE *d_v; */
  /* cu( cudaMalloc( (void **) &d_v, DIMS * N * sizeof(STYPE) ) ); */
	/* cudaMemcpy( d_v, v, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice ); */

  // Loop
  printf("streams pre\n");

  // malloc data for all batches before starting streams

  // TODO why not a single array for d_y_batch, d_y_block?
  // TODO only if forcing affinity in malloc to be spreaded?
  //   or it this done auto through the order of malloc calls?
  // TODO ask roel?
  // note that vector of arrays can be more readable
  for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
    d_y_batch[i_stream] = thrust::device_vector<double>(BATCH_SIZE * 2);
#ifdef MEMCPY_ASYNC
    cu( cudaMallocHost( (void **) &d_y_stream[i_stream],
                    BATCH_SIZE * sizeof(WTYPE_cuda) ) );
    cu( cudaMallocHost( (void **) &d_y_block[i_stream],
                    BATCH_SIZE * BLOCKDIM * 2 * sizeof(double) ) );
#else
    cu( cudaMalloc( (void **) &d_y_stream[i_stream],
                    BATCH_SIZE * sizeof(WTYPE_cuda) ) );
    cu( cudaMalloc( (void **) &d_y_block[i_stream],
                    BATCH_SIZE * BLOCKDIM * 2 * sizeof(double) ) );
#endif
  }


  // Init memory
	cudaMemcpy( d_x, x, N * sizeof(WTYPE), cudaMemcpyHostToDevice );
	cudaMemcpy( d_u, u, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
#ifdef MEMCPY_ASYNC
  // TODO 
	cudaMemcpy( d_v, v, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
  // note that N_BATCHES != number of streams
  for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
    /* const size_t i = i_stream * STREAM_SIZE * DIMS; */
    printf("stream %i create \n", i_stream);
    cudaStreamCreate(&streams[i_stream]);
    printf("stream %i cpy init \n", i_stream);
    // TODO cp d_v in batches
    /* cudaMemcpyAsync( &d_v[i], &v[i], DIMS * STREAM_SIZE * sizeof(STYPE), */
    /*                cudaMemcpyHostToDevice, streams[i_stream] ); */
    printf("stream %i nxt \n", i_stream);
  }
#else
	cudaMemcpy( d_v, v, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice );
#endif


  printf("streams post\n");
  // in case of heterogeneous computing, start 4 batches async on gpu, then agg them on cpu, and repeat
  // in case of pure gpu computing, send single batch to all streams, and repeat (instead filling a single stream with many batches at time)
  for (size_t i_batch = 0; i_batch < N_BATCHES; i_batch+=N_STREAMS) {
    for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
      if (N_BATCHES > 10 && i_batch % (int) (N_BATCHES / 10) == 0)
        printf("batch %0.1fk\t / %0.3fk\n", i_batch * 1e-3, N_BATCHES * 1e-3);

      // TODO split up function per kernel call?
      transform_batch(i_batch,
                      x, y, d_x,
                      u, v, d_u, d_v,
                      direction, streams[i_stream],
                      /* y_block[i_stream], */
                      d_y_stream[i_stream],
                      d_y_batch[i_stream],
                      d_y_block[i_stream]
                      /* , thrust::raw_pointer_cast(d_y_batch[i_stream].data()) */
                      );
#ifdef DEBUG
      cudaDeviceSynchronize();
      /* printf("test > 0 ..\n"); */
      for (size_t m = 0; m < BATCH_SIZE; ++m) {
        // use full y-array in agg
        size_t i = m + i_batch * BATCH_SIZE;
        assert(ABS(y[i]) > 0);
      }
#endif
      i_batch += 1;
    }
    i_batch -= N_STREAMS;
  }

#ifdef MEMCPY_ASYNC
  printf("destroy streams\n");
  cudaDeviceSynchronize();
  for (unsigned int i = 0; i < N_STREAMS; ++i)
    cudaStreamDestroy(streams[i]);
#endif

#ifdef DEBUG
  /* for (size_t i_batch = 0; i_batch < N_BATCHES; i_batch+=1) { */
  for (size_t i_batch = 0; i_batch < 2; i_batch+=1) {
    /* assert(y[i_batch] != 10); */
    /* printf("y[%i]: \n", i_batch * BATCH_SIZE); */
    /* assert(y[i_batch * BATCH_SIZE] != 10); */
    assert(ABS(y[i_batch * BATCH_SIZE]) > 0);
  }

  for (size_t i_batch = 0; i_batch < N_BATCHES; i_batch+=1)
      for (unsigned int i = 0; i < BATCH_SIZE; ++i)
        assert(ABS(y[i + i_batch * BATCH_SIZE]) > 0);

  for(size_t i = 0; i < N; ++i) assert(ABS(y[i]) > 0);
#endif

  // implicit sync?

	cu( cudaFree( d_x ) );
	cu( cudaFree( d_u ) );
	cu( cudaFree( d_v ) ); // TODO free host for d_v_batch
  for (unsigned int i = 0; i < N_STREAMS; ++i) {
#ifdef MEMCPY_ASYNC
    cu( cudaFreeHost( d_y_stream[i] ) );
    cu( cudaFreeHost( d_y_block[i] ) );
#else
    cu( cudaFree( d_y_stream[i] ) );
    cu( cudaFree( d_y_block[i] ) );
#endif
  }
  /* cudaDeviceReset(); // TODO check if used correctly */
  normalize_amp(y, N, 0);
}
