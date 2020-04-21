#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "macros.h"
#include "util.h"
#include "kernel.cu"
#include "superposition.cu"


inline void cp_batch_data(const STYPE *v, STYPE *d_v, const size_t count, cudaStream_t stream) {
  // copy "v[i]" for i in batch, where v are the spatial positions belonging to the target datapoints y
#ifdef MEMCPY_ASYNC
  cudaMemcpyAsync( d_v, v, count, cudaMemcpyHostToDevice, stream );
#else
  cudaMemcpy( d_v, v, count, cudaMemcpyHostToDevice );
#endif
}

template<const Direction direction>
inline void transform_batch(WTYPE *d_x, STYPE *d_u, STYPE *d_v,
                            cudaStream_t stream, double *d_y_block)
{
  // TODO consider
  // for (auto i = 0u;;)
  for (unsigned int i = 0; i < KERNELS_PER_BATCH; ++i) {
    // d_y_block : BATCH_SIZE x GRIDDIM x 2

    const unsigned int j = i * GRIDDIM * KERNEL_BATCH_SIZE; // * 2
    const unsigned int k = i * KERNEL_BATCH_SIZE;
    superposition::per_block<direction><<< GRIDDIM, BLOCKDIM, 0, stream >>>
      (d_x, d_u, &d_y_block[j], &d_v[k * DIMS] );
  }
}

inline void agg_batch_blocks(cudaStream_t stream,
                             double *d_y_batch,
                             double *d_y_block)
{
  // TODO use maxBlocks, maxSize?
  thrust::device_ptr<double> ptr_d_y_batch(d_y_batch);
  // TODO is a reduction call for each datapoint really necessary?
  for (unsigned int m = 0; m < BATCH_SIZE; ++m) {
    // Assume two independent reductions are at least as fast as a large reduction.
    // I.e. no kernel overhead and better work distribution
    thrust::device_ptr<double> ptr(d_y_block + m * GRIDDIM);

    // launch 1x1 kernels in selected streams, which calls thrust indirectly inside that stream
#ifdef MEMCPY_ASYNC
    // TODO why is here no template?
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>(), &ptr_d_y_batch[m]);
    ptr += GRIDDIM * BATCH_SIZE;
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>(), &ptr_d_y_batch[m + BATCH_SIZE]);
#else
    ptr_d_y_batch[m] = thrust::reduce(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>());
    ptr += GRIDDIM * BATCH_SIZE;
    ptr_d_y_batch[m + BATCH_SIZE] = thrust::reduce(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>());
#endif
  }
}

inline void agg_batch(WTYPE *y,
                      cudaStream_t stream,
                      WTYPE *d_y_stream,
                      double *d_y_batch) {
  // wrapper for thrust call using streams
  kernel::zip_arrays<<< 1,1 >>>(d_y_batch, &d_y_batch[BATCH_SIZE], BATCH_SIZE, d_y_stream);

#ifdef MEMCPY_ASYNC
	cu( cudaMemcpyAsync(y, d_y_stream, BATCH_SIZE * sizeof(WTYPE),
                      cudaMemcpyDeviceToHost, stream ) );
#else
	cu( cudaMemcpy(y, d_y_stream, BATCH_SIZE * sizeof(WTYPE),
                 cudaMemcpyDeviceToHost ) );
#endif
}

template<const Direction direction>
inline void transform(const WTYPE *x, WTYPE *y,
                      const STYPE *u, const STYPE *v) {
  /**
   d_y_stream = reserved memory for each stream, containing batch result as complex doubles.
   d_y_batch  = batch results, using doubles because thrust doesn't support cuComplexDouble
   d_y_block  = block results (because blocks cannot sync), agg by thrust
   */
  WTYPE *d_y_stream[N_STREAMS];
  double *d_y_batch[N_STREAMS];
  double *d_y_block[N_STREAMS];
  cudaStream_t streams[N_STREAMS];
	STYPE *d_v[N_STREAMS], *d_u;
	WTYPE *d_x;

  // Malloc memory, no need to used pinned (page-locked) memory for input
  cu( cudaMalloc( (void **) &d_x, N * sizeof(WTYPE) ) );
  cu( cudaMalloc( (void **) &d_u, DIMS * N * sizeof(STYPE) ) );
  // alt
  //
  /*   auto x = thrust::host_vector<WTYPE>(size); */
  /*   auto d_x = thrust::device_vector<WTYPE>(x.begin(), x.end()); */
  /*   auto u = thrust::host_vector<STYPE>(size); */
  /*   auto d_u = thrust::device_vector<STYPE>(u.begin(), u.end()); */
  /*   thrust::device_vector<STYPE> d_u = u;
  /*   /\* auto d_v = thrust::device_vector<STYPE>(size); *\/ */
  /* } */

  // malloc data for all batches before starting streams
  // vector arrays are slightly more readable than sub-arrays
  for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
    // TODO use single malloc for each type?
#ifdef MEMCPY_ASYNC
    cu( cudaMallocHost( (void **) &d_y_stream[i_stream],
                    BATCH_SIZE * sizeof(WTYPE) ) );
    cu( cudaMallocHost( (void **) &d_y_batch[i_stream],
                    BATCH_SIZE * 2 * sizeof(WTYPE) ) );
    cu( cudaMallocHost( (void **) &d_y_block[i_stream],
                    BATCH_SIZE * GRIDDIM * 2 * sizeof(double) ) );
    cu( cudaMallocHost( (void **) &d_v[i_stream],
                    BATCH_SIZE * DIMS * N_STREAMS * sizeof(STYPE) ) );
#else
    cu( cudaMalloc( (void **) &d_y_stream[i_stream],
                    BATCH_SIZE * sizeof(WTYPE) ) );
    cu( cudaMalloc( (void **) &d_y_batch[i_stream],
                    BATCH_SIZE * 2 * sizeof(WTYPE) ) );
    cu( cudaMalloc( (void **) &d_y_block[i_stream],
                    BATCH_SIZE * GRIDDIM * 2 * sizeof(double) ) );
    cu( cudaMalloc( (void **) &d_v[i_stream],
                    BATCH_SIZE * DIMS * N_STREAMS * sizeof(STYPE) ) );
#endif
  }


  // Init memory
	cu ( cudaMemcpy( d_x, x, N * sizeof(WTYPE), cudaMemcpyHostToDevice ) );
	cu ( cudaMemcpy( d_u, u, DIMS * N * sizeof(STYPE), cudaMemcpyHostToDevice ) );

  /* { */
  /*   // alt, using thrust */
  /*   // type declarations in c make it more complex/verbose than e.g. python */
  /*   // c++ has even more type syntax (e.g. `std::func<>`, multiple ways of variable init) */
  /*   // `auto` gives simpler python-like syntax, with variable names on the right and constructors on the left */

  /*   // TODO test if ptr conversion (thrust to *x) is flawless for large arrays */

  /*   auto size = 100; */
  /*   auto x = thrust::host_vector<WTYPE>(size); */
  /*   auto y = thrust::host_vector<WTYPE>(size); */
  /*   auto d_x = thrust::device_vector<WTYPE>(x.begin(), x.end()); */
  /*   auto d_v = thrust::device_vector<WTYPE>(size); */

  /*   auto u = thrust::host_vector<STYPE>(size); */
  /*   auto v = thrust::host_vector<STYPE>(size); */
  /*   auto d_u = thrust::device_vector<STYPE>(u.begin(), u.end()); */
  /*   /\* auto d_v = thrust::device_vector<STYPE>(size); *\/ */
  /* } */

#ifdef MEMCPY_ASYNC
  // note that N_BATCHES != number of streams
  for (auto& stream : streams)
    cudaStreamCreate(&stream);

  // alt
  /* for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) { */
  /*   /\* const size_t i = i_stream * STREAM_SIZE * DIMS; *\/ */
  /*   printf("stream %i create \n", i_stream); */
  /*   /\* cudaStreamCreate(&streams[i_stream]); *\/ */
  /*   printf("stream %i cpy init \n", i_stream); */
  /*   printf("stream %i nxt \n", i_stream); */
  /* } */
#else
  streams[0] = 0; // default stream
#endif

  printf("streams post\n");
  // assume N_BATCHES is divisible by N_STREAMS
  for (size_t i = 0; i < N_BATCHES; i+=N_STREAMS) {
    // start each distinct kernel in batches
    // TODO don't do this in case of non-uniform workloads

    // TODO use single stream forloop
    for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
      const size_t i_batch = i + i_stream;
      if (N_BATCHES > 10 && i_batch % (int) (N_BATCHES / 10) == 0)
        printf("batch %0.1fk\t / %0.3fk\n", i_batch * 1e-3, N_BATCHES * 1e-3);

      /* printf("batch %3i stream %3i\n", i_batch, i_stream); */
      cp_batch_data(&v[i_batch * BATCH_SIZE * DIMS], d_v[i_stream],
                    DIMS * BATCH_SIZE * sizeof(STYPE), streams[i_stream]);

      transform_batch<direction>(d_x, d_u, d_v[i_stream],
                      streams[i_stream], d_y_block[i_stream]
                      );
    }

    // do aggregations in separate stream-loops; yielding a ~2.5x speedup
    for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
      /* const size_t i_batch = i + i_stream; */
      agg_batch_blocks(streams[i_stream],
                       d_y_batch[i_stream],
                       d_y_block[i_stream]);
    }

    for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
      const size_t i_batch = i + i_stream;
      agg_batch(&y[i_batch * BATCH_SIZE],
                streams[i_stream],
                d_y_stream[i_stream],
                d_y_batch[i_stream]);
#ifdef DEBUG
      cudaDeviceSynchronize();
      for (size_t m = 0; m < BATCH_SIZE; ++m) {
        // use full y-array in agg
        size_t i = m + i_batch * BATCH_SIZE;
        assert(cuCabs(y[i]) > 0);
      }
#endif
    }
  }

#ifdef MEMCPY_ASYNC
  printf("destroy streams\n");
  cudaDeviceSynchronize();

  // TODO is this required?
  for (unsigned int i = 0; i < N_STREAMS; ++i)
    cudaStreamSynchronize(streams[i]);

  for (unsigned int i = 0; i < N_STREAMS; ++i)
    cudaStreamDestroy(streams[i]);
#endif

#ifdef DEBUG
  /* for (size_t i_batch = 0; i_batch < N_BATCHES; i_batch+=1) { */
  for (size_t i_batch = 0; i_batch < 2; i_batch+=1) {
    /* assert(y[i_batch] != 10); */
    /* printf("y[%i]: \n", i_batch * BATCH_SIZE); */
    /* assert(y[i_batch * BATCH_SIZE] != 10); */
    assert(cuCabs(y[i_batch * BATCH_SIZE]) > 0);
  }

  for (size_t i_batch = 0; i_batch < N_BATCHES; i_batch+=1)
      for (unsigned int i = 0; i < BATCH_SIZE; ++i)
        assert(cuCabs(y[i + i_batch * BATCH_SIZE]) > 0);

  for(size_t i = 0; i < N; ++i) assert(cuCabs(y[i]) > 0);
#endif

  // implicit sync?

	cu( cudaFree( d_x ) );
	cu( cudaFree( d_u ) );
	/* cu( cudaFree( d_v ) ); // TODO free host for d_v_batch */
  for (unsigned int i = 0; i < N_STREAMS; ++i) {
#ifdef MEMCPY_ASYNC
    cu( cudaFreeHost( d_y_stream[i] ) );
    cu( cudaFreeHost( d_y_block[i] ) );
    cu( cudaFreeHost( d_v[i] ) );
#else
    cu( cudaFree( d_y_stream[i] ) );
    cu( cudaFree( d_y_block[i] ) );
    cu( cudaFree( d_v[i] ) );
#endif
  }
  /* cudaDeviceReset(); // TODO check if used correctly */
  normalize_amp(y, N, 0);
}
