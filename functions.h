#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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
  cudaMemcpyAsync( d_v, v, count, cudaMemcpyHostToDevice, stream );
  /* cudaMemcpy( d_v, v, count, cudaMemcpyHostToDevice ); */
}

template<const Direction direction>
inline void transform_batch(WTYPE *d_x, STYPE *d_u, STYPE *d_v,
                            cudaStream_t stream, double *d_y_block)
{
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
                             double *d_y_block) {
  thrust::device_ptr<double> ptr_d_y_batch(d_y_batch);
  // TODO is a reduction call for each datapoint really necessary?
  for (unsigned int m = 0; m < BATCH_SIZE; ++m) {
    // Assume two independent reductions are at least as fast as a large reduction.
    // I.e. no kernel overhead and better work distribution
    thrust::device_ptr<double> ptr(d_y_block + m * GRIDDIM);

    // launch 1x1 kernels in selected streams, which calls thrust indirectly inside that stream
    // TODO (syntax) why is here no template?
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>(), &ptr_d_y_batch[m]);
    ptr += GRIDDIM * BATCH_SIZE;
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + GRIDDIM, 0.0, thrust::plus<double>(), &ptr_d_y_batch[m + BATCH_SIZE]);
  }
}

inline void agg_batch(WTYPE *y, cudaStream_t stream,
                      WTYPE *d_y_stream, double *d_y_batch) {
  // wrapper for thrust call using streams
  kernel::zip_arrays<<< 1,1 >>>(d_y_batch, &d_y_batch[BATCH_SIZE], BATCH_SIZE, d_y_stream);

	/* cu( cudaMemcpyAsync(y, d_y_stream, BATCH_SIZE * sizeof(WTYPE), */
  /*                     cudaMemcpyDeviceToHost, stream ) ); */
}

/**
   d_x, d_u are stored in normal (non-pinned) GPU memory
   d_y, d_v are stored partially, and copied back to CPU on the fly

   Additional temporary memory:
   d_y_stream = reserved memory for each stream, containing batch result as complex doubles.
   d_y_batch  = batch results, using doubles because thrust doesn't support cuComplexDouble
   d_y_block  = block results (because blocks cannot sync), agg by thrust

   type& X is used to reference X instead of copying it (similar to a pointer *x, which would require later dereferencing)
*/
template<const Direction direction>
inline void transform(const std::vector<WTYPE>& X, WTYPE *y,
                      const std::vector<STYPE>& U, const STYPE *v) {
  /* inline void transform(const WTYPE *x, WTYPE *y, */
  /*                       const STYPE *u, const STYPE *v) { */

  // alt, using thrust */
  // type declarations in c make it more complex/verbose than e.g. python */
  // c++ has even more type syntax (e.g. `std::func<>`, multiple ways of variable init) */
  // `auto` gives simpler python-like syntax, with variable names on the right and constructors on the left */

  // TODO test if ptr conversion (thrust to *x) is flawless for large arrays */

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  thrust::device_vector<WTYPE> d_X = X;
  thrust::device_vector<STYPE> d_U = U;
  // alt syntax
  /* auto d_X = thrust::device_vector<WTYPE>(X);
     /* auto d_U = thrust::device_vector<STYPE>(U); */

  cudaStream_t streams[N_STREAMS];



#ifdef Vecs
  WTYPE *d_y_stream_ptr;
  auto d_y_stream = pinnedMallocVector<WTYPE>(&d_y_stream_ptr, N_STREAMS, BATCH_SIZE);
  double *d_y_batch_ptr;
  auto d_y_batch = pinnedMallocMatrix<double>(&d_y_batch_ptr, N_STREAMS, 2 * BATCH_SIZE);
#else
  WTYPE *d_y_stream[N_STREAMS];
  double *d_y_batch[N_STREAMS];
#endif
  double *d_y_block[N_STREAMS];
	STYPE *d_v[N_STREAMS];

  // compatibility
  const WTYPE *x = &X[0];
  const STYPE *u = &U[0];
  WTYPE *d_x = thrust::raw_pointer_cast(&d_X[0]);
  STYPE *d_u = thrust::raw_pointer_cast(&d_U[0]);

  // malloc data using pinned memory for all batches before starting streams
  // note that these mallocs are executed only once per stream
  // vector arrays are slightly more readable than sub-arrays
  for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
    // TODO use single malloc for each type?
    // manuall malloc because thrust support for pinned memory is limited

#ifndef Vecs
    cu( cudaMallocHost( (void **) &d_y_stream[i_stream],
                        BATCH_SIZE * sizeof(WTYPE) ) );
    cu( cudaMallocHost( (void **) &d_y_batch[i_stream],
                        BATCH_SIZE * 2 * sizeof(WTYPE) ) );
#endif
    cu( cudaMallocHost( (void **) &d_y_block[i_stream],
                        BATCH_SIZE * GRIDDIM * 2 * sizeof(double) ) );
    cu( cudaMallocHost( (void **) &d_v[i_stream],
                        BATCH_SIZE * DIMS * N_STREAMS * sizeof(STYPE) ) );
  }

  /* cu( cudaMallocHost( (void **) &d_y_stream, */
  /*                     N_STREAMS * BATCH_SIZE * sizeof(WTYPE) )); */
  /* cu( cudaMallocHost( (void **) &d_y_batch, */
  /*                     N_STREAMS * BATCH_SIZE * 2 * sizeof(WTYPE) )); */
  /* cu( cudaMallocHost( (void **) &d_y_block, */
  /*                     N_STREAMS * BATCH_SIZE * GRIDDIM * 2 * sizeof(double) )); */
  /* cu( cudaMallocHost( (void **) &d_v, */
  /*                     N_STREAMS * BATCH_SIZE * DIMS * N_STREAMS * sizeof(STYPE) )); */


  for (auto& stream : streams)
    cudaStreamCreate(&stream);

  printf("streams post\n");
  // assume N_BATCHES is divisible by N_STREAMS
  for (size_t i = 0; i < N_BATCHES; i+=N_STREAMS) {
    // start each distinct kernel in batches
    // TODO don't do this in case of non-uniform workloads

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
                       d_y_batch[i_stream].data,
                       d_y_block[i_stream]);
    }

    for (unsigned int i_stream = 0; i_stream < N_STREAMS; ++i_stream) {
      const size_t i_batch = i + i_stream;
      agg_batch(&y[i_batch * BATCH_SIZE],
                streams[i_stream],
                d_y_stream[i_stream],
                d_y_batch[i_stream].data);
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

  printf("destroy streams\n");
  cudaDeviceSynchronize();

  // TODO is this required?
  for (unsigned int i = 0; i < N_STREAMS; ++i)
    cudaStreamSynchronize(streams[i]);

  for (unsigned int i = 0; i < N_STREAMS; ++i)
    cudaStreamDestroy(streams[i]);

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

#ifdef Vecs
  cu( cudaFreeHost(d_y_stream_ptr) );
  cu( cudaFreeHost(d_y_batch_ptr ) );
#else
  for (auto d : d_y_stream) cu( cudaFreeHost(d) );
  for (auto d : d_y_batch)  cu( cudaFreeHost(d) );
#endif
  for (auto d : d_y_block)  cu( cudaFreeHost(d) );
  for (auto d : d_v)        cu( cudaFreeHost(d) );
  normalize_amp(y, N, 0);
}
