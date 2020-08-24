#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "macros.h"
#include "hyper_params.h"
#include "util.h"
#include "init.h"
#include "kernel.cu"
#include "superposition.cu"

// host superposition functions

template<typename T = double>
inline void cp_batch_data_to_device(const T *v, T *v_pinned, CUDAVector<T> d_v, cudaStream_t stream) {
  // any host memory involved in async/overlapping data transfers must be page-locked
  for (size_t i = 0; i < d_v.size; ++i)
    v_pinned[i] = v[i];

  cu ( cudaMemcpyAsync( d_v.data, v_pinned, d_v.size * sizeof(T),
                        cudaMemcpyHostToDevice, stream ) );
}

template<typename T = WAVE>
inline void cp_batch_data_to_host(const T *d_y, T *y_pinned, const size_t len, cudaStream_t stream) {
  // any host memory involved in async/overlapping data transfers must be page-locked
  cu ( cudaMemcpyAsync( y_pinned, d_y, len * sizeof(T),
                        cudaMemcpyDeviceToHost, stream ) );
}

#define SuperpositionPerBlock(blockDim_y) {                             \
    assert(blockDim_x * blockDim_y <= 1024);                            \
    superposition::per_block<direction, blockDim_x, blockDim_y, algorithm, shared_memory> \
      <<< gridDim, blockDim, 0, stream >>>                              \
      (N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }

#define SuperpositionPerBlockHelper(blockDim_x) {                       \
    superposition_per_block_helper<direction, blockDim_x, algorithm, shared_memory> \
      (gridDim, blockDim, stream,                                       \
       N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }

template<Direction direction, unsigned int blockDim_x, Algorithm algorithm, bool shared_memory>
inline void superposition_per_block_helper(const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
                                           const size_t N, const size_t M, const size_t width,
                                           const Polar *d_x_ptr, const double *d_u_ptr, const double *d_v,
                                           WAVE *d_y_tmp, const bool append_result)
// double *d_y_tmp_re, double *d_y_tmp_im)
{
  // unrolled for loop to allow constant blockDim
  // TODO add computation for shared memory size
  switch (blockDim.y) {
  case   1: SuperpositionPerBlock(  1) break;
  case   2: SuperpositionPerBlock(  2) break;
  case   4: SuperpositionPerBlock(  4) break;
  case   8: SuperpositionPerBlock(  8) break;
  case  16: SuperpositionPerBlock( 16) break;
  case  32: SuperpositionPerBlock( 32) break;
  case  64: SuperpositionPerBlock( 64) break;
  default: {fprintf(stderr, "BlockSize.y: %u not implemented\n", blockDim.y); exit(1);}
  }
}

template<Direction direction, Algorithm algorithm, bool shared_memory>
inline void superposition_per_block(const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
                                    const size_t N, const size_t M, const size_t width,
                                    const Polar *d_x_ptr, const double *d_u_ptr, const double *d_v,
                                    WAVE *d_y_tmp, const bool append_result)
  // double *d_y_tmp_re, double *d_y_tmp_im)
{
  // unrolled for loop to allow constant blockDim
  // Note that the max number of threads per block is 1024
  switch (blockDim.x) {
  case   1: SuperpositionPerBlockHelper(  1) break;
  case   2: SuperpositionPerBlockHelper(  2) break;
  case   4: SuperpositionPerBlockHelper(  4) break;
  case   8: SuperpositionPerBlockHelper(  8) break;
  case  16: SuperpositionPerBlockHelper( 16) break;
  case  32: SuperpositionPerBlockHelper( 32) break;
  case  64: SuperpositionPerBlockHelper( 64) break;
  default: {fprintf(stderr, "BlockSize.x: %u not implemented\n", blockDim.x); exit(1);}
  }
}

template<bool add_constant = false>
void normalize_amp(std::vector<WAVE> &c, double to = 1., bool log_normalize = false) {
  double max_amp = 0;
  for (size_t i = 0; i < c.size(); ++i)
    max_amp = fmax(max_amp, cuCabs(c[i]));

  if (max_amp < 1e-6) {
    printf("WARNING, max_amp << 1\n");
    return;
  }
  max_amp /= to;

  // zero constant is equivalent to no constant and will be removed by compiler
  const auto constant = from_polar(add_constant ? 1.0 : 0.0, ARBITRARY_PHASE);
  if (add_constant)
    max_amp *= 2.;

  for (size_t i = 0; i < c.size(); ++i) {
    if (add_constant) {
      c[i].x = c[i].x / max_amp + constant.x / 2.;
      c[i].y = c[i].y / max_amp + constant.y / 2.;
    } else {
      c[i].x = c[i].x / max_amp;
      c[i].y = c[i].y / max_amp;
    }
  }

  if (log_normalize)
    for (size_t i = 0; i < c.size(); ++i) {
      if (c[i].x > 0) c[i].x = -log(c[i].x);
      if (c[i].y > 0) c[i].y = -log(c[i].y);
    }
}

void rm_phase(std::vector<WAVE> &c) {
  // Set phase to zero, note that `a * exp(0 I) == {a, 0}`
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = {cuCabs(c[i]), 0.};
}

template<Direction direction, Algorithm algorithm = Algorithm::Naive, bool shared_memory = false>
inline std::vector<WAVE> transform(const std::vector<Polar> &x,
                                   const std::vector<double> &u,
                                   const std::vector<double> &v,
                                   const Geometry& p) {
  // x = input or source data, y = output or target data
  if (algorithm == Algorithm::Naive) assert(!shared_memory);

  // derive size of matrix y_tmp
  size_t tmp_out_size = p.batch_size.x * p.batch_size.y;
  if (algorithm == Algorithm::Alt)
    if (shared_memory)
      tmp_out_size = MIN(p.batch_size.x, p.gridDim.x) * p.batch_size.y;
    else
      tmp_out_size = MIN(p.batch_size.x, p.gridSize.x) * p.batch_size.y;

  assert(tmp_out_size > 0);
  assert(u[2] != v[2]);
  // printf("batch out size %lu\n", tmp_out_size);
  // printf("gridSize: %u, %u\n", p.gridSize.x, p.gridSize.y);
  // printf("geometry new: <<< {%u, %u}, {%u, %u} >>>\n", p.gridDim.x, p.gridDim.y, p.blockDim.x, p.blockDim.y);

#ifdef DEBUG
  assert(std::any_of(x.begin(), x.end(), abs_of_is_positive));
  assert(x.size() >= 1);
#endif

  // TODO duplicate stream batches to normal memory if too large
  auto y = std::vector<WAVE>(p.batch_size.y * p.n_batches.y);

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  thrust::device_vector<Polar> d_x = x;
  thrust::device_vector<double> d_u = u;
  // cast to pointers to allow usage in non-thrust kernels
  Polar *d_x_ptr = thrust::raw_pointer_cast(d_x.data());
  auto *d_u_ptr = thrust::raw_pointer_cast(d_u.data());

  // d_y_tmp contains block results of (partial) superposition kernel (TODO rename => d_y_block?)
  // d_y_sum contains the full superpositions, i.e. the summed rows of d_y_tmp

  // malloc data using pinned memory for all batches before starting streams
  // TODO consider std::unique_ptr<>
  WAVE *y_pinned_ptr, *d_y_tmp_ptr, *d_y_sum_ptr, *d_y_prev_ptr;
  double *v_pinned_ptr, *d_v_ptr;
  // TODO don't use pinned memory for d_y_
  auto d_y_tmp  = init::malloc_matrix<WAVE>(  &d_y_tmp_ptr,  p.n_streams, tmp_out_size);
  auto d_y_sum  = init::malloc_matrix<WAVE>(  &d_y_sum_ptr,  p.n_streams, p.batch_size.y);
  auto d_y_prev = init::malloc_matrix<WAVE>(  &d_y_prev_ptr, p.n_streams, p.batch_size.y);
  auto d_v      = init::malloc_matrix<double>(&d_v_ptr,      p.n_streams, p.batch_size.y * DIMS);

  auto v_pinned = init::pinned_malloc_vectors<double>(&v_pinned_ptr, p.n_streams, p.batch_size.y * DIMS);
  auto y_pinned = init::pinned_malloc_vectors<WAVE>(  &y_pinned_ptr, p.n_streams, p.batch_size.y);

  // TODO d_b is too large
  // const auto d_unit = thrust::device_vector<WAVE>(p.batch_size.y, {1., 0.}); // unit vector for blas
  const auto d_unit_ = thrust::device_vector<WAVE>(d_y_tmp[0].size / p.batch_size.y, {1., 0.}); // unit vector for blas
  const auto d_unit  = ConstCUDAVector<WAVE> {thrust::raw_pointer_cast(d_unit_.data()),
                                              d_unit_.size()};

  cudaStream_t streams[p.n_streams];
  cublasHandle_t handles[p.n_streams];
  for (auto& stream : streams)
    cu( cudaStreamCreate(&stream) );

  for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
    cuB( cublasCreate(&handles[i_stream]) );
    cublasSetStream(handles[i_stream], streams[i_stream]);
  }

  // batch/iter indices
  auto m_per_stream = range(p.n_streams); // TODO add range<size_t> template>
  auto n_per_stream = std::vector<size_t>(p.n_streams, 0); // rename => i_iter
  auto converging = std::vector<bool>(p.n_streams, false); // not strictly converging but "undergoing convergence checking"

  // const auto shuffle_period = FLOOR(p.n_batches.x, p.n_streams);

  // data indcides
  // rand alg 1: resample, independent samples
  // - shuffle global x, then use diagonal matrix to select indices per stream and repeat
  // - worst case n_streams times lower shuffles
  //  let float *r = rand(len) floats
  //  do  x = sort_by_key(x, r) // radix sort? linear; using hash buckets
  //   note: use Cartesian<double> to avoid column-major storage

  // rand alg 2: shuffle
  // - gen rand indices for each y-batch, then use rand accessing in kernel
  // indices = sort_by_key(range(len), r)

  // cu( cudaMalloc( &d_rand, p.n.x * sizeof(float) ) );
  auto d_rand = thrust::device_vector<float>(p.n.x); // sort_by_key() requires device vectors
  auto d_rand_ptr = thrust::raw_pointer_cast(d_rand.data());
  curandGenerator_t generator;
  init_random(&generator);

  // size_t *x_indices_ptr;
  // auto x_indices = init::malloc_matrix<size_t>(x_indices_ptr, p.n_streams, p.n.x);
  // std::vector<std::vector<size_t>>(p.n_streams, std::vector<size_t>(p.n.x));
  // for (auto &i : range(p.n_streams))
  //   x_indices[i];
  //   x_indices[i] = range(p.n.x);

  // assert(0); // TODO shuffle x_indices for every y-batch

  // auto n_datapoints = std::vector<size_t>(p.n_streams, 0);
  // auto n = range(p.n_streams);
  // for (size_t i = 0; i < p.n_batches.y; i+=p.n_streams) {
  const size_t min_n_datapoints = 1000; // before convergence computation
  unsigned int
    i_shuffle = 0,
    i_shuffle_max = FLOOR(p.n_batches.x, p.n_streams); // number of batches between shuffles
  const bool randomize = 1 && p.n_batches.x > 1;
  auto compare_func = [&](auto m){ return m < p.n_batches.y;};
  while (std::any_of(m_per_stream.begin(), m_per_stream.end(), compare_func)) {
    if (randomize && i_shuffle >= i_shuffle_max) {
      // shuffle
      // Note that the following operations are blocking
      print("Shuffle source data...");
      curandGenerateUniform(generator, d_rand_ptr, p.n.x);
      printf("p.n: <%lu, %lu>\n", p.n.x, p.n.y);
      thrust::sort_by_key(d_rand.begin(), d_rand.end(), d_x_ptr); // Note, thrust::raw_pointer_cast causes Segfaults
      thrust::sort_by_key(d_rand.begin(), d_rand.end(), (Cartesian<double> *) d_u_ptr); // reorder all dims at once

        // reset
      i_shuffle = 0;
    }

    for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const size_t
        n = n_per_stream[i_stream],
        m = m_per_stream[i_stream],
        n_datapoints = n * p.batch_size.x;
      if (m >= p.n_batches.y) continue;
      else i_shuffle++;

      if (n == 0)
        cp_batch_data_to_device<double>(&v[m * d_v[i_stream].size],
                                        v_pinned[i_stream], d_v[i_stream],
                                        streams[i_stream]);
      if (n == 0)
        assert(!converging[i_stream]);
      // // TODO memset can be avoided by directly overwriting the sum_rows result
      if (n == 0)
        cu( cudaMemsetAsync(d_y_sum_ptr, 0, d_y_sum[i_stream].size * sizeof(WAVE), streams[i_stream]) );

      // Derive current batch size in case of underutilized x-batches
      size_t local_batch_size = p.batch_size.x;
      if (!randomize)
        if (p.n.x != p.n_batches.x * p.batch_size.x && n == p.n_batches.x - 1)
          local_batch_size = p.n.x - n * p.batch_size.x;

      const size_t n_offset = (randomize ? i_shuffle : n) * p.batch_size.x;
      // const size_t n_offset = n * p.batch_size.x;
      // Note, appending result to prev results slows down computation
      const bool append_result = n > 0 && !converging[i_stream];
      // const bool append_result = n > 0 && n_datapoints < min_n_datapoints;

      superposition_per_block<direction, algorithm, shared_memory> \
        (p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
         p.batch_size.x,
         d_x_ptr + n_offset, d_u_ptr + n_offset * DIMS,
         d_v[i_stream].data, d_y_tmp[i_stream].data, append_result);

      bool finished = false;
      if (n >= p.n_batches.x - 1)
        finished = true;

      if (finished || n_datapoints > min_n_datapoints) {
        // TODO instead of memset, do
        // after the first sum_rows call, the following results should be added the that first result
        const auto beta = converging[i_stream] ? WAVE {1,0} : WAVE {0,0};
        kernel::sum_rows<false>(d_y_tmp[i_stream].size / p.batch_size.y, p.batch_size.y,
                                handles[i_stream], d_y_tmp[i_stream].data, d_unit,
                                d_y_sum[i_stream].data, beta);
        if (!finished && converging[i_stream]) {
          // double threshold = 1e-9;
          // kernel::any_is_greater<<<,,>>>(d_y_tmp, d_y_prev, &finished); // stop as soon as first nonconv. item is found
          // auto is_smaller = [](auto a, auto b) { return abs(a - b * n / (n-1) ) < threshold; };
          // thrust_equal<<<, streams[i_stream]>>>(d_y_tmp, d_y_prev, is_smaller);
          // thrust::equal<double>(d_y_sum, d_y_prev, binary_pred);
          // // any(|y - y_prev| > threshold)
          // if (converged)
          // finished = true;
        }
        // if still not finished
        if (!finished) {
          // cp d_y_tmp to dedicated memory (d_y_prev)
          // TODO check if cudaMemcpyAsync blocks kernels in other streams
          cu( cudaMemcpyAsync(d_y_prev[i_stream].data, d_y_sum[i_stream].data,
                              d_y_sum[i_stream].size * sizeof(WAVE),
                              cudaMemcpyDeviceToDevice, streams[i_stream]) );
          converging[i_stream] = true;
        }
      }
      assert(n >= p.n_batches.x - 1 || !finished);

      if (!finished) {
        // // cp d_y_tmp to dedicated memory (d_y_prev)
        // // TODO check if cudaMemcpyAsync blocks kernels in other streams
        // cu( cudaMemcpyAsync(d_y_prev[i_stream].data, d_y_sum[i_stream].data, d_y_sum[i_stream].size * sizeof(WAVE),
        //                     cudaMemcpyDeviceToDevice, streams[i_stream]) );
        n_per_stream[i_stream]++;
      }
      else {
        // Note that a subset of d_y_tmp is re-used // TODO rm?
        cp_batch_data_to_host<WAVE>(d_y_sum[i_stream].data, y_pinned[i_stream],
                                    p.batch_size.y, streams[i_stream]);

        // TODO add dedicated loop for better work distribution (similar to in func transform_full)
        cudaStreamSynchronize(streams[i_stream]);
        // TODO stage copy-phase of next batch before copy/sync?
        for (size_t j = 0; j < p.batch_size.y; ++j) {
          y[j + m * p.batch_size.y] = y_pinned[i_stream][j];
        }

        n_per_stream[i_stream] = 0;
        m_per_stream[i_stream] = *std::max_element(m_per_stream.begin(), m_per_stream.end()) + 1;
        converging[i_stream] = false;
      }
    }
  } // end while

  curandDestroyGenerator(generator);
  // sync all streams before returning
  // cudaDeviceSynchronize();
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

#ifdef TEST_CONST_PHASE
  for (size_t j = 0; j < p.n.y; ++j)
    assert(y[j].amp == N);
#endif

  for (unsigned int i = 0; i < p.n_streams; ++i)
    cudaStreamDestroy(streams[i]);
  for (auto& handle : handles)
    cuB( cublasDestroy(handle) );

  cu( cudaFree(d_y_tmp_ptr ) );
  cu( cudaFree(d_y_sum_ptr ) );
  cu( cudaFree(d_y_prev_ptr ) );
  cu( cudaFree(d_v_ptr       ) );
  cu( cudaFreeHost(v_pinned_ptr ) );
  cu( cudaFreeHost(y_pinned_ptr ) );
  return y;
}

// template<Direction direction, Algorithm algorithm = Algorithm::Naive, bool shared_memory = false>
// inline std::vector<WAVE> transform_full(const std::vector<Polar> &x,
//                                         const std::vector<double> &u,
//                                         const std::vector<double> &v,
//                                         const Geometry& p) {
//   // x = input or source data, y = output or target data
//   if (algorithm == Algorithm::Naive) assert(!shared_memory);

//   // derive size of matrix y_tmp
//   size_t tmp_out_size = p.batch_size.x * p.batch_size.y;
//   if (algorithm == Algorithm::Alt)
//     if (shared_memory)
//       tmp_out_size = MIN(p.batch_size.x, p.gridDim.x) * p.batch_size.y;
//     else
//       tmp_out_size = MIN(p.batch_size.x, p.gridSize.x) * p.batch_size.y;

//   assert(tmp_out_size > 0);
//   assert(u[2] != v[2]);
//   // printf("batch out size %lu\n", tmp_out_size);
//   // printf("gridSize: %u, %u\n", p.gridSize.x, p.gridSize.y);
//   // printf("geometry new: <<< {%u, %u}, {%u, %u} >>>\n", p.gridDim.x, p.gridDim.y, p.blockDim.x, p.blockDim.y);

// #ifdef DEBUG
//   assert(std::any_of(x.begin(), x.end(), abs_of_is_positive));
//   assert(x.size() >= 1);
// #endif
// //   if (x.size() < p.gridSize.x)
// //     printf("Warning, suboptimal input size: %u < %u\n", x.size(), p.gridSize.x);

//   // TODO duplicate stream batches to normal memory if too large
//   auto y = std::vector<WAVE>(p.batch_size.y * p.n_batches.y);

//   // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
//   const thrust::device_vector<Polar> d_x = x;
//   const thrust::device_vector<double> d_u = u;
//   // cast to pointers to allow usage in non-thrust kernels
//   const Polar* d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
//   const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);

//   // malloc data using pinned memory for all batches before starting streams
//   // TODO consider std::unique_ptr<>
//   WAVE *y_pinned_ptr;
//   WAVE *d_y_tmp_ptr;
//   double *v_pinned_ptr, *d_v_ptr;
//   // TODO don't use pinned memory for d_y_
//   auto d_y_tmp  = init::malloc_matrix<WAVE>(          &d_y_tmp_ptr,  p.n_streams, tmp_out_size);
//   auto d_v      = init::malloc_matrix<double>(        &d_v_ptr,      p.n_streams, p.batch_size.y * DIMS);
//   auto v_pinned = init::pinned_malloc_vectors<double>(&v_pinned_ptr, p.n_streams, p.batch_size.y * DIMS);
//   auto y_pinned = init::pinned_malloc_vectors<WAVE>(  &y_pinned_ptr, p.n_streams, p.batch_size.y);

//   // TODO d_b is too large
//   // const auto d_unit = thrust::device_vector<WAVE>(p.batch_size.y, {1., 0.}); // unit vector for blas
//   const auto d_unit = thrust::device_vector<WAVE>(d_y_tmp[0].size / p.batch_size.y, {1., 0.}); // unit vector for blas
//   const auto *d_b = thrust::raw_pointer_cast(d_unit.data());

//   cudaStream_t streams[p.n_streams];
//   cublasHandle_t handles[p.n_streams];
//   for (auto& stream : streams)
//     cu( cudaStreamCreate(&stream) );

//   for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
//     cuB( cublasCreate(&handles[i_stream]) );
//     cublasSetStream(handles[i_stream], streams[i_stream]);
//   }

//   for (size_t i = 0; i < p.n_batches.y; i+=p.n_streams) {
//     for (size_t n = 0; n < p.n_batches.x; ++n) {
//       // // each final x-batch may be under-used/occupied
//       if (i == 0) {
//         // cp x batch data for all streams and sync
//         // TODO
//       }

//       // Derive current batch size in case of underutilized x-batches
//       size_t local_batch_size = p.batch_size.x;
//       if (p.n.x != p.n_batches.x * p.batch_size.x && n == p.n_batches.x - 1)
//         local_batch_size = p.n.x - n * p.batch_size.x;

//       for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
//         const auto m = i + i_stream;
//         if (m >= p.n_batches.y) break;
//         if (p.n_batches.y > 10 && m % (int) (p.n_batches.y / 10) == 0 && n == 0)
//           printf("\tbatch %0.3fk / %0.3fk\n", m * 1e-3, p.n_batches.y * 1e-3);

//         if (n == 0)
//           cp_batch_data_to_device<double>(&v[m * d_v[i_stream].size],
//                                           v_pinned[i_stream], d_v[i_stream],
//                                           streams[i_stream]);

//         const bool append_result = n > 0;
//         const size_t n_offset = n * p.batch_size.x;
//         superposition_per_block<direction, algorithm, shared_memory> \
//           (p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
//            p.batch_size.x,
//            d_x_ptr + n_offset, d_u_ptr + n_offset * DIMS,
//            d_v[i_stream].data, d_y_tmp[i_stream].data, append_result);
//       }
// #ifdef TEST_CONST_PHASE
//       for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
//         const auto m = i + i_stream;
//         if (m >= p.n_batches.y) break;
//         cudaStreamSynchronize(streams[i_stream]);
//         auto d = thrust::device_vector<WAVE> (d_y_tmp[i_stream].data, d_y_tmp[i_stream].data + d_y_tmp[i_stream].size);
//         auto h = thrust::host_vector<WAVE> (d);
//         auto x_per_batch = p.batch_size.x * p.batch_size.y / tmp_out_size;
//         // printf("%lu \t %lu \t %lu\n", tmp_out_size, p.batch_size.x * p.batch_size.y, x_per_batch);
//         assert(x_per_batch > 0);
//         for (size_t j = 0; j < tmp_out_size; ++j)
//           assert(cuCabs(h[j]) == (1. + n % p.n_batches.x) * x_per_batch);
//         for (size_t j = 0; j < tmp_out_size; ++j)
//           assert(cuCabs(h[j]) == (1. + n ) * x_per_batch);
//         }
// #endif
//     } // end for n in [0,p.n_batches.x)

//     for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
//       const auto m = i + i_stream;
//       if (m >= p.n_batches.y) break;
//       kernel::sum_rows<false>(d_y_tmp[i_stream].size / p.batch_size.y, p.batch_size.y,
//                               handles[i_stream], d_y_tmp[i_stream].data, d_b, d_y_tmp[i_stream].data);
//       // Note that a subset of d_y_tmp is re-used
//       cp_batch_data_to_host<WAVE>(d_y_tmp[i_stream].data, y_pinned[i_stream],
//                                   p.batch_size.y, streams[i_stream]);
// #ifdef TEST_CONST_PHASE
//       cudaStreamSynchronize(streams[i_stream]);
//       for (size_t j = 0; j < p.batch_size.y; ++j)
//         assert(cuCabs(y_pinned[i_stream][j]) == N);
// #endif
//     }
//     for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
//       const auto m = i + i_stream;
//       if (m >= p.n_batches.y) break;
//       cudaStreamSynchronize(streams[i_stream]);
//       // TODO stage copy-phase of next batch before copy/sync?
//       for (size_t j = 0; j < p.batch_size.y; ++j) {
//         y[j + m * p.batch_size.y] = y_pinned[i_stream][j];
//       }
//     }
//   }

//   // sync all streams before returning
//   // cudaDeviceSynchronize();
//   CubDebugExit(cudaPeekAtLastError());
//   CubDebugExit(cudaDeviceSynchronize());

// #ifdef TEST_CONST_PHASE
//   for (size_t j = 0; j < p.n.y; ++j)
//     assert(y[j].amp == N);
// #endif

//   for (unsigned int i = 0; i < p.n_streams; ++i)
//     cudaStreamDestroy(streams[i]);
//   for (auto& handle : handles)
//     cuB( cublasDestroy(handle) );

//   cu( cudaFree(d_y_tmp_ptr ) );
//   cu( cudaFree(d_v_ptr       ) );
//   cu( cudaFreeHost(v_pinned_ptr ) );
//   cu( cudaFreeHost(y_pinned_ptr ) );
//   return y;
// }

/**
 * Time the transform operation over the full input.
 * Do a second transformation if add_reference is true.
 */
template<Direction direction, bool add_constant_wave = false, bool add_reference_wave = false>
std::vector<Polar> time_transform(const std::vector<Polar> &x,
                                  const std::vector<double> &u,
                                  const std::vector<double> &v,
                                  const Geometry& p,
                                  struct timespec *t1, struct timespec *t2, double *dt,
                                  bool verbose = false) {
  clock_gettime(CLOCK_MONOTONIC, t1);
  auto weights = std::vector<double> {1,
                                      add_constant_wave ? 1 : 0,
                                      add_reference_wave ? 1 : 0};
  normalize(weights);

  // for 512x512 planes, griddim 128x1, blockdim 64x16, 1 stream:
  // transform with custom agg: 25.617345 s
  // transform naive (Alt algo) with shared memory: 9.337457 s
  // (2.7 speedup)
  // for one-to-many input: speedup was at least ~10

  std::vector<WAVE> y;
  switch (p.algorithm) {
  case 1: y = transform<direction, Algorithm::Naive, false>(x, u, v, p); break;
  case 2: y = transform<direction, Algorithm::Alt, false>(x, u, v, p); break;
  case 3: y = transform<direction, Algorithm::Alt, true>(x, u, v, p); break;
  default: {fprintf(stderr, "algorithm is incorrect"); exit(1); }
  }

  const bool shared_memory = false;
  // average of transformation and constant if any
  normalize_amp<add_constant_wave>(y, weights[0] + weights[1]);

  assert(!add_constant_wave);
  if (add_reference_wave) {
    /**
     * Add single far away light source behind the second (v) plane,
     * with arbitrary (but constant) phase
     * adding the planar wave should happen before squaring the amplitude
    */
    // TODO do this on CPU?
    const double z_offset = v[2] - DISTANCE_REFERENCE_WAVE; // assume v[:, 2] is constant
    printf("ref v[2]: %e\n", v[2]);
    auto y_reference = transform<direction, Algorithm::Alt, shared_memory>({{1, 0.}}, {{0., 0., z_offset}}, v, p);
    normalize_amp<false>(y_reference, weights[2]);
    // let full reference wave (amp+phase) interfere with original wave
    add_complex(y, y_reference);
    // reset phase of result, because the projector is limited
    for (size_t i = 0; i < p.n.y; ++i)
      y[i] = from_polar(cuCabs(y[i]), angle(y_reference[i]));
  }

  clock_gettime(CLOCK_MONOTONIC, t2);
  *dt = diff(*t1, *t2);
  if (verbose)
    print_result(std::vector<double>{*dt}, x.size(), y.size());

  auto y_result = std::vector<Polar>(p.n.y, {0,0});
  for (size_t i = 0; i < y_result.size(); ++i)
    y_result[i] = {cuCabs(y[i]), angle(y[i])};
  return y_result;
}
