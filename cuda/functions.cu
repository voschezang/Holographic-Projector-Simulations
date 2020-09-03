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
inline void cp_batch_data_to_device(const std::vector<T> v, const size_t v_offset,
                                    T *v_pinned, CUDAVector<T> d_v, cudaStream_t stream) {
  // any host memory involved in async/overlapping data transfers must be page-locked
  // Note that d_v.size <= v.size
  const auto
    max_len = v.size() - v_offset,
    len = MIN(d_v.size, max_len),
    remaining_len = d_v.size - len;
  assert(v_offset <= v.size());
  assert(len <= max_len);
  assert(len + v_offset <= v.size());
  for (size_t i = 0; i < len; ++i)
    v_pinned[i] = v[v_offset + i];
  if (remaining_len)
    memset(v_pinned + len, 0, remaining_len * sizeof(T));

  cu ( cudaMemcpyAsync( d_v.data, v_pinned, d_v.size * sizeof(T),
                        cudaMemcpyHostToDevice, stream ) );
}

template<typename T = WAVE>
inline void cp_batch_data_to_host(const T *d_y, T *y_pinned, const size_t len, cudaStream_t stream) {
  // any host memory involved in async/overlapping data transfers must be page-locked
  cu ( cudaMemcpyAsync( y_pinned, d_y, len * sizeof(T),
                        cudaMemcpyDeviceToHost, stream ) );
}

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
#define SuperpositionPerBlock(blockDim_y) {                             \
    assert(blockDim_x * blockDim_y <= 1024);                            \
    superposition::per_block<direction, blockDim_x, blockDim_y, algorithm, shared_memory> \
      <<< gridDim, blockDim, 0, stream >>>                              \
      (state, seed, i_stream,                                           \
       N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }
#else
#define SuperpositionPerBlock(blockDim_y) {                             \
    assert(blockDim_x * blockDim_y <= 1024);                            \
    superposition::per_block<direction, blockDim_x, blockDim_y, algorithm, shared_memory> \
      <<< gridDim, blockDim, 0, stream >>>                              \
      (N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }
#endif

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
#define SuperpositionPerBlockHelper(blockDim_x) {                       \
    superposition_per_block_helper<direction, blockDim_x, algorithm, shared_memory> \
      (state, seed, i_stream,                                           \
       gridDim, blockDim, stream,                                       \
       N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }
#else
#define SuperpositionPerBlockHelper(blockDim_x) {                       \
    superposition_per_block_helper<direction, blockDim_x, algorithm, shared_memory> \
      (gridDim, blockDim, stream,                                       \
       N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
  }
#endif

template<Direction direction, unsigned int blockDim_x, Algorithm algorithm, bool shared_memory>
inline void superposition_per_block_helper(
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
                                           curandState *state, const unsigned int seed, const size_t i_stream,
#endif
                                           const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
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
inline void superposition_per_block(
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
                                    curandState *state, const unsigned int seed, const size_t i_stream,
#endif
                                    const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
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
                                   const std::vector<double> &v2,
                                   const Geometry& p) {

  auto v = v2;
  const bool shuffle_v = 0;
  const bool reorder_v = 0, reorder_v_rm_phase = 1;
  auto map = std::vector<size_t>(p.n.y);
  if (shuffle_v) {
    std::iota(map.begin(), map.end(), 0);
    std::random_shuffle(map.begin(), map.end());
    for (size_t i = 0; i < map.size(); ++i)
      for (int dim = 0; dim < DIMS; ++dim)
        v[Ix(i,dim)] = v2[Ix(map[i],dim)];

    if (0) {
      // test revert
      auto v_copy = v;
      for (size_t i = 0; i < map.size(); ++i)
        for (int dim = 0; dim < DIMS; ++dim)
          v[Ix(map[i],dim)] = v_copy[Ix(i,dim)];

      for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == v2[i]);
    }
  }

  // only in case of second transformation
  if (reorder_v && p.n.x > p.batch_size.x) {
    // reorder v data s.t. each grid represents a square area in space
    // neglect boundary cells/pixels

    // assume aspect ratio = 1
    size_t m_sqrt = (size_t) sqrt(p.n.y);
    assert(m_sqrt*m_sqrt == p.n.y);
    size_t G = (size_t) sqrt(p.batch_size.y); // batch_size
    size_t m_sqrt2 = FLOOR(m_sqrt, G) * G; // minus boundaries
    assert(m_sqrt > 0);

    for (size_t i = 0; i < m_sqrt2; ++i) {
      for (size_t j = 0; j < m_sqrt2; ++j) {
        dim2
          i_batch = {i / G, j / G},
          g = {i % G, j % G};
        size_t i_transpose = (i_batch.x * m_sqrt2/G + i_batch.y) * G*G + g.x * G + g.y;
        for (int dim = 0; dim < DIMS; ++dim) {
          // v[Ix(i_transpose, dim)];
          v[Ix(i_transpose, dim)] = v2[Ix2D(i,j,dim,m_sqrt)];
          // v[Ix2D(i_batch.x * G + g.x,
          //        i_batch.y * G + g.y, dim, m_sqrt2)] = v2[Ix2D(i,j,dim,m_sqrt)];
        }
      }
    }
  }




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
  auto d_y_tmp  = init::malloc_matrix<WAVE>(  &d_y_tmp_ptr,  p.n_streams, tmp_out_size);
  auto d_y_sum  = init::malloc_matrix<WAVE>(  &d_y_sum_ptr,  p.n_streams, p.batch_size.y);
  auto d_y_prev = init::malloc_matrix<WAVE>(  &d_y_prev_ptr, p.n_streams, p.batch_size.y);
  auto d_v      = init::malloc_matrix<double>(&d_v_ptr,      p.n_streams, p.batch_size.y * DIMS);

  auto v_pinned = init::pinned_malloc_vectors<double>(&v_pinned_ptr, p.n_streams, p.batch_size.y * DIMS);
  auto y_pinned = init::pinned_malloc_vectors<WAVE>(  &y_pinned_ptr, p.n_streams, p.batch_size.y);

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
  auto converging   = std::vector<bool>(p.n_streams, false); // not strictly converging but "undergoing convergence checking"
  auto finished     = std::vector<bool>(p.n_streams, false);

  // const auto shuffle_period = FLOOR(p.n_batches.x, p.n_streams);

  // cu( cudaMalloc( &d_rand, p.n.x * sizeof(float) ) );
  auto d_rand = thrust::device_vector<float>(p.n.x); // sort_by_key() requires device vectors
  auto d_rand_ptr = thrust::raw_pointer_cast(d_rand.data());
  curandGenerator_t generator;
  init_random(&generator);

  cudaDeviceSynchronize();
  // curandGenerateUniform(generator, d_rand_ptr, p.n.x);
  cuR( curandGenerateUniform(generator, d_rand_ptr, p.n.x * sizeof(float) / 32) );
  cudaDeviceSynchronize();

  // Copy indices because both keys and values will be sorted
  auto d_rand2 = d_rand;

  auto d_u_row_ptr = thrust::device_ptr<Cartesian<double>> ( (Cartesian<double> *) d_u_ptr );
  const thrust::device_vector<Polar> d_x_original = d_x;
  const thrust::device_vector<Cartesian<double>> d_u_original (d_u_row_ptr, d_u_row_ptr + p.n.x);

  // init randoms
  thrust::device_vector<size_t> indices(p.n.x);
  assert(indices.begin()[0] == 0);
  assert(indices.begin()[0] == 0);
  thrust::counting_iterator<size_t> identity(0);
  thrust::copy_n(identity, p.n.x, indices.begin());

  if (0) {
    // Sort twice, using a copy of the randomized keys
    thrust::sort_by_key(d_rand.begin(), d_rand.end(), d_x.begin()); // Note, thrust::raw_pointer_cast causes Segfaults
    thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), d_u_row_ptr); // reorder all dims at once
  }
  if (0) {
    // Assume sorting is much more expensive than copying data (even though both are O(n))
    // Sort once, then do scatter operations on the data
    thrust::sort_by_key(d_rand.begin(), d_rand.end(), indices.begin());
    // thrust::permutation_iterator(d_x.begin(), d_x.end(), indices.begin());
    thrust::scatter(d_x_original.begin(), d_x_original.end(), indices.begin(), d_x.begin());
    thrust::scatter(d_u_original.begin(), d_u_original.end(), indices.begin(), d_u_row_ptr);
  }

  // if (4) {
  //   // shuffle v
  //   auto d_rand_v = thrust::device_vector<float>(p.n.y); // sort_by_key() requires device vectors
  //   auto d_rand_v_ptr = thrust::raw_pointer_cast(d_rand_v.data());

  //   thrust::device_vector<double> d_v2 = v;
  //   auto d_v_ptr = thrust::raw_pointer_cast(d_v2.data());
  //   auto d_v_row_ptr = thrust::device_ptr<Cartesian<double>> ( (Cartesian<double> *) d_v_ptr );
  //   const thrust::device_vector<Cartesian<double>> d_v_original (d_v_row_ptr, d_v_row_ptr + p.n.x);

  //   thrust::device_vector<size_t> indices_v(p.n.y);
  //   thrust::copy_n(identity, p.n.y, indices.begin());
  //   assert(indices.begin()[0] == 0);
  //   thrust::sort_by_key(d_rand_v.begin(), d_rand_v.end(), indices_v.begin());
  //   thrust::scatter(d_v_original.begin(), d_v_original.end(), indices_v.begin(), d_v2.begin());
  // }


  // if (0) {
  //   // this throws [functions.cu:28] CUDA Runtime Error: an illegal memory access was encountered
  //   thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), (Cartesian<double> *) d_u_ptr); // reorder all dims at once
  //   cudaDeviceSynchronize();
  // }
  // else {
  //   // TODO this copies data
  //   // use either of:
  //   //  1) sort using casting (see above)
  //   //  2) ArrayOfStruct: use Cartesian<double> everywhere
  //   //  3) StructOfArray: split u.x,y,z and sort them individually - this requires additional copying of indices
  //   // d_u_vector = thrust::device_vector<Cartesian<double>>(d_u_ptr2, d_u_ptr2 + p.n.x);
  //   thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), d_u_row_ptr); // reorder all dims at once
  //   // thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), d_u_vector.begin()); // reorder all dims at once
  //   // d_u = thrust::device_vector<double>((double *) d_u_ptr2, 3*p.n.x + (double *) d_u_ptr2); // this does nothing
  //   d_u_vector_ptr = (double *) thrust::raw_pointer_cast(d_u_vector.data());
  //   // (optional), update d_u
  //   d_u = thrust::device_vector<double>(d_u_vector_ptr, 3*p.n.x + d_u_vector_ptr);
  //   // cudaDeviceSynchronize();
  //   // // printf("equals: %e == %e ?\n", d_u[0], d_u_vector[0]);
  //   // if (p.n.x >= 100)
  //   //   for (int i = p.n.x - 100; i < p.n.x; ++i) {
  //   //     // for (int i = 0; i < p.n.x; ++i) {
  //   //     Cartesian<double> w = d_u_vector[i];
  //   //     double xx = w.x;
  //   //     thrust::host_vector<Cartesian<double>> yy = d_u_vector;
  //   //     bool eq = d_u[Ix(i,0)] == yy[0].x;
  //   //     eq = (bool) eq;
  //   //     printf("equals[%i]: %e == %e ? -> %d\n", i, d_u[Ix(i,0)], yy[0].x, eq);
  //   //     printf("equals[%i]: %e == %e ? -> %d\n", i, d_u[Ix(i,0)], yy[0].x, !eq);
  //   //     eq = d_u[Ix(i,0)] == xx;
  //   //     eq = (bool) eq;
  //   //     printf("equals[%i]: %e == %e ? -> %d\n", i, d_u[Ix(i,0)], xx, eq);
  //   //     printf("equals[%i]: %e == %e ? -> %d\n", i, d_u[Ix(i,0)], xx, !eq);
  //   //     assert(eq == 0 || eq == 1);
  //   //     // printf("equals[%i]: %e == %e?\n", i, d_u[Ix(i,0)], w.x);
  //   //     // printf("equals[%i]: %e == %e?\n", i, d_u[Ix(i,1)], w.y);
  //   //     // printf("equals[%i]: %e == %e?\n", i, d_u[Ix(i,2)], w.z);
  //   //     assert(d_u[Ix(i,0)] == w.x);
  //   //     assert(d_u[Ix(i,1)] == w.y);
  //   //     assert(d_u[Ix(i,2)] == w.z);
  //   //   }
  // }
  // printf("bool %d, %d\n", 3 == 4, 4 == 4);

  // cudaDeviceSynchronize();
  // if (p.n.x > 100) {
  //   print("raw1");
  //   double raw;
  //   print("raw2");
  //   raw = thrust::reduce(d_u.begin(), d_u.end(), (double) 0);
  //   cudaDeviceSynchronize();
  //   print("assert2");
  //   assert(equals(raw, sum(u)));
  //   cudaDeviceSynchronize();

  //   auto x3 = thrust::device_vector<double>((double *) d_x_ptr, 2*p.n.x + (double *) d_x_ptr);
  //   raw = thrust::reduce(x3.begin(), x3.end(), (double) 0);
  //   cudaDeviceSynchronize();
  //   print("assert1");
  //   assert(equals(raw, sum((double*) x.data(), 2*x.size())));
  // }
  //



#ifdef RANDOMIZE_SUPERPOSITION_INPUT
  static unsigned int seed = 12345;
  seed++; // TODO handle properly
  curandState *rng_state;
  if (p.n.x > p.gridSize.x) {
    cu( cudaMalloc(&rng_state, p.n_streams * p.gridSize.x * p.gridSize.y * sizeof(curandState)) );
    printf("---> max i_state %zu, n streams: %zu\n", p.n_streams * p.gridSize.x * p.gridSize.y, p.n_streams);
    // curandStatePhilox4_32_10_t *rng_state;
    // cu( cudaMalloc(&rng_state, n_streams * p.gridSize.x * p.gridSize.y * sizeof(curandStatePhilox4_32_10_t)) );
    // TODO time different rngs
    // is MRG32k3a better than default XORWOW?
    // curandStateMRG32k3a *devMRGStates;
    // curandStatePhilox4_32_10_t *devPHILOXStates;
    // for (auto& stream : streams)
    print("init rng:");
    cudaDeviceSynchronize();
    for (auto& i_stream : range(p.n_streams))
      kernel::init_rng<<<p.gridDim, p.blockDim, 0, streams[i_stream]>>>(rng_state, seed, i_stream);
    cudaDeviceSynchronize();
    print("init rng post");
  }
#endif



  const size_t
    min_n_datapoints = 1024, // before convergence computation
    batches_between_convergence = p.batch_size.x >= min_n_datapoints ? 1 : CEIL(min_n_datapoints, p.batch_size.x),
    //      - TODO make dependent on batch_size.x?
    i_shuffle_max = FLOOR(p.n_batches.x, p.n_streams); // number of batches between shuffles
  size_t
    i_shuffle = i_shuffle_max; // init high s.t. shuffling will be triggered

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
  if (p.n.x > p.gridSize.x)
    assert(p.n.x >= min_n_datapoints); // TODO not implemented

  // used iff (p.n.x > p.gridSize.x)
  const size_t
    min_kernels_per_estimate = CEIL(min_n_datapoints, p.batch_size.x),
    n_sample_bins_total = min_kernels_per_estimate * p.batch_size.x,
    sample_bin_size = CEIL(p.n.x, n_sample_bins_total);
  // n_sample_bins_per_kernel = (n_sample_bins_total)
  if (p.n.x > p.gridSize.x)
    printf("N: %zu, min_kernels_per_estimate: %zu, n_sample_bins_total: %zu, sample_bin_size: %zu\n",
           p.n.x, min_kernels_per_estimate, n_sample_bins_total, sample_bin_size);
#endif

  const bool randomize = 0 && p.n.x > 1 && p.n_batches.x > 1;

  if (randomize && i_shuffle_max * p.batch_size.x >= p.n.x) {
    printf("TODO?, p.n.x: %zu\n", p.n.x);
    // i_shuffle_max--;
    assert(0);
  }

  auto compare_func = [&](auto m){ return m < p.n_batches.y;};
  while (std::any_of(m_per_stream.begin(), m_per_stream.end(), compare_func)) {

    if (randomize && i_shuffle >= i_shuffle_max) {
      // TODO use stages; this can be done in parallel with copying data but not with superposition kernels
      cudaDeviceSynchronize(); // finish all superposition kernels
      cuR( curandGenerateUniform(generator, d_rand_ptr, p.n.x * sizeof(float) / 32) );
      cudaDeviceSynchronize();
      if (0) {
        d_rand2 = d_rand;
        thrust::sort_by_key(d_rand.begin(), d_rand.end(), d_x.begin()); // Note, thrust::raw_pointer_cast causes Segfaults
        thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), d_u_row_ptr); // reorder all dims at once
        // thrust::sort_by_key(d_rand2.begin(), d_rand2.end(), d_u_vector.begin()); // reorder all dims at once

        // (optional), update d_u
        // d_u = thrust::device_vector<double>(d_u_vector_ptr, 3*p.n.x + d_u_vector_ptr);
      } else {
        thrust::sort_by_key(d_rand.begin(), d_rand.end(), indices.begin());
        thrust::scatter(d_x_original.begin(), d_x_original.end(), indices.begin(), d_x.begin());
        thrust::scatter(d_u_original.begin(), d_u_original.end(), indices.begin(), d_u_row_ptr);
      }
      // cudaDeviceSynchronize();

      // reset
      i_shuffle = 0;
    }

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const size_t
        n = n_per_stream[i_stream],
        m = m_per_stream[i_stream];
      if (m >= p.n_batches.y) continue;

      if (p.n_batches.y > 10 && m % (int) (p.n_batches.y / 10) == 0 && n == 0)
        printf("\tbatch %0.3fk / %0.3fk\n", m * 1e-3, p.n_batches.y * 1e-3);

      if (n == 0)
        cp_batch_data_to_device<double>(v, m * d_v[i_stream].size,
                                        v_pinned[i_stream], d_v[i_stream],
                                        streams[i_stream]);
      if (n == 0)
        assert(!converging[i_stream]);

      // Derive current batch size in case of underutilized x-batches
      size_t local_batch_size = p.batch_size.x;
      if (!randomize)
        if (p.n.x != p.n_batches.x * p.batch_size.x && n == p.n_batches.x - 1)
          local_batch_size = p.n.x - n * p.batch_size.x;

      size_t n_offset = (randomize ? i_shuffle : n) * p.batch_size.x;
      assert(i_shuffle_max * p.batch_size.x < p.n.x);
      if (randomize)
        assert(n_offset + p.batch_size.x < p.n.x);

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
      if (p.n.x > p.gridSize.x) {
        // TODO add iteration number for x-data in kernel, i.e. independent control of batch size and number of iterations
        // (currently the local_batch_size controls both the number of iterations in the kernel and the range of data that is sampled from)
        // local_batch_size = p.n.x;
        // n_offset = 0;

        // min_kernels_per_estimate, n_sample_bins_total, sample_bin_size
        const size_t n_relative = n * (p.batch_size.x) / min_n_datapoints;
        local_batch_size = p.batch_size.x * sample_bin_size;
        n_offset = local_batch_size * n_relative;
        assert(local_batch_size * (1+n_relative) <= p.n.x);
      }
#endif

      // const size_t n_offset = n * p.batch_size.x;
      // Note, appending result to prev results slows down computation
      const bool append_result = n > 0 && !converging[i_stream];
      // const bool append_result = n > 0 && n_datapoints < min_n_datapoints;

      superposition_per_block<direction, algorithm, shared_memory>  \
        (
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
         rng_state, seed, i_stream, p.thread_size.x, sample_bin_size
#endif
         p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
         p.batch_size.x,
         d_x_ptr + n_offset, d_u_ptr + n_offset * DIMS,
         d_v[i_stream].data, d_y_tmp[i_stream].data, append_result
         );
#ifndef RANDOMIZE_SUPERPOSITION_INPUT
      i_shuffle++;
#endif
    }

    for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const size_t
        n = n_per_stream[i_stream],
        m = m_per_stream[i_stream],
        n_datapoints = (n+1) * p.batch_size.x;
      if (m >= p.n_batches.y) continue;

      // if (randomize && n >= (p.n_batches.x - 1) / 2)
      //   finished[i_stream] = true;
      // if (!randomize && n >= p.n_batches.x - 1)
      //   finished[i_stream] = true;
      if (n >= p.n_batches.x - 1)
        finished[i_stream] = true;

      // if (n >= 1)
      //   finished[i_stream] = true;

      if (finished[i_stream] || n_datapoints >= min_n_datapoints) {
        // TODO instead of memset, do
        // after the first sum_rows call, the following results should be added the that first result
        const auto beta = converging[i_stream] ? WAVE {1,0} : WAVE {0,0};
        kernel::sum_rows<false>(d_y_tmp[i_stream].size / p.batch_size.y, p.batch_size.y,
                                handles[i_stream], d_y_tmp[i_stream].data, d_unit,
                                d_y_sum[i_stream].data, beta);

        if (!finished[i_stream] && converging[i_stream] && n % batches_between_convergence == 0) {
          // TODO make threshold dependent on max distance?
          // const double threshold = 1e-4;
          const double threshold = 0;
          const double
            prev_n = (n+1 - batches_between_convergence) * p.batch_size.x,
            scale_a = 1. / (double) n_datapoints,
            scale_b = 1. / prev_n; // to re-normalize the prev result
          // TOOD try this, and don't use pinned memory
          finished[i_stream] = thrust::equal(thrust::cuda::par.on(streams[i_stream]),
                                             (double *) d_y_sum[i_stream].data,
                                             (double *) d_y_sum[i_stream].data + d_y_sum[i_stream].size * 2,
                                             (double *) d_y_prev[i_stream].data,
                                             is_smaller(scale_a, scale_b, threshold));
          // alt, in case thrust stream selection doesn't work
          // kernel::equal<<< 1, 1, 0, streams[i_stream]>>>((double *) d_y_sum[i_stream].data, (double *) d_y_prev[i_stream].data,
          //                                                d_y_sum[i_stream].size * 2,
          //                                                is_smaller(scale_a, scale_b, threshold),
          //                                                converged + i_stream);
          // cudaStreamSynchronize(streams[i_stream]);
          // if (converged[i_stream])
          //   finished[i_stream] = true;
          if (finished[i_stream])
            printf("converged/finished at batch.x: %lu/%lu\n", n, p.n_batches.x);
        }
        // if still not finished
        if (randomize && !finished[i_stream]) {
          // cp superpositions as previous result
          cu( cudaMemcpyAsync(d_y_prev[i_stream].data, d_y_sum[i_stream].data,
                              d_y_sum[i_stream].size * sizeof(WAVE),
                              cudaMemcpyDeviceToDevice, streams[i_stream]) );
          converging[i_stream] = true;
        }
      }

      if (!finished[i_stream])
        n_per_stream[i_stream]++;
      else
        cp_batch_data_to_host<WAVE>(d_y_sum[i_stream].data, y_pinned[i_stream],
                                    p.batch_size.y, streams[i_stream]);
    }

    for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      if (finished[i_stream]) {
        const size_t
          n = n_per_stream[i_stream],
          m = m_per_stream[i_stream],
          n_datapoints = (n+1) * p.batch_size.x;
        assert(m < p.n_batches.y);
        const double div_by_n = 1. / (double) n_datapoints;
        // TODO add dedicated loop for better work distribution (similar to in func transform_full)
        // TODO stage copy-phase of next batch before copy/sync?
        cudaStreamSynchronize(streams[i_stream]);
        for (size_t j = 0; j < p.batch_size.y; ++j) {
          // save average w.r.t the number of samples used per y-batch
          // TODO make transformation_full compatible with this average
          y[j + m * p.batch_size.y].x = y_pinned[i_stream][j].x * div_by_n;
          y[j + m * p.batch_size.y].y = y_pinned[i_stream][j].y * div_by_n;
          // TODO assert y index is within bounds
        }

        n_per_stream[i_stream] = 0;
        m_per_stream[i_stream] = *std::max_element(m_per_stream.begin(), m_per_stream.end()) + 1;
        converging[i_stream] = false;
        // converged[i_stream] = false;
        finished[i_stream] = false;
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

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
  if (p.n.x > p.gridSize.x)
    cu( cudaFree(rng_state) );
#endif

  cu( cudaFree(d_y_tmp_ptr  ) );
  cu( cudaFree(d_y_sum_ptr  ) );
  cu( cudaFree(d_y_prev_ptr ) );
  cu( cudaFree(d_v_ptr      ) );
  cu( cudaFreeHost(v_pinned_ptr) );
  cu( cudaFreeHost(y_pinned_ptr) );


  // only in case of second transformation
  if (reorder_v && p.n.x > p.batch_size.x) {
    // revert y data (v data)
    assert(!shuffle_v);
    // assume aspect ratio = 1
    size_t m_sqrt = (size_t) sqrt(p.n.y);
    assert(m_sqrt*m_sqrt == p.n.y);
    size_t G = (size_t) sqrt(p.batch_size.y); // batch_size
    size_t m_sqrt2 = FLOOR(m_sqrt, G) * G; // minus boundaries
    assert(m_sqrt > 0);

    auto y2 = y;
    for (size_t i = 0; i < m_sqrt2; ++i) {
      for (size_t j = 0; j < m_sqrt2; ++j) {
        dim2
          i_batch = {i / G, j / G},
          g = {i % G, j % G};
        size_t i_transpose = (i_batch.x * m_sqrt2/G + i_batch.y) * G*G + g.x * G + g.y;
        y2[i * m_sqrt + j] = y[i_transpose];
        if (reorder_v_rm_phase) {
          if (i_batch.x % 2 == 0)
            if (i_batch.y % 2 == 0)
              y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 0);
            else
              y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 1.5);
          else
            if (i_batch.y % 2 == 0)
              y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 3);
            else
              y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 4.5);
        }
      }
    }
    return y2;
  }

  if (shuffle_v) {
    auto y2 = y;
    for (size_t i = 0; i < map.size(); ++i)
      for (int dim = 0; dim < DIMS; ++dim)
        y2[map[i]] = {y[i].x, y[i].y};

    return y2;
  }
  return y;
}

template<Direction direction, Algorithm algorithm = Algorithm::Naive, bool shared_memory = false>
inline std::vector<WAVE> transform_full(const std::vector<Polar> &x2,
                                        const std::vector<double> &u2,
                                        const std::vector<double> &v,
                                        const Geometry& p) {

  //
  auto x = x2;
  auto u = u2;
  if (0 && p.n.x >= 1000) {
    double tmp;
    for (int n = 0; n < p.n.x; ++n) {
      auto j = rand() % p.n.x;
      assert(j < p.n.x);
      // j = n;
      // if (j > 1)
      //   j = n - 1;
      tmp = x[n].amp; x[n].amp = x[j].amp; x[j].amp = tmp;
      tmp = x[n].phase; x[n].phase = x[j].phase; x[j].phase = tmp;
      for (int dim = 0; dim < DIMS; ++dim) {
        tmp = u[Ix(n, dim)]; u[Ix(n, dim)] = u[Ix(j, dim)]; u[Ix(j, dim)] = tmp;
      }
    }
    assert(x[0].amp != x2[0].amp ||
           x[1].amp != x2[1].amp ||
           x[2].amp != x2[2].amp);
    printf("sum x: %f vs. %f\n",
           sum( (double *) x.data(), x.size() * 2),
           sum( (double *) x2.data(), x2.size() * 2));
    assert(abs(sum( (double *) x.data(), x.size() * 2) - sum( (double *) x2.data(), x2.size() * 2)) < 1e-6);
    assert(abs(sum(u) - sum(u2)) < 1e-6);
  }
  //


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
//   if (x.size() < p.gridSize.x)
//     printf("Warning, suboptimal input size: %u < %u\n", x.size(), p.gridSize.x);

  // TODO duplicate stream batches to normal memory if too large
  auto y = std::vector<WAVE>(p.batch_size.y * p.n_batches.y);

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  const thrust::device_vector<Polar> d_x = x;
  const thrust::device_vector<double> d_u = u;
  // cast to pointers to allow usage in non-thrust kernels
  const Polar* d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);

  // malloc data using pinned memory for all batches before starting streams
  // TODO consider std::unique_ptr<>
  WAVE *y_pinned_ptr;
  WAVE *d_y_tmp_ptr;
  double *v_pinned_ptr, *d_v_ptr;
  // TODO don't use pinned memory for d_y_
  auto d_y_tmp  = init::malloc_matrix<WAVE>(          &d_y_tmp_ptr,  p.n_streams, tmp_out_size);
  auto d_v      = init::malloc_matrix<double>(        &d_v_ptr,      p.n_streams, p.batch_size.y * DIMS);
  auto v_pinned = init::pinned_malloc_vectors<double>(&v_pinned_ptr, p.n_streams, p.batch_size.y * DIMS);
  auto y_pinned = init::pinned_malloc_vectors<WAVE>(  &y_pinned_ptr, p.n_streams, p.batch_size.y);

  // const auto d_unit = thrust::device_vector<WAVE>(p.batch_size.y, {1., 0.}); // unit vector for blas
  // const auto d_unit = thrust::device_vector<WAVE>(d_y_tmp[0].size / p.batch_size.y, {1., 0.}); // unit vector for blas
  // const auto *d_b = thrust::raw_pointer_cast(d_unit.data());
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

  for (size_t i = 0; i < p.n_batches.y; i+=p.n_streams) {
    for (size_t n = 0; n < p.n_batches.x; ++n) {
      // // each final x-batch may be under-used/occupied
      if (i == 0) {
        // cp x batch data for all streams and sync
        // TODO
      }

      // Derive current batch size in case of underutilized x-batches
      size_t local_batch_size = p.batch_size.x;
      if (p.n.x != p.n_batches.x * p.batch_size.x && n == p.n_batches.x - 1)
        local_batch_size = p.n.x - n * p.batch_size.x;

      for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
        const auto m = i + i_stream;
        if (m >= p.n_batches.y) break;
        if (p.n_batches.y > 10 && m % (int) (p.n_batches.y / 10) == 0 && n == 0)
          printf("\tbatch %0.3fk / %0.3fk\n", m * 1e-3, p.n_batches.y * 1e-3);

        if (n == 0)
          cp_batch_data_to_device<double>(v, m * d_v[i_stream].size,
                                          v_pinned[i_stream], d_v[i_stream],
                                          streams[i_stream]);

        const bool append_result = n > 0;
        const size_t n_offset = n * p.batch_size.x;
        superposition_per_block<direction, algorithm, shared_memory> \
          (p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
           p.batch_size.x,
           d_x_ptr + n_offset, d_u_ptr + n_offset * DIMS,
           d_v[i_stream].data, d_y_tmp[i_stream].data, append_result);
      }
#ifdef TEST_CONST_PHASE
      for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
        const auto m = i + i_stream;
        if (m >= p.n_batches.y) break;
        cudaStreamSynchronize(streams[i_stream]);
        auto d = thrust::device_vector<WAVE> (d_y_tmp[i_stream].data, d_y_tmp[i_stream].data + d_y_tmp[i_stream].size);
        auto h = thrust::host_vector<WAVE> (d);
        auto x_per_batch = p.batch_size.x * p.batch_size.y / tmp_out_size;
        // printf("%lu \t %lu \t %lu\n", tmp_out_size, p.batch_size.x * p.batch_size.y, x_per_batch);
        assert(x_per_batch > 0);
        for (size_t j = 0; j < tmp_out_size; ++j)
          assert(cuCabs(h[j]) == (1. + n % p.n_batches.x) * x_per_batch);
        for (size_t j = 0; j < tmp_out_size; ++j)
          assert(cuCabs(h[j]) == (1. + n ) * x_per_batch);
        }
#endif
    } // end for n in [0,p.n_batches.x)

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto m = i + i_stream;
      if (m >= p.n_batches.y) break;
      kernel::sum_rows<false>(d_y_tmp[i_stream].size / p.batch_size.y, p.batch_size.y,
                              handles[i_stream], d_y_tmp[i_stream].data, d_unit,
                              d_y_tmp[i_stream].data);
      // Note that a subset of d_y_tmp is re-used
      cp_batch_data_to_host<WAVE>(d_y_tmp[i_stream].data, y_pinned[i_stream],
                                  p.batch_size.y, streams[i_stream]);
#ifdef TEST_CONST_PHASE
      cudaStreamSynchronize(streams[i_stream]);
      for (size_t j = 0; j < p.batch_size.y; ++j)
        assert(cuCabs(y_pinned[i_stream][j]) == N);
#endif
    }
    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto m = i + i_stream;
      if (m >= p.n_batches.y) break;
      cudaStreamSynchronize(streams[i_stream]);
      // TODO stage copy-phase of next batch before copy/sync?
      for (size_t j = 0; j < p.batch_size.y; ++j) {
        y[j + m * p.batch_size.y] = y_pinned[i_stream][j];
      }
    }
  }

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
  cu( cudaFree(d_v_ptr       ) );
  cu( cudaFreeHost(v_pinned_ptr ) );
  cu( cudaFreeHost(y_pinned_ptr ) );
  return y;
}

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
  // switch (p.algorithm) {
  // case 1: y = transform<direction, Algorithm::Naive, false>(x, u, v, p); break;
  // case 2: y = transform<direction, Algorithm::Alt, false>(x, u, v, p); break;
  // case 3: y = transform<direction, Algorithm::Alt, true>(x, u, v, p); break;
  // default: {fprintf(stderr, "algorithm is incorrect"); exit(1); }
  // }
  y = transform<direction, Algorithm::Alt, false>(x, u, v, p);
  // y = transform_full<direction, Algorithm::Alt, false>(x, u, v, p);

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
    const bool shared_memory = false;
    auto y_reference = transform_full<Direction::Forwards, Algorithm::Alt, shared_memory>({{1, 0.}}, {{0., 0., z_offset}}, v, p);
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
