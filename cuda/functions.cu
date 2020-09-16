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
      bin_size, bins_per_thread,                                        \
       N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
    cu( cudaPeekAtLastError() );                                        \
  }
#else
#define SuperpositionPerBlock(blockDim_y) {                             \
    assert(blockDim_x * blockDim_y <= 1024);                            \
    superposition::per_block<direction, blockDim_x, blockDim_y, algorithm, shared_memory> \
      <<< gridDim, blockDim, 0, stream >>>                              \
      (N, M, width, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);     \
    cu( cudaPeekAtLastError() );                                        \
  }
#endif

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
#define SuperpositionPerBlockHelper(blockDim_x) {                       \
    superposition_per_block_helper<direction, blockDim_x, algorithm, shared_memory> \
      (state, seed, i_stream,                                           \
       bin_size, bins_per_thread,                                       \
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
                                           curandState *state, const unsigned int seed, const unsigned int i_stream,
                                           const unsigned int bin_size, const unsigned int bins_per_thread,
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
                                    curandState *state, const unsigned int seed, const unsigned int i_stream,
                                    const unsigned int bin_size, const unsigned int bins_per_thread,
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

  printf("\nmax amp: %e\n", max_amp);
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
                                   const Geometry& p,
                                   const Plane& x_plane,
                                   const Plane& y_plane) {

  auto v = v2;
  /* Performance (runtime) for randomize: 1, 1024x1024 data size
   * reshuffle_between_kernels: 0 -> 26.112164 s
   * reshuffle_between_kernels: 1 -> 20.338126 s // due to faster convergence? worst cases are averaged out
   */
  const double
    // max_width = x_plane.width > y_plane.width ? x_plane.width : y_plane.width,
    // max_height = x_plane.width > y_plane.width ? x_plane.width / x_plane.aspect_ratio : y_plane.width / y_plane.aspect_ratio,
    // min_distance = y_plane.offset.z - x_plane.offset.z, // assume parallel planes, with constant third dim
    // max_distance = norm3d_host(min_distance, max_width, max_height),
    max_distance = max_distance_between_planes(x_plane, u, y_plane, v),
    threshold = 1e-4;
  const bool
    randomize = 1 && p.n.x > 1 && p.n_batches.x > 1, // TODO rename => shuffle_source
    reshuffle_between_kernels = 1 && randomize, // each kernel uses mutually excl. random input data
    shuffle_v = 0,
    reorder_v = 1 && randomize,
    reorder_v_rm_phase = 0, // for debugging
    amp_convergence = 1;
  // TODO conditional shuffling, i.e. reorder x+u (source dataset) and only shuffle rows
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
  const bool conditional_MC = 0;
  assert(!conditional_MC); // TODO debug for min_n_datapoints > 2048
#endif
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
    // reorder v data s.t. each batch covers square area in space
    // neglect boundary cells

    // assume aspect ratio = 1
    size_t m_sqrt = (size_t) sqrt(p.n.y);
    assert(m_sqrt*m_sqrt == p.n.y);
    size_t G = (size_t) sqrt(p.batch_size.y); // batch_size
    size_t m_sqrt2 = FLOOR(m_sqrt, G) * G; // minus boundaries
    assert(m_sqrt > 0);

    for (size_t i = 0; i < m_sqrt2; ++i) {
      for (size_t j = 0; j < m_sqrt2; ++j) {
        // define spatial 2D indices (both for target dataset y)
        dim2
          i_batch_major = {i / G, j / G},
          i_batch_minor = {i % G, j % G};
        // size_t i_transpose = (i_batch_major.x * m_sqrt2/G + i_batch_major.y) * G*G + i_batch_minor.x * G + i_batch_minor.y;
        size_t i_transpose = (i_batch_major.x * m_sqrt2 + i_batch_major.y * G) * G + i_batch_minor.x * G + i_batch_minor.y;
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
  // auto converging   = std::vector<bool>(p.n_streams, false); // not strictly converging but "undergoing convergence checking"
  auto finished     = std::vector<bool>(p.n_streams, false);
  auto convergence_per_stream = std::vector<SumRange>(p.n_streams);
  // init with extreme values
  for (auto& i : range(convergence_per_stream.size()))
    convergence_per_stream[i] = {sum: 0, min: p.batch_size.x, max: 0};

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

  const size_t
    min_n_datapoints = MAX(4*1024, p.batch_size.x), // before convergence computation
    min_n_datapoints0 = MAX(8*1024, p.batch_size.x),
    // min_n_datapoints = 4,
    // i_shuffle_max = FLOOR(p.n_batches.x, p.n_streams); // number of batches between shuffles
    i_shuffle_max = p.n_batches.x; // TODO rm
  size_t
    batches_per_estimate = CEIL(min_n_datapoints, p.batch_size.x),
    batches_for_first_estimate = CEIL(min_n_datapoints0, p.batch_size.x),
    i_shuffle = i_shuffle_max; // init high s.t. shuffling will be triggered

  if (randomize && p.n.x > p.gridSize.x) {
    if (p.n_batches.x < p.n_streams)
      printf("\nWARNING: unused streams\tp.n_batches.x: %zu >= p.n_streams: %u\n--------------\n", p.n_batches.x, p.n_streams);
    assert(batches_per_estimate > 0);
    assert(i_shuffle_max >= 1);
    if (min_n_datapoints > p.n.x)
      printf("Warning not enough datapoints: n: %lu, n_min: %lu\n", p.n.x, min_n_datapoints);
    if (min_n_datapoints0 > p.n.x)
      printf("Warning not enough datapoints0: n: %lu, n_min: %lu\n", p.n.x, min_n_datapoints0);
  }

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

  size_t
    n_sample_bins = batches_per_estimate * p.batch_size.x, // per estimate
    sample_bin_size = CEIL(p.n.x, n_sample_bins),
    // each thread draws 1 sample per bin and covers thread_size.x bins per kernel call
    potential_batch_size = p.batch_size.x * sample_bin_size;

  if (p.n.x > p.gridSize.x) {
    printf("N: %zu, batches_per_estimate: %zu, batch_size.x: %zu, n_sample_bins: %zu, sample_bin_size: %zu\n",
           p.n.x, batches_per_estimate, p.batch_size.x, n_sample_bins, sample_bin_size);

    assert(batches_per_estimate >= 1);
    assert(p.n.x >= min_n_datapoints); // TODO not implemented
    assert(min_n_datapoints >= p.batch_size.x);
    assert(batches_per_estimate > 0);
    assert(batches_per_estimate * p.batch_size.x == min_n_datapoints);
    assert(n_sample_bins * sample_bin_size == p.n.x);
    assert(potential_batch_size <= p.n.x);
    assert(potential_batch_size * batches_per_estimate == p.n.x);
  }
#endif


  unsigned int print_convergence = 10;
  // const bool reduce_append_frequency = 0; // default false, optional performance improvement
  // TODO consider list of struct {stream, n, m}
  // {cudaStream_t stream, size_t n, size_t m, bool finished, bool converging};

  auto compare_func = [&](auto m){ return m < p.n_batches.y;};
  while (std::any_of(m_per_stream.begin(), m_per_stream.end(), compare_func)) {

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
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
          //   // conditional shuffling for conditional MC
          //   // TODO this requires reordering of x/u, -> make new reordering (transpose) function
          //   const size_t n_sqrt = sqrt(p.n.x);
          //   assert(n_sqrt * n_sqrt == p.n.x);
          //   for (size_t i = 0; i < n_sqrt; ++i) {
          //     const auto di = i * n_sqrt;
          //     thrust::scatter(d_x_original.begin() + di, d_x_original.begin() + (i+1) * n_sqrt, indices.begin() + di, d_x.begin() + di);
          //     thrust::scatter(d_u_original.begin() + di, d_u_original.begin() + (i+1) * n_sqrt, indices.begin() + di, d_u_row_ptr + di);
          //   }
          // }

        }
        i_shuffle = 0;
      }

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

      // Derive current batch size in case of underutilized x-batches
      size_t local_batch_size = p.batch_size.x;
      if (!randomize)
        if (p.n.x != p.n_batches.x * p.batch_size.x && n == p.n_batches.x - 1)
          local_batch_size = p.n.x - n * p.batch_size.x;

      // if (101) { // TODO rm
      //   finished[i_stream] = 1;
      //   // local_batch_size = 1;
      // }

      // size_t n_offset = n * p.batch_size.x;
      size_t n_offset = (reshuffle_between_kernels ? i_shuffle : n ) * p.batch_size.x;
      if (randomize && p.n.x > 1) {
        assert(i_shuffle <= p.n_batches.x);
        assert(i_shuffle < i_shuffle_max);
        // assert(i_shuffle < p.n_batches.x);
        if (p.n_batches.x * p.batch_size.x == p.n.x) {
          // printf("p.n_batches.x: %zu * p.batch_size.x: %zu == p.n.x: %zu, \tn_offset: %zu / %zu\n",
          //        p.n_batches.x, p.batch_size.x, p.n.x,
          //        n_offset, n_offset + p.batch_size.x);
          assert(( i_shuffle_max - 1 ) * p.batch_size.x <= p.n.x);
          assert(n_offset + p.batch_size.x <= p.n.x);
        } }

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
      if (p.n.x > p.gridSize.x) {
        assert(p.n.x == p.n_batches.x * p.batch_size.x); // underutilized x-batches would invalidate the estimate (E[Y|Z])
        assert(local_batch_size == p.batch_size.x);

        if (conditional_MC) {
          // TODO use with reorder_v
          n_offset = potential_batch_size * (n % batches_per_estimate);
          assert(n_offset < p.n.x);
          local_batch_size = potential_batch_size; // used n kernel e.g. as if (tid.x < N)
        } else {
          // unconditional MC
          n_offset = 0;
          local_batch_size = p.n.x;
          sample_bin_size = 0;
        }
      }
#endif

      // Note that appending result to prev results slows down computation
      // // const bool append_result = n > 0 && n_datapoints < min_n_datapoints;
      // const bool append_result = n > 0 && !converging[i_stream];
      // const bool append_result = (n > 0 && !converging[i_stream]) ||
      //   (converging[i_stream] && n % min_n_datapoints != 0);
      bool append_result = n > 0;
      // if (reduce_append_frequency && n+1 >= batches_per_estimate && (n+1) % batches_per_estimate != 0)
      //   append_result = 0;

      // cudaDeviceSynchronize(); print("pre s kernel");
      superposition_per_block<direction, algorithm, shared_memory>  \
        (
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
         rng_state, seed, i_stream, sample_bin_size, p.thread_size.x,
#endif
         p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
         p.batch_size.x,
         d_x_ptr + n_offset, d_u_ptr + n_offset * DIMS,
         d_v[i_stream].data, d_y_tmp[i_stream].data, append_result
         );
      cu( cudaPeekAtLastError() );
      // cudaDeviceSynchronize(); print("post s kernel");
      if (reshuffle_between_kernels) i_shuffle++;
    }

    // cudaDeviceSynchronize(); print("sum rows");
    for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const size_t
        n = n_per_stream[i_stream],
        m = m_per_stream[i_stream],
        n_datapoints = (n+1) * p.batch_size.x;
      // const bool converging = n+1 > batches_per_estimate; // strict comparison; true after the first aggregation
      const bool converging = n+1 > batches_for_first_estimate; // strict comparison; true after the first aggregation
      if (m >= p.n_batches.y) continue;

      if (n >= p.n_batches.x - 1)
        finished[i_stream] = true;

      // if (finished[i_stream] || n >= batches_per_estimate) {
      // if (finished[i_stream] || (n+1 >= batches_per_estimate && (n+1) % batches_per_estimate == 0)) {
      if (finished[i_stream] || (n+1 >= batches_for_first_estimate && (n+1) % batches_per_estimate == 0)) {
        // TODO instead of memset, do
        // after the first sum_rows call, the following results should be added the that first result
        // const auto beta = converging[i_stream] ? WAVE {1,0} : WAVE {0,0};
        // const auto beta = reduce_append_frequency && converging \
        //   ? WAVE {1,0} : WAVE {0,0};
        const auto beta = WAVE {0,0};
        kernel::sum_rows<false>(d_y_tmp[i_stream].size / p.batch_size.y, p.batch_size.y,
                                handles[i_stream], d_y_tmp[i_stream].data, d_unit,
                                d_y_sum[i_stream].data, beta);

        cu( cudaPeekAtLastError() );
        // cudaDeviceSynchronize(); print("sum rows post");

        if (!finished[i_stream] && converging && (n+1) % batches_per_estimate == 0) {
          const double
            prev_n = (n+1 - batches_per_estimate) * p.batch_size.x;

          // check either amp or both the re,im parts (implicit phase)
          // TODO (optional) save amplitude as separate array (only half n), to prevent the scaling and abs value at each comparison
          if (amp_convergence)
            finished[i_stream] = thrust::equal(thrust::cuda::par.on(streams[i_stream]),
                                               d_y_sum[i_stream].data,
                                               d_y_sum[i_stream].data + d_y_sum[i_stream].size,
                                               d_y_prev[i_stream].data,
                                               is_smaller_phasor(n_datapoints, prev_n, max_distance, threshold));
          else
            finished[i_stream] = thrust::equal(thrust::cuda::par.on(streams[i_stream]),
                                               (double *) d_y_sum[i_stream].data,
                                               (double *) d_y_sum[i_stream].data + d_y_sum[i_stream].size * 2,
                                               (double *) d_y_prev[i_stream].data,
                                               is_smaller(n_datapoints, prev_n, max_distance, threshold));

          if (finished[i_stream] && print_convergence > 0)
            printf("converged/finished at batch.x: %lu/%lu \t (%3u, threshold: %.2e)\n", n, p.n_batches.x, print_convergence--, threshold);

        }
        // if still not finished
        if (randomize && !finished[i_stream]) {
          // cp superpositions as previous result
          cu( cudaMemcpyAsync(d_y_prev[i_stream].data, d_y_sum[i_stream].data,
                              d_y_sum[i_stream].size * sizeof(WAVE),
                              cudaMemcpyDeviceToDevice, streams[i_stream]) );
          // converging[i_stream] = true;
        }
      }

      if (finished[i_stream])
        cp_batch_data_to_host<WAVE>(d_y_sum[i_stream].data, y_pinned[i_stream],
                                    p.batch_size.y, streams[i_stream]);
      else
        n_per_stream[i_stream]++;
    }

    // cudaDeviceSynchronize(); print("final");
    for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      if (finished[i_stream]) {
        const size_t
          n = n_per_stream[i_stream],
          m = m_per_stream[i_stream];
        assert(m < p.n_batches.y);
        const double div_by_n_datapoints = 1. / (double) ((n+1) * p.batch_size.x);
        // TODO add dedicated loop for better work distribution (similar to in func transform_full)
        // TODO stage copy-phase of next batch before copy/sync?
        cudaStreamSynchronize(streams[i_stream]);
        for (size_t j = 0; j < p.batch_size.y; ++j) {
          // save average w.r.t the number of samples used per y-batch
          // TODO make transformation_full compatible with this average
          y[j + m * p.batch_size.y].x = y_pinned[i_stream][j].x * div_by_n_datapoints;
          y[j + m * p.batch_size.y].y = y_pinned[i_stream][j].y * div_by_n_datapoints;
          // TODO assert y index is within bounds
        }

        n_per_stream[i_stream] = 0;
        m_per_stream[i_stream] = *std::max_element(m_per_stream.begin(), m_per_stream.end()) + 1;
        // converging[i_stream] = false;
        // converged[i_stream] = false;
        finished[i_stream] = false;

        // update statistics summary
        convergence_per_stream[i_stream].sum += n+1;
        if (n+1 < convergence_per_stream[i_stream].min)
          convergence_per_stream[i_stream].min = n+1;
        if (n+1 > convergence_per_stream[i_stream].max)
          convergence_per_stream[i_stream].max = n+1;
      }
    }
  } // end while (m < p.n_batches.y)

  cu( cudaPeekAtLastError() );
  cu( cudaDeviceSynchronize() );
  curandDestroyGenerator(generator);

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


  {
    size_t
      min = std::accumulate(convergence_per_stream.begin(), convergence_per_stream.end(), convergence_per_stream[0].min,
                            [](auto acc, auto next) {return MIN(acc, next.min); }),
      max = std::accumulate(convergence_per_stream.begin(), convergence_per_stream.end(), convergence_per_stream[0].max,
                            [](auto acc, auto next) {return MAX(acc, next.max); });
    double
      sum = transform_reduce<SumRange>(convergence_per_stream, [](auto x) {return (double) x.sum; }),
      mean = sum / (double) p.n_batches.y;
    printf("Convergence ratio: %.4f%%, \t%.3f / %lu, range: [%lu, %lu] \t(min_n_datapoints/batch_size.x: %.3f)\n",
           100 * mean / (double) p.n_batches.x, mean, p.n_batches.x, min, max, min_n_datapoints / (double) p.batch_size.x);
    if (p.n.x > p.batch_size.x)
      assert(sum <= p.n_batches.y * p.n_batches.x);
    if (p.n.x > p.batch_size.x)
      assert(mean <= p.n_batches.x);
  }

  // only in case of second transformation
  if (reorder_v && p.n.x > p.batch_size.x) {
    // revert y data (v data)
    assert(!shuffle_v);
    // assume aspect ratio = 1
    auto y2 = y;
    size_t m_sqrt = (size_t) sqrt(p.n.y);
    assert(m_sqrt*m_sqrt == p.n.y);
    size_t G = (size_t) sqrt(p.batch_size.y); // batch_size
    size_t m_sqrt2 = FLOOR(m_sqrt, G) * G; // minus boundaries
    assert(m_sqrt > 0);

    for (size_t i = 0; i < m_sqrt; ++i) {
      for (size_t j = 0; j < m_sqrt; ++j) {
        y2[i * m_sqrt + j] = {0,0};
      } }
    for (size_t i = 0; i < m_sqrt; ++i) {
      for (size_t j = 0; j < m_sqrt; ++j) {
        if (i >= m_sqrt2 || j >= m_sqrt2) {
          // clear unused boundary indices
          y2[i * m_sqrt + j] = from_polar(1, 0.111);
          continue;
        }
        // define spatial 2D indices (both for target dataset y)
        dim2
          i_batch_major = {i / G, j / G},
          i_batch_minor = {i % G, j % G};
        // size_t i_transpose = (i_batch_major.x * m_sqrt2/G + i_batch_major.y) * G*G + i_batch_minor.x * G + i_batch_minor.y;
        size_t i_transpose = (i_batch_major.x * m_sqrt2 + i_batch_major.y * G) * G + i_batch_minor.x * G + i_batch_minor.y;
        assert(i_transpose < p.n.y);
        // if (i_transpose >= m_sqrt2*m_sqrt2)
        // if (i_batch_major.x == 0 && i_batch_major.y < 2 && i_batch_minor.y == 0)
        //   printf("i_t: %zu / %zu \t[%zu,%zu] (/ %zu)\t[%zu,%5u]\t(%zu^2 = %zu)\n", i_transpose, m_sqrt2*m_sqrt2,
        //          i_batch_major.x, i_batch_major.y, m_sqrt2 / G,
        //          i_batch_minor.x, i_batch_minor.y,
        //          G, p.batch_size.y);
        assert(i_transpose < m_sqrt2*m_sqrt2);
        y2[i * m_sqrt + j] = y[i_transpose];
        // y2[j * m_sqrt + i] = y[i_transpose];
        // y2[i * m_sqrt + j] = from_polar(i_batch_minor.x / (double) G, i_batch_minor.y / (double) G);

        // // add offset to avoid zero amp, which can influence phase
        // y2[i * m_sqrt + j] = from_polar(1 + (i_batch_minor.x * G + i_batch_minor.y) / (double) G,
        //                                 1 + (i_batch_major.x * G*m_sqrt2 + i_batch_major.y * G) / (double) p.n.y);
        // y2[i * m_sqrt + j] = from_polar(1, rand());

            if (reorder_v_rm_phase) {
              if (i_batch_major.x % 2 == 0)
                if (i_batch_major.y % 2 == 0)
                  y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 0);
                else
                  y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 0.5);
              else
                if (i_batch_major.y % 2 == 0)
                  y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 1);
                else
                  y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 1.5);
            }

      } }

    // for (size_t i = 0; i < m_sqrt2; ++i) {
    //   for (size_t j = 0; j < m_sqrt2; ++j) {
    //     dim2
    //       i_batch = {i / G, j / G},
    //       g = {i % G, j % G};
    //     size_t i_transpose = (i_batch.x * m_sqrt2/G + i_batch.y) * G*G + g.x * G + g.y;
    //     y2[i * m_sqrt + j] = y[i_transpose];
    //     if (reorder_v_rm_phase) {
    //       if (i_batch.x % 2 == 0)
    //         if (i_batch.y % 2 == 0)
    //           y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 0);
    //         else
    //           y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 1.5);
    //       else
    //         if (i_batch.y % 2 == 0)
    //           y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 3);
    //         else
    //           y2[i * m_sqrt + j] = from_polar(cuCabs(y2[i * m_sqrt + j]), 4.5);
    //     }
    //   }
    // }
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

#ifdef RANDOMIZE_SUPERPOSITION_INPUT
        assert(0); // not implemented (TODO)
        curandState rng_placeholder;
#endif
        superposition_per_block<direction, algorithm, shared_memory> \
          (
#ifdef RANDOMIZE_SUPERPOSITION_INPUT
           &rng_placeholder, 0,0,0,0,
#endif
           p.gridDim, p.blockDim, streams[i_stream], local_batch_size, p.batch_size.y,
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
                                  const Plane& x_plane,
                                  const Plane& y_plane,
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
// #ifdef RANDOMIZE_SUPERPOSITION_INPUT
  y = transform<direction, Algorithm::Alt, false>(x, u, v, p, x_plane, y_plane);
// #else
  // y = transform_full<direction, Algorithm::Alt, false>(x, u, v, p);
// #endif

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
