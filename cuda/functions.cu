#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>

#include "macros.h"
#include "hyper_params.h"
#include "util.h"
#include "init.h"
#include "kernel.cu"
#include "superposition.cu"

// host superposition functions

template<typename T = double>
inline void cp_batch_data_to_device(const T *v, T *v_pinned, DeviceVector<T> d_v, cudaStream_t stream) {
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
      (N, M, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);          \
  }

#define SuperpositionPerBlockHelper(blockDim_x) {                       \
    superposition_per_block_helper<direction, blockDim_x, algorithm, shared_memory> \
      (gridDim, blockDim, stream,                                       \
       N, M, d_x_ptr, d_u_ptr, d_v, d_y_tmp, append_result);          \
  }

template<Direction direction, unsigned int blockDim_x, Algorithm algorithm, bool shared_memory>
inline void superposition_per_block_helper(const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
                                           const size_t N, const size_t M,
                                           const WAVE *d_x_ptr, const SPACE *d_u_ptr, const SPACE *d_v,
                                           WAVE *d_y_tmp, const bool append_result)
// double *d_y_tmp_re, double *d_y_tmp_im)
{
  // unrolled for loop to allow constant blockDim
  // TODO add computation for shared memory size
  switch (blockDim.y) {
  case   1: SuperpositionPerBlock(  1) break;
  case   2: SuperpositionPerBlock(  2) break;
  // case   4: SuperpositionPerBlock(  4) break;
  // case   8: SuperpositionPerBlock(  8) break;
  // case  16: SuperpositionPerBlock( 16) break;
  // case  32: SuperpositionPerBlock( 32) break;
// #if KERNEL_SIZE >= 4
  case   4: SuperpositionPerBlock(  4) break;
// #endif
// #if KERNEL_SIZE >= 8
  case   8: SuperpositionPerBlock(  8) break;
// #endif
// #if KERNEL_SIZE >= 16
  case  16: SuperpositionPerBlock( 16) break;
// #endif
// #if KERNEL_SIZE >= 32
  case  32: SuperpositionPerBlock( 32) break;
// #endif
  case  64: SuperpositionPerBlock( 64) break;
  // case  64: { if (!shared_memory) SuperpositionPerBlock( 64) } break; // TODO
  // case 128: SuperpositionPerBlock(128) break;
  // case 256: SuperpositionPerBlock(256) break;
  // case 512: SuperpositionPerBlock(512) break;
  default: {fprintf(stderr, "BlockSize.y: %u not implemented\n", blockDim.y); exit(1);}
  }
}

template<Direction direction, Algorithm algorithm, bool shared_memory>
inline void superposition_per_block(const dim3 gridDim, const dim3 blockDim, cudaStream_t stream,
                                    const size_t N, const size_t M,
                                    const WAVE *d_x_ptr, const SPACE *d_u_ptr, const SPACE *d_v,
                                    WAVE *d_y_tmp, const bool append_result)
  // double *d_y_tmp_re, double *d_y_tmp_im)
{
  // unrolled for loop to allow constant blockDim
  // Note that the max number of threads per block is 1024
  switch (blockDim.x) {
  case   1: SuperpositionPerBlockHelper(  1) break;
  case   2: SuperpositionPerBlockHelper(  2) break;
  case   4: SuperpositionPerBlockHelper(  4) break;
// #if KERNEL_SIZE <= 128
  case   8: SuperpositionPerBlockHelper(  8) break;
// #endif
// #if KERNEL_SIZE <= 64
  case  16: SuperpositionPerBlockHelper( 16) break;
// #endif
// #if KERNEL_SIZE <= 32
  case  32: SuperpositionPerBlockHelper( 32) break;
// #endif
// #if KERNEL_SIZE <= 16
  case  64: SuperpositionPerBlockHelper( 64) break;
// #endif
// #if KERNEL_SIZE <= 8
  // case 128: SuperpositionPerBlockHelper(128) break;
// #endif
// #if KERNEL_SIZE <= 4
  // case 256: SuperpositionPerBlockHelper(256) break;
// #endif
// #if KERNEL_SIZE <= 2
  // case 512: SuperpositionPerBlockHelper(512) break;
// #endif
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
inline std::vector<WAVE> transform(const std::vector<WAVE> &x,
                                   const std::vector<SPACE> &u,
                                   const std::vector<SPACE> &v,
                                   const Geometry& p) {
#ifdef TEST_CONST_PHASE
  const size_t N = p.n.x;
  const size_t M = p.n.y;
#endif
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
  if (x.size() < p.gridSize.x)
    printf("Warning, suboptimal input size: %u < %u\n", x.size(), p.gridSize.x);

  // TODO duplicate stream batches to normal memory if too large
  auto y = std::vector<WAVE>(p.n.y);

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  const thrust::device_vector<WAVE> d_x = x;
  const thrust::device_vector<SPACE> d_u = u;
  // cast to pointers to allow usage in non-thrust kernels
  const WAVE* d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);

  // malloc data using pinned memory for all batches before starting streams
  // TODO consider std::unique_ptr<>
  WAVE *y_pinned_ptr, *d_y_tmp_ptr;
  SPACE *v_pinned_ptr, *d_v_ptr;
  // TODO don't use pinned memory for d_y_
  auto d_y_tmp  = init::malloc_vectors<WAVE>(        &d_y_tmp_ptr,  p.n_streams, tmp_out_size);
  auto d_v      = init::malloc_matrix<SPACE>(        &d_v_ptr,      p.n_streams, p.batch_size.y * DIMS);
  auto v_pinned = init::pinned_malloc_vectors<SPACE>(&v_pinned_ptr, p.n_streams, p.batch_size.y * DIMS);
  auto y_pinned = init::pinned_malloc_vectors<WAVE>( &y_pinned_ptr, p.n_streams, p.batch_size.y);

  // TODO d_b is too large
  // const auto d_unit = thrust::device_vector<WAVE>(p.batch_size.y, {1., 0.}); // unit vector for blas
  const auto d_unit = thrust::device_vector<WAVE>(tmp_out_size / p.batch_size.y, {1., 0.}); // unit vector for blas
  const auto *d_b = thrust::raw_pointer_cast(d_unit.data());

  cudaStream_t streams[p.n_streams];
  cublasHandle_t handles[p.n_streams];
  for (auto& stream : streams)
    cu( cudaStreamCreate(&stream) );

  for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
    cuB( cublasCreate(&handles[i_stream]) );
    cublasSetStream(handles[i_stream], streams[i_stream]);
  }

  for (size_t i = 0; i < p.n_batches.y; i+=p.n_streams) {
    // size_t batch_size_x = p.batch_size.x;
    for (size_t n = 0; n < p.n_batches.x; ++n) {
      // // each final x-batch may be under-used/occupied
      // if (n == p.n_batches.x - 1) batch_size_x = N - n * p.batch_size.x;
      if (i == 0) {
        // cp x batch data for all streams and sync
        // TODO
      }
      for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
        const auto m = i + i_stream;
        if (m >= p.n_batches.y) break;
        if (p.n_batches.y > 10 && m % (int) (p.n_batches.y / 10) == 0 && n == 0)
          printf("\tbatch %0.3fk / %0.3fk\n", m * 1e-3, p.n_batches.y * 1e-3);

        if (n == 0)
          cp_batch_data_to_device<SPACE>(&v[m * p.batch_size.y * DIMS],
                                         v_pinned[i_stream], d_v[i_stream],
                                         streams[i_stream]);

        const bool append_result = n > 0;
        const size_t xu_offset = n * p.batch_size.x;
        superposition_per_block<direction, algorithm, shared_memory> \
          (p.gridDim, p.blockDim, streams[i_stream], p.batch_size.x, p.batch_size.y,
           d_x_ptr + xu_offset, d_u_ptr + xu_offset * DIMS,
           d_v[i_stream].data, d_y_tmp[i_stream], append_result);
      }
#ifdef TEST_CONST_PHASE
      for (size_t i_stream = 0; i_stream < p.n_streams; ++i_stream) {
        const auto m = i + i_stream;
        if (m >= p.n_batches.y) break;
        cudaStreamSynchronize(streams[i_stream]);
        auto d = thrust::device_vector<WAVE> (d_y_tmp[i_stream], d_y_tmp[i_stream] + tmp_out_size);
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
      kernel::sum_rows<false>(tmp_out_size / p.batch_size.y, p.batch_size.y,
                              handles[i_stream], d_y_tmp[i_stream], d_b, d_y_tmp[i_stream]);
      // TODO transform `re, im => a, phi ` (complex to polar)
      cp_batch_data_to_host<WAVE>(d_y_tmp[i_stream], y_pinned[i_stream],
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
      for (size_t j = 0; j < p.batch_size.y; ++j)
        y[j + m * p.batch_size.y] = y_pinned[i_stream][j];
    }
  }

  // sync all streams before returning
  cudaDeviceSynchronize();
#ifdef TEST_CONST_PHASE
  for (size_t j = 0; j < M; ++j)
    assert(cuCabs(y[j]) == N);
#endif

#ifdef DEBUG
  printf("done, destroy streams\n");
#endif

  for (unsigned int i = 0; i < p.n_streams; ++i)
    cudaStreamDestroy(streams[i]);

#ifdef DEBUG
  printf("free device memory\n");
#endif

  for (auto& handle : handles)
    cuB( cublasDestroy(handle) );

  cu( cudaFree(d_y_tmp_ptr ) );
  cu( cudaFree(d_v_ptr       ) );
  cu( cudaFreeHost(v_pinned_ptr ) );
  cu( cudaFreeHost(y_pinned_ptr ) );

#ifdef DEBUG
  size_t len = min(100L, y.size());
  assert(std::any_of(y.begin(), y.begin() + len, abs_of_is_positive));
#endif
  return y;
}

/**
 * Time the transform operation over the full input.
 * Do a second transformation if add_reference is true.
 */
template<Direction direction, bool add_constant_wave = false, bool add_reference_wave = false>
std::vector<WAVE> time_transform(const std::vector<WAVE> &x,
                                 const std::vector<SPACE> &u,
                                 const std::vector<SPACE> &v,
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

  const bool shared_memory = true;
  // const bool shared_memory = false;
  // // auto y = transform<direction, Algorithm::Naive, shared_memory>(x, u, v, p);
  // auto y = transform<direction, Algorithm::Alt, shared_memory>(x, u, v, p);
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
    // auto y_reference = transform<direction, Algorithm::Naive, shared_memory>({from_polar(1.)}, {{0.,0., z_offset}}, v, p);
    auto y_reference = transform<direction, Algorithm::Alt, shared_memory>({from_polar(1.)}, {{0.,0., z_offset}}, v, p);
    normalize_amp<false>(y_reference, weights[2]);
    // let full reference wave (amp+phase) interfere with original wave
    add_complex(y, y_reference);
    // reset phase of result, because the projector is limited
    for (size_t i = 0; i < y.size(); ++i)
      y[i] = from_polar(cuCabs(y[i]), angle(y_reference[i]));
  }

  clock_gettime(CLOCK_MONOTONIC, t2);
  *dt = diff(*t1, *t2);
  if (verbose)
    print_result(std::vector<double>{*dt}, x.size(), y.size());

  return y;
}
