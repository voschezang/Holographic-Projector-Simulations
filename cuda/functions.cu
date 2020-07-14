#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "macros.h"
#include "hyper_params.h"
#include "util.h"
#include "init.h"
#include "kernel.cu"
#include "superposition.cu"

// host superposition functions

template<typename T = double>
inline void cp_batch_data_to_device(const T *v, DeviceVector<T> d_v, cudaStream_t stream) {
  // copy "v[i]" for i in batch, where v are the spatial positions belonging to the target datapoints y
  cu ( cudaMemcpyAsync( d_v.data, v, d_v.size * sizeof(T),
                        cudaMemcpyHostToDevice, stream ) );
}

#define SuperpositionPerBlock(size) {                                   \
    /* const size_t local_memory_size = p.kernel_size * sizeof(WAVE);  */ \
    /* TODO add SHARED_MEMORY_SIZE * sizeof(WAVE) */                    \
    superposition::per_block<direction, size><<< p.gridSize, p.blockSize, 0, stream >>> \
    (p, d_x, Nx, d_u, &d_v[k * DIMS], &d_y_block[j] );                  \
  }

template<Direction direction>
inline void partial_superposition_per_block(const Geometry& p, const size_t Nx,
                                            const WAVE *d_x, const SPACE *d_u, SPACE *d_v,
                                            cudaStream_t stream, double *d_y_block)
{
  assert(p.blockSize <= 512); // not implemented
  for (unsigned int i = 0; i < p.batch_size; ++i) {
    const unsigned int j = i * p.gridSize * p.kernel_size; // * 2
    const unsigned int k = i * p.kernel_size;
    switch (p.blockSize) {
    case   1: SuperpositionPerBlock(  1) break;
    case   2: SuperpositionPerBlock(  2) break;
    case   4: SuperpositionPerBlock(  4) break;
    case   8: SuperpositionPerBlock(  8) break;
    case  16: SuperpositionPerBlock( 16) break;
    case  32: SuperpositionPerBlock( 32) break;
    case  64: SuperpositionPerBlock( 64) break;
    case 128: SuperpositionPerBlock(128) break;
    case 256: SuperpositionPerBlock(256) break;
    case 512: SuperpositionPerBlock(512) break;
    default: printf("BlockSize incorrect\n");
    }
/* #pragma unroll */
/*     for (unsigned int size = 1; size < 2048; size*=2) */
/*       if (size == p.blockSize) */
/*         { */
          /* const int s = size; */
        /* } */
  }
}

inline void agg_batch_blocks_naive(const size_t N, const size_t M, cudaStream_t stream,
                                   DeviceVector<double> d_y_batch,
                                   double *d_y_block) {
  // aggregate d_y_block and save to d_y_batch
  // launch 1x1 kernel in the specified selected stream, from which multiple thrust are called indirectly
  thrust::device_ptr<double>
    x (d_y_block),
    y (d_y_batch.data);
  kernel::reduce_rows<<< 1,1,0, stream >>>(x, N, M, 0.0, thrust::plus<double>(), y);
}

inline void agg_batch_blocks(const Geometry& p, cudaStream_t stream,
                             DeviceVector<double> d_y_batch,
                             double *d_y_block) {
  // aggregate d_y_block and save to d_y_batch
  auto y1 = thrust::device_ptr<double>(&d_y_batch.data[0]);
  auto y2 = thrust::device_ptr<double>(&d_y_batch.data[d_y_batch.size / 2]);
  // TODO is a reduction call for each datapoint really necessary?
  for (unsigned int m = 0; m < p.n_per_batch; ++m) {
    // Assume two independent reductions are at least as fast as a large reduction.
    // I.e. no kernel overhead and better work distribution
    thrust::device_ptr<double> ptr(d_y_block + m * p.gridSize);

    // launch 1x1 kernels in selected streams, which calls thrust indirectly inside that stream
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + p.gridSize, 0.0, thrust::plus<double>(), &y1[m]);
    ptr += p.gridSize * p.n_per_batch;
    kernel::reduce<<< 1,1,0, stream >>>(ptr, ptr + p.gridSize, 0.0, thrust::plus<double>(), &y2[m]);
  }
}

inline void agg_batch_naive(const size_t half_n, WAVE *y,
                            cudaStream_t stream, double *d_y_batch, WAVE *d_y_stream) {
  // TODO replace zip by `re, im => a, phi ` (complex to polar)
  kernel::zip_arrays<<< KERNEL_SIZE, 1, 0, stream >>>(d_y_batch, d_y_batch + half_n, half_n, d_y_stream);
	cu( cudaMemcpyAsync(y, d_y_stream, half_n * sizeof(WAVE), cudaMemcpyDeviceToHost, stream ) );
}

inline void agg_batch(const Geometry& p, WAVE *y, cudaStream_t stream,
                      WAVE *d_y_stream, double *d_y_batch) {
  // wrapper for thrust call using streams
  // TODO replace zip by `re, im => a, phi ` (complex to polar)
  kernel::zip_arrays<<< 1,1 >>>(d_y_batch, &d_y_batch[p.n_per_batch], p.n_per_batch, d_y_stream);
	cu( cudaMemcpyAsync(y, d_y_stream, p.n_per_batch * sizeof(WAVE),
                      cudaMemcpyDeviceToHost, stream ) );
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


template<Direction direction, Algorithm algorithm = Algorithm::Naive>
inline std::vector<WAVE> transform_naive(const std::vector<WAVE> &x,
                                         const std::vector<SPACE> &u,
                                         const std::vector<SPACE> &v,
                                         const Geometry& p) {
  assert(u[2] != v[2]);
  const size_t N = u.size() / DIMS;
  const size_t M = v.size() / DIMS;
  const dim3
    gridDim = {(unsigned int) p.gridSize, 1, 1},
    blockDim = {(unsigned int) p.blockSize, KERNEL_SIZE, 1},
    gridSize = {blockDim.x * gridDim.x,
                blockDim.y * gridDim.y};

  // size_t gridSize = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  size_t batch_out_size = MIN(N, gridSize.x) * p.n_per_batch;
  if (algorithm == Algorithm::Naive)
    batch_out_size = N * p.n_per_batch;
  printf("batch out size %lu\n", batch_out_size);

#ifdef DEBUG
  assert(std::any_of(x.begin(), x.end(), abs_of_is_positive));
#endif
  if (x.size() < p.gridSize * p.blockSize)
    printf("Warning, suboptimal input size: %f < %f", x.size(), p.gridSize * p.blockSize);

  auto y = std::vector<WAVE>(M);

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  const thrust::device_vector<WAVE> d_x = x;
  const thrust::device_vector<SPACE> d_u = u;
  // cast to pointers to allow usage in non-thrust kernels
  const auto d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);

  cudaStream_t streams[p.n_streams];
  // malloc data using pinned memory for all batches before starting streams
  // TODO consider std::unique_ptr<>
  WAVE *d_y_stream_ptr;
  double *d_y_block_ptr;
  double *d_y_batch_ptr;
  SPACE *d_v_ptr;
  auto d_y_block  = init::pinned_malloc_vector<double>(&d_y_block_ptr, p.n_streams, 2 * batch_out_size);
  auto d_y_batch  = init::pinned_malloc_matrix<double>(&d_y_batch_ptr, p.n_streams, 2 * p.n_per_batch);
  auto d_y_stream = init::pinned_malloc_vector<WAVE>(&d_y_stream_ptr, p.n_streams, p.n_per_batch); // TODO rm
  auto d_v        = init::pinned_malloc_matrix<SPACE>(&d_v_ptr, p.n_streams, p.n_per_batch * DIMS);

  for (auto& stream : streams)
    cudaStreamCreate(&stream);

  for (size_t i = 0; i < p.n_batches; i+=p.n_streams) {
    // start each distinct kernel in batches

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto i_batch = i + i_stream;
      if (p.n_batches > 10 && i_batch % (int) (p.n_batches / 10) == 0)
        printf("\tbatch %0.3fk / %0.3fk\n", i_batch * 1e-3, p.n_batches * 1e-3);

      cp_batch_data_to_device<SPACE>(&v[i_batch * p.n_per_batch * DIMS], d_v[i_stream],
                                     streams[i_stream]);

      // const size_t k = i_batch * p.n_per_batch;
      superposition::per_block_naive<direction, algorithm>              \
        <<< gridDim, blockDim, 0, streams[i_stream] >>>                 \
        (p, N, p.n_per_batch, d_x_ptr, d_u_ptr, d_v[i_stream].data,
         d_y_block[i_stream], d_y_block[i_stream] + batch_out_size );
    }

    // do aggregations in separate stream-loops because of imperfect async functions calls on host
    // this may yield a ~2.5x speedup
    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      agg_batch_blocks_naive(batch_out_size / p.n_per_batch, p.n_per_batch * 2,
                             streams[i_stream],
                             d_y_batch[i_stream],
                             d_y_block[i_stream]);
    }

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto i_batch = i + i_stream;
      agg_batch_naive(p.n_per_batch, &y[i_batch * p.n_per_batch],
                      streams[i_stream], d_y_batch[i_stream].data, d_y_stream[i_stream]);
    }
  }

  // sync all streams before returning
  cudaDeviceSynchronize();

#ifdef DEBUG
  printf("done, destroy streams\n");
#endif

  for (unsigned int i = 0; i < p.n_streams; ++i)
    cudaStreamDestroy(streams[i]);

#ifdef DEBUG
  printf("free device memory\n");
#endif

  cu( cudaFreeHost(d_y_stream_ptr) );
  cu( cudaFreeHost(d_y_batch_ptr ) );
  cu( cudaFreeHost(d_y_block_ptr ) );
  cu( cudaFreeHost(d_v_ptr       ) );

#ifdef DEBUG
  size_t len = min(100L, y.size());
  assert(std::any_of(y.begin(), y.begin() + len, abs_of_is_positive));
#endif
  return y;
}


/**
 * d_x, d_u are stored in normal (non-pinned) GPU memory
 * d_y, d_v are stored partially, and copied back to CPU on the fly
 *
 * Additional temporary memory:
 * d_y_stream = reserved memory for each stream, containing batch result as complex doubles.
 * d_y_batch  = batch results, using doubles because thrust doesn't support cuComplexDouble
 * d_y_block  = block results (because blocks cannot sync), aggregated by thrust
 */
template<Direction direction>
inline std::vector<WAVE> transform(const std::vector<WAVE> &x,
                                    const std::vector<SPACE> &u,
                                    const std::vector<SPACE> &v,
                                    const Geometry& p) {
  assert(u[2] != v[2]);
  const size_t n = v.size() / DIMS;
#ifdef DEBUG
  assert(std::any_of(x.begin(), x.end(), abs_of_is_positive));
#endif
  if (x.size() < p.gridSize * p.blockSize)
    print("Warning, suboptimal input size");

  auto y = std::vector<WAVE>(n);

  // Copy CPU data to GPU, don't use pinned (page-locked) memory for input data
  const thrust::device_vector<WAVE> d_x = x;
  const thrust::device_vector<SPACE> d_u = u;
  // cast to pointers to allow usage in non-thrust kernels
  const auto d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);

  // Note that in case x.size < GRIDDIM the remaining entries in the agg array are zero
  cudaStream_t streams[p.n_streams];
  // malloc data using pinned memory for all batches before starting streams
  // TODO consider std::unique_ptr<>
  WAVE *d_y_stream_ptr;
  double *d_y_block_ptr;
  double *d_y_batch_ptr;
  SPACE *d_v_ptr;
  auto d_y_stream = init::pinned_malloc_vector<WAVE>(&d_y_stream_ptr, p.n_streams, p.n_per_batch);
  auto d_y_block  = init::pinned_malloc_vector<double>(&d_y_block_ptr, p.n_streams, 2 * p.n_per_batch * p.gridSize);
  auto d_y_batch  = init::pinned_malloc_matrix<double>(&d_y_batch_ptr, p.n_streams, 2 * p.n_per_batch);
  auto d_v        = init::pinned_malloc_matrix<SPACE>(&d_v_ptr, p.n_streams, p.n_per_batch * DIMS);

  for (auto& stream : streams)
    cudaStreamCreate(&stream);

  // assume n_batches is divisible by n_streams
  for (size_t i = 0; i < p.n_batches; i+=p.n_streams) {
    // start each distinct kernel in batches
    // TODO don't do this in case of non-uniform workloads

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto i_batch = i + i_stream;
      if (p.n_batches > 10 && i_batch % (int) (p.n_batches / 10) == 0)
        printf("\tbatch %0.3fk / %0.3fk\n", i_batch * 1e-3, p.n_batches * 1e-3);

      cp_batch_data_to_device<SPACE>(&v[i_batch * p.n_per_batch * DIMS], d_v[i_stream],
                                     streams[i_stream]);

      partial_superposition_per_block<direction>(p, x.size(), d_x_ptr, d_u_ptr,
                                                 d_v[i_stream].data,
                                                 streams[i_stream], d_y_block[i_stream]);
    }

    // do aggregations in separate stream-loops because of imperfect async functions calls on host
    // this may yield a ~2.5x speedup
    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      agg_batch_blocks(p, streams[i_stream],
                       d_y_batch[i_stream],
                       d_y_block[i_stream]);
    }

    for (unsigned int i_stream = 0; i_stream < p.n_streams; ++i_stream) {
      const auto i_batch = i + i_stream;
      agg_batch(p, &y[i_batch * p.n_per_batch],
                streams[i_stream],
                d_y_stream[i_stream],
                d_y_batch[i_stream].data);
    }
  }

  // sync all streams before returning
  cudaDeviceSynchronize();

#ifdef DEBUG
  printf("done, destroy streams\n");
#endif

  for (unsigned int i = 0; i < p.n_streams; ++i)
    cudaStreamDestroy(streams[i]);

#ifdef DEBUG
  printf("free device memory\n");
#endif

  cu( cudaFreeHost(d_y_stream_ptr) );
  cu( cudaFreeHost(d_y_batch_ptr ) );
  cu( cudaFreeHost(d_y_block_ptr ) );
  cu( cudaFreeHost(d_v_ptr       ) );

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
  // auto y = transform<direction>(x, u, v, p);
  // auto y = transform_naive<direction, Algorithm::Naive>(x, u, v, p);
  auto y = transform_naive<direction, Algorithm::Alt>(x, u, v, p);
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
    assert(0);
    const double z_offset = v[2] - DISTANCE_REFERENCE_WAVE; // assume v[:, 2] is constant
    auto y_reference = transform<Direction::Forwards>({from_polar(1.)}, {{0.,0., z_offset}}, v, p);
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
