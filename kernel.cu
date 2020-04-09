#ifndef KERNEL
#define KERNEL

#include <assert.h>
// #include <math.h>
#include <stdio.h>
#include <time.h>
// #include <complex.h>
#include <cuComplex.h>

#include "macros.h"

#define cu(result) { cudaCheck((result), __FILE__, __LINE__); }

inline
cudaError_t cudaCheck(cudaError_t result, const char *file, int line)
{
  // check for cuda errors
// #ifdef DEBUG
  if (result != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA Runtime Error: %s\n", file, line, cudaGetErrorString(result));
    // assert(result == cudaSuccess);
    exit(result);
  }
// #endif
  return result;
}

__host__ __device__ void cuCheck(cuDoubleComplex  z) {
  double a = cuCreal(z), b = cuCimag(z);
  if (isnan(a)) printf("cu found nan re\n");
  if (isinf(a)) printf("cu found inf re\n");
  if (isnan(b)) printf("cu found nan I\n");
  if (isinf(b)) printf("cu found inf I\n");
}

inline __host__ __device__ double angle(cuDoubleComplex  z) {
  return atan2(cuCreal(z), cuCimag(z));
}

inline __device__ cuDoubleComplex polar(double a, double phi) {
  // Convert polar coordinates (a,phi) to complex number a * e^(phi I)
  cuDoubleComplex res;
  sincos(phi, &res.x, &res.y);
  // return cuCmul(make_cuDoubleComplex(a, 0), res);
  return make_cuDoubleComplex(a * res.x, a * res.y);
}


__global__ void kernel_zero(WTYPE_cuda *x, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < n; i += stride)
    x[i] = ZERO;
}

inline __device__ WTYPE_cuda superposition_single(const size_t i, const size_t j,
                        const WTYPE_cuda *x, const STYPE *u, STYPE *v,
                        const char direction) {
  // TODO consider unguarded functions, intrinsic functions
#ifdef DEBUG
  assert(direction == -1 || direction == 1);
#endif

  const size_t
    n = i * DIMS,
    m = j * DIMS;
  // TODO use softeningSquared?
  const double
    distance = norm3d(v[m] - u[n], v[m+1] - u[n+1], v[m+2] - u[n+2]),
    amp = cuCabs(x[i]),
    phase = angle(x[i]);

#ifdef DEBUG
  if (distance == 0) { printf("ERROR: distance must be nonzero\n"); asm("trap;"); }
#endif
  // TODO __ddiv_rd, __dmul_ru
  return polar(amp / distance, phase - distance * direction * TWO_PI_OVER_LAMBDA);
}

inline __device__ void superposition_partial(WTYPE_cuda *x, STYPE *u, WTYPE_cuda *y_local, STYPE *v, const char direction) {
#ifdef CACHE_BATCH
  // cache v[batch] because it is read by every thread
  // v_cached is constant and equal for each block
  __shared__ STYPE v_cached[KERNEL_BATCH_SIZE * DIMS];
  // use strides when BLOCKDIM < BATCH_SIZE * DIMS
  for (unsigned int i = threadIdx.x; i < KERNEL_BATCH_SIZE * DIMS; i+=BLOCKDIM)
    v_cached[i] = v[i];

  __syncthreads();
#else
  STYPE *v_cached = v;
#endif

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  WTYPE_cuda sum;

  // for each y-datapoint in current batch
  // TODO test performance diff when switching inner/outer loop and with um cache
  // TODO change cache size and find new optimal batch size w/ um cache
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
    // add single far away light source, with arbitrary (but constant) phase
    // assume threadIdx.x is a runtime constant
    if (direction == -1 && threadIdx.x == 0 && idx == 0) sum = polar(1, 0.4912);
    else sum = ZERO;

    // Usage of stride allows <<<1,1>>> kernel invocation
    for (size_t i = idx; i < N; i += stride)
      sum = cuCadd(superposition_single(i, m, x, u, v_cached, direction), sum);

    // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = sum;
    y_local[m] = sum;
#ifdef DEBUG
    cuCheck(sum);
#endif
  }
}

inline __device__ void superposition_cp_result(WTYPE_cuda *local, WTYPE_cuda *tmp) {
  // TODO rename tmp => shared
#if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1)
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
    tmp[threadIdx.x + m * BLOCKDIM] = local[m];
    // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = local[m];
#else
  // write before first sync. note that integer division is used

  if (KERNEL_BATCH_SIZE < 2) {
    // use 1/2 a much shared memory

    const unsigned int halfBlockSize = BLOCKDIM / 2;
    // simple reduction, first half writes first, then second half add their result
    if (threadIdx.x >= halfBlockSize)
      for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
        tmp[threadIdx.x - halfBlockSize + m * BLOCKDIM] = local[m];
        // tmp[m + (threadIdx.x - halfBlockSize ) * KERNEL_BATCH_SIZE] = local[m];

    __syncthreads();
    if (threadIdx.x < halfBlockSize)
      for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
        tmp[threadIdx.x + m * KERNEL_BATCH_SIZE] = \
          cuCadd(local[m], tmp[threadIdx.x + m * KERNEL_BATCH_SIZE]);
        // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = \
        //   cuCadd(local[m], tmp[m + threadIdx.x * KERNEL_BATCH_SIZE]);
  }
  else {
    // still use 1/2 of shared memory, but without idle threads
    // TODO consider another step, resulting in 1/4
    // dm  = threadIdx.x < halfBlockSize ? 0 : BLOCKSIZE * KERNEL_BATCH_SIZE / 2,

    const unsigned int size = BLOCKDIM / REDUCE_SHARED_MEMORY;
    // const unsigned int halfBlockSize = BLOCKDIM / 2;
    const unsigned int idx = \
      threadIdx.x < size ? threadIdx.x : threadIdx.x - size;
    {
      const unsigned int
        dm  = threadIdx.x < size ? 0 : KERNEL_BATCH_SIZE / 2;
      for(unsigned int m = 0; m < KERNEL_BATCH_SIZE / 2; ++m) {
        // first half writes first half of batch, idem for second half
        // tmp[(m + dm) + idx * KERNEL_BATCH_SIZE] = local[m + dm];
        tmp[idx + (m + dm) * size] = local[m + dm];
      }
    }

    __syncthreads();
    const unsigned int
      dm = threadIdx.x < size ? KERNEL_BATCH_SIZE / 2 : 0;
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE / 2; ++m) {
      // first half adds+writes first half of batch, idem for second half
      const unsigned int i = idx + (m + dm) * size;
      // const unsigned int i = (m + dm) + idx * KERNEL_BATCH_SIZE;
      tmp[i] = cuCadd(tmp[i], local[m + dm]);
    }

#if (REDUCE_SHARED_MEMORY == 4)
    not implemented // TODO
#endif

  } // end else
#endif

  __syncthreads();
}

template <unsigned int blockSize, typename T>
inline __device__ void warp_reduce(volatile T *s, unsigned int i) {
  // example code from Nvidia
  if (blockSize >= 64) s[i] += s[i + 32];
  if (blockSize >= 32) s[i] += s[i + 16];
  if (blockSize >= 16) s[i] += s[i +  8];
  if (blockSize >=  8) s[i] += s[i +  4];
  if (blockSize >=  4) s[i] += s[i +  2];
  if (blockSize >=  2) s[i] += s[i +  1]; // TODO rm last line
}

// volatile WTYPE_cuda& operator=(volatile WTYPE_cuda&) volatile;

template <unsigned int size, typename T>
inline __device__ void warp_reduce_c(Volatile T *s, const unsigned int i) {
  // example code from Nvidia

  if (size >= 64) s[i] = cuCadd(s[i], s[i + 32]);
  __threadfence();
  if (size >= 32) s[i] = cuCadd(s[i], s[i + 16]);
  __threadfence();
  if (size >= 16) s[i] = cuCadd(s[i], s[i +  8]);
  __threadfence();
  if (size >=  8) s[i] = cuCadd(s[i], s[i +  4]);
  __threadfence();
  if (size >=  4) s[i] = cuCadd(s[i], s[i +  2]);
  __threadfence();
  if (size >=  2) s[i] = cuCadd(s[i], s[i +  1]); // TODO rm last line
  __threadfence();
}

inline __device__ void superposition_agg(WTYPE_cuda *tmp, double *y) {
  const unsigned int tid = threadIdx.x;
  const unsigned int size = BLOCKDIM / REDUCE_SHARED_MEMORY;

  // inter warp
  // TODO do this in parallel for the next warp in case of next batch
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
    // TOOD check for memory bank conflicts
    WTYPE_cuda *x = &tmp[m * size];
    if (size > 512) assert(0);
    if(size == 512){if(tid < 256){ x[tid] = cuCadd(x[tid], x[tid + 256]); __syncthreads();}}
    if(size >= 256){if(tid < 128){ x[tid] = cuCadd(x[tid], x[tid + 128]); __syncthreads();}}
    if(size >= 128){if(tid <  64){ x[tid] = cuCadd(x[tid], x[tid +  64]); __syncthreads();}}
  }


  // final intra warp aggregation
  // TODO do this in parallel for the next warp in case of next batch
  const unsigned int n_warps = BLOCKDIM / WARP_SIZE;
  const unsigned int wid = tid / WARP_SIZE;
  const unsigned int lane = tid % 32;
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; m+=n_warps)
    for(unsigned int w = 0; w < n_warps; ++w)
      if (wid == w
          && (m+w) < KERNEL_BATCH_SIZE
          && lane < size)
        warp_reduce_c<size, WTYPE_cuda>(&tmp[(m+w) * size], lane);

  // for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
  //   if (tid < WARP_SIZE && tid < size)
  //     warp_reduce_c<size>(&tmp[m * size], tid);


#if (BLOCKDIM >= KERNEL_BATCH_SIZE)
  if (tid < KERNEL_BATCH_SIZE) {
    unsigned int m = tid;
#else
    // for each y-datapoint in current batch
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
#endif

    WTYPE_cuda sum;
    // sum = ZERO;
    sum = tmp[m * size];

#ifdef DEBUG
    cuCheck(sum);
#endif

    const unsigned int i = blockIdx.x + m * GRIDDIM;
    y[i] = sum.x;
    y[i + GRIDDIM * BATCH_SIZE] = sum.y; // note the use of stream batch size
    // }
  }

  // do not sync blocks, exit kernel and agg block results locally or in diff kernel
}

// TODO optimize memory / prevent Shared memory bank conflicts for x,u arrays
// TODO use __restrict__, const
// TODO template<unsigned int blockSize>
__global__ void kernel3(WTYPE_cuda *x, STYPE *u, double *y, STYPE *v,
                        const char direction)
{
  /** First compute local sum, then do nested aggregation
   *
   * out[BATCH_SIZE * blockDim] = array with output per block
   * v[BATCH_SIZE * DIM] = locations of y-datapoints
   */
  //
  // TODO use shared mem for u-data
#if (REDUCE_SHARED_MEMORY > 1 && KERNEL_BATCH_SIZE >= REDUCE_SHARED_MEMORY)
  __shared__ WTYPE_cuda tmp[KERNEL_BATCH_SIZE * BLOCKDIM / REDUCE_SHARED_MEMORY];
#else
  __shared__ WTYPE_cuda tmp[KERNEL_BATCH_SIZE * BLOCKDIM];
#endif
  // TODO transpose tmp array? - memory bank conflicts
  {
    WTYPE_cuda local[KERNEL_BATCH_SIZE];
    superposition_partial(x, u, local, v, direction);
    superposition_cp_result(local, tmp);
  }
  superposition_agg(tmp, y);
}


__global__ void zip_arrays(double *__restrict__ a, double *__restrict__ b, size_t len, WTYPE_cuda *out) {
  // convert two arrays into array of tuples (i.e. complex numbers)
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < len; i+=stride) {
    out[i] = make_cuDoubleComplex(a[i], b[i]);
  }
}

template<typename Iterator, typename T, typename BinaryOperation, typename Pointer>
__global__ void reduce_kernel(Iterator first, Iterator last, T init, BinaryOperation binary_op, Pointer result)
{
  // from https://github.com/thrust/thrust/blob/master/examples/cuda/async_reduce.cu
  *result = thrust::reduce(thrust::cuda::par, first, last, init, binary_op);
}


__global__ void kernel1(WTYPE_cuda *x, STYPE *u, WTYPE_cuda  *y, STYPE *v)
{
  // Single kernel, used in y_i = \sum_j superposition_single(y_i,x_j)
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  WTYPE_cuda sum = ZERO;

  for(int n = 0; n < N; ++n)
    sum = cuCadd(superposition_single(n, i, x, u, v, 1), sum);

  y[i] = sum;
}

#endif
