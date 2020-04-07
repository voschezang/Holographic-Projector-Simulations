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

__device__ void cuCheck(cuDoubleComplex  z) {
  double a = cuCreal(z), b = cuCimag(z);
  if (isnan(a)) printf("cu found nan re\n");
  if (isinf(a)) printf("cu found inf re\n");
  if (isnan(b)) printf("cu found nan I\n");
  if (isinf(b)) printf("cu found inf I\n");
}

inline
__host__ __device__ double angle(cuDoubleComplex  z) {
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

// TODO consider non-complex types (double real, double imag)
// and check computational cost
inline __device__ WTYPE_cuda superposition_single(const size_t i, const size_t j,
                        const WTYPE_cuda *x, const STYPE *u, STYPE *v,
                        const char direction) {
  // TODO unpack input to u1,u2,3 v1,v2,v3?
  // TODO consider unguarded functions, intrinsic functions
#ifdef DEBUG
  assert(direction == -1 || direction == 1);
#endif

  const size_t
    n = i * DIMS,
    m = j * DIMS; // TODO use struct?
  // TODO use softeningSquared?
  // TODO check coalesing
  const double
    distance = norm3d(v[m] - u[n], v[m+1] - u[n+1], v[m+2] - u[n+2]),
    amp = cuCabs(x[i]),
    phase = angle(x[i]);

#ifdef DEBUG
  if (distance == 0) { printf("ERROR: distance must be nonzero\n"); asm("trap;"); }
  // if (amp > 0) printf(">0 \ti: %i, abs: %0.4f, dis: %0.3f\n", i, amp, distance);
  // // TODO check overflows
  if (isnan(amp)) printf("found nan\n");
  if (isinf(amp)) printf("found inf\n");
  if (isnan(distance)) printf("found nan\n");
  if (isinf(distance)) printf("found inf\n");
  // if (amp > 0) printf("amp = %0.5f > 0\n", amp);
  // if (distance > 0) printf("dis: %0.4f\n\n", distance);
  const cuDoubleComplex res = polar(amp, phase);
  if (amp > 0) assert(cuCabs(res) > 0);
#endif

  // TODO __ddiv_rd, __dmul_ru
  return polar(amp / distance, phase - distance * direction * TWO_PI_OVER_LAMBDA);
}

inline __device__ void superposition_partial(WTYPE_cuda *x, STYPE *u, WTYPE_cuda *y_local, STYPE *v, const char direction) {
  // inner scope
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    // size_t j;
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
        sum = cuCadd(superposition_single(i, m, x, u, v, direction), sum);

      // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = sum;
      y_local[m] = sum;
#ifdef DEBUG
      cuCheck(sum);
#endif
    }
  }
}

inline __device__ void superposition_cp_result(WTYPE_cuda *local, WTYPE_cuda *tmp) {
#ifndef PARTIAL_AGG
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
    tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = local[m];
#else
  if (BLOCKDIM == 1) {
    // note that gridDim can still be nonzero
    // TODO
    assert(0);
    return; // no divergence because this cannot be true for other threads
  }

  // use 1/2 a much shared memory

  const unsigned int halfBlockSize = BLOCKDIM / 2;
  // write before first sync. note that integer division is used
  if (threadIdx.x >= halfBlockSize)
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
      tmp[m + (threadIdx.x - halfBlockSize ) * KERNEL_BATCH_SIZE] = local[m];

  __syncthreads();
  if (threadIdx.x < BLOCKDIM / 2)
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
      tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = \
        cuCadd(local[m], tmp[m + threadIdx.x * KERNEL_BATCH_SIZE]);
#endif

  __syncthreads();
}


// TODO optimize memory / prevent Shared memory bank conflicts for x,u arrays
// TODO use __restrict__, const
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
#if (defined(PARTIAL_AGG) && KERNEL_BATCH_SIZE > 1)
  __shared__ WTYPE_cuda tmp[BLOCKDIM * KERNEL_BATCH_SIZE / 2];
#else
  __shared__ WTYPE_cuda tmp[BLOCKDIM * KERNEL_BATCH_SIZE];
#endif

  {
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

    // TODO use cuda.y-stride? - note the double for loop - how much memory fits in an SM?
    // TODO switch y-loop and x-loop and let sum : [BATCH_SIZE]? assuming y-batch is in local memory
    // printf("idx %i -", threadIdx.x);

    {
      WTYPE_cuda local[KERNEL_BATCH_SIZE];
      superposition_partial(x, u, local, v_cached, direction);
      superposition_cp_result(local, tmp);
    }
  // free v_cached // TODO check if this does anything
  }

  const unsigned int quarterBlockSize = BLOCKDIM / 2;

  // TODO let first quarter do y1, let second quarter do y2..?
  // if (threadIdx.x < quarterBlockSize) {
  //   // TOOD spread out mem access to reduce memory bank conflicts
  //   for(unsigned int m = 0; m < BATCH_SIZE; ++m) {
  //     const unsigned int i = m + threadIdx.x * BATCH_SIZE;
  //     tmp[i] = \
  //       cuCadd(tmp[i], tmp[i + quarterBlockSize]);
  //   }
  // }

  // aggregate locally (within blocks)
  __syncthreads();
  if (threadIdx.x == 0) {
    // for each y-datapoint in current batch
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
      WTYPE_cuda sum;
      sum = ZERO;

#if (defined(PARTIAL_AGG) && BLOCKDIM > 1)
      for (unsigned int k = 0; k < BLOCKDIM / 2; ++k)
#else
      for (unsigned int k = 0; k < BLOCKDIM; ++k)
#endif
        sum = cuCadd(sum, tmp[m + k * KERNEL_BATCH_SIZE]);
      // for (unsigned int k = 0; k < BLOCKDIM; ++k)
      //   sum = cuCadd(sum, tmp[k + m * BLOCKDIM]);

#ifdef DEBUG
      cuCheck(sum);
#endif

      // TODO foreach batch element
      // y[blockIdx.x + m * GRIDDIM] = sum;
      const unsigned int i = blockIdx.x + m * GRIDDIM;
      y[i] = sum.x;
      y[i + GRIDDIM * KERNEL_BATCH_SIZE] = sum.y;
      // y[m + blockIdx.x * BATCH_SIZE] = sum;
    }
  }

  // do not sync blocks, exit kernel and agg block results locally or in diff kernel
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
