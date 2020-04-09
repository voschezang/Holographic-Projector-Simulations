#ifndef KERNEL
#define KERNEL

#include <assert.h>
#include <stdio.h>
#include <time.h>
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
inline __device__ void warp_reduce_c(T *s, const unsigned int i) {
#pragma unroll
  for (unsigned int n = 32; n >= 1; n/=2) {
    if (size >= n+n)
      s[i] = cuCadd(s[i], s[i + n]);

    __threadfence();
  }

  // example code from Nvidia
  // if (size >= 64) s[i] = cuCadd(s[i], s[i + 32]);
  // __threadfence();
  // if (size >= 32) s[i] = cuCadd(s[i], s[i + 16]);
  // __threadfence();
  // if (size >= 16) s[i] = cuCadd(s[i], s[i +  8]);
  // __threadfence();
  // if (size >=  8) s[i] = cuCadd(s[i], s[i +  4]);
  // __threadfence();
  // if (size >=  4) s[i] = cuCadd(s[i], s[i +  2]);
  // __threadfence();
  // if (size >=  2) s[i] = cuCadd(s[i], s[i +  1]);
  // __threadfence();
}

__global__ void kernel_zero(WTYPE_cuda *x, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < n; i += stride)
    x[i] = ZERO;
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

#endif
