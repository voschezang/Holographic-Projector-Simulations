#ifndef KERNEL
#define KERNEL

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>

#include "macros.h"
#include "hyper_params.h"

#define cu(result) { cudaCheck((result), __FILE__, __LINE__); }


/** GPU version of std::vector
 */
template<typename T>
struct DeviceVector {
  T *data;
  size_t size;
};

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

inline
__host__ __device__ double norm3d_host(double a, double b, double c) {
  // simplified and host & device-compatible version of norm3d from CUDA math,  without overflow protection
  return pow(a * a + b * b + c * c, 0.5);
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

// volatile WAVE& operator=(volatile WAVE&) volatile;

template <unsigned int blockSize>
inline __device__ void warp_reduce_complex(WAVE *s, const unsigned int i) {
  // TODO assert size <= 2*WARP_SIZE
  // TODO if (1 < size <= 64) for (n = size / 2;;)
#pragma unroll
  for (int n = 32; n >= 1; n/=2) {
    if (blockSize >= n+n)
      s[i] = cuCadd(s[i], s[i + n]);

    __threadfence(); // TODO mv inside the prev if
  }

  // // example code from Nvidia
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

inline __host__ __device__ void cos_sin(double x, double *cos, double *sin) {
  // Save cosine(x), sine(x) to &cos, &sin.
  // Flipped arguments for readability.
  sincos(x, sin, cos);
}

inline __host__ __device__ double angle(cuDoubleComplex phasor) {
  return atan2(phasor.y, phasor.x);
}

inline __host__ __device__ cuDoubleComplex from_polar(double r, double phi = 0.) {
  // Convert polar coordinates (r,phi) to Cartesian coordinates (re, im)
  // Using `r * e^(phi I) = r (cos phi + I sin phi)`
  // TODO rename => to_phasor?
  // Note that result = {amp,0} if phase = 0, but adding such a branch may slow down performance
  cuDoubleComplex result;
  cos_sin(phi, &result.x, &result.y);
  return {r * result.x, r * result.y};
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace kernel {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

__global__ void zip_arrays(double *re, double *im, size_t len, WAVE *out) {
  // __global__ void zip_arrays(double *__restrict__ re, double *__restrict__ im, size_t len, cuDoubleComplex *out) {
  // __global__ void zip_arrays(double *__restrict__ re, double *__restrict__ im, size_t len, cuDoubleComplex *__restrict__ out) {
  // out can technically point to the array a
  // convert two arrays into array of tuples (i.e. complex numbers)
  // i.e. transpose & flatten the matrix (a,b)
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < len; i+=stride)
    out[i] = {re[i], im[i]};
}

template<typename Iterator, typename T, typename BinaryOperation, typename Pointer>
__global__ void reduce(Iterator first, Iterator last, T init, BinaryOperation binary_op, Pointer result)
{
  // from https://github.com/thrust/thrust/blob/master/examples/cuda/async_reduce.cu
  *result = thrust::reduce(thrust::cuda::par, first, last, init, binary_op);
}

template<typename Iterator, typename T, typename BinaryOperation, typename Pointer>
__global__ void reduce_rows(Iterator first, const size_t width, const size_t n_rows, T init, BinaryOperation binary_op, Pointer results)
{
  // TODO consider thrust::reduce_by_key
  for (unsigned int i = 0; i < n_rows; ++i) {
    const size_t di = i * width;
    // from https://github.com/thrust/thrust/blob/master/examples/cuda/async_reduce.cu
    results[i] = thrust::reduce(thrust::cuda::par, first + di, first + di + width, init, binary_op);
  }
}

  ///////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
