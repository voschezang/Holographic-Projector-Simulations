#ifndef KERNEL
#define KERNEL

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>

#include "macros.h"

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

// volatile WTYPE& operator=(volatile WTYPE&) volatile;

template <unsigned int blockSize, typename T>
inline __device__ void warp_reduce_c(T *s, const unsigned int i) {
  // TODO assert size <= 2*WARP_SIZE
  // TODO if (1 < size <= 64) for (n = size / 2;;)
#pragma unroll
  for (int n = 32; n >= 1; n/=2) {
    if (blockSize >= n+n)
      s[i] = cuCadd(s[i], s[i + n]);

    __threadfence(); // TODO can this be moved inside the prev if
    // TODO check if fence is required for shared memory and not just for global memory
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

template<typename T>
std::vector<T*> pinnedMallocVector(T **d_ptr, size_t dim1, size_t dim2) {
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto vec = std::vector<T*>(dim1);
  // for (auto&& row : matrix)
  for (size_t i = 0; i < dim1; ++i)
    vec[i] = *d_ptr + i * dim2;
  return vec;
}

template<typename T>
std::vector<DeviceVector<T>> pinnedMallocMatrix(T **d_ptr, size_t dim1, size_t dim2) {
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto matrix = std::vector<DeviceVector<T>>(dim1);
  // std::vector<T>(*d_ptr + a, *d_ptr + b); has weird side effects
  // note that *ptr+i == &ptr[i], but that ptr[i] cannot be read
  for (size_t i = 0; i < dim1; ++i)
    matrix[i] = DeviceVector<T>{.data = *d_ptr + i * dim2, .size = dim2};
  return matrix;
}


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace kernel {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

__global__ void zip_arrays(double *__restrict__ a, double *__restrict__ b, size_t len, WTYPE *out) {
  // convert two arrays into array of tuples (i.e. complex numbers)
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < len; i+=stride) {
    out[i] = make_cuDoubleComplex(a[i], b[i]);
    // out[i] = {a[i], b[i]}; // TODO
  }
}

template<typename Iterator, typename T, typename BinaryOperation, typename Pointer>
__global__ void reduce(Iterator first, Iterator last, T init, BinaryOperation binary_op, Pointer result)
{
  // from https://github.com/thrust/thrust/blob/master/examples/cuda/async_reduce.cu
  *result = thrust::reduce(thrust::cuda::par, first, last, init, binary_op);
}

  ///////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
