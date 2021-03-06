#ifndef KERNEL
#define KERNEL

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "macros.h"

#define cu(result) cudaCheck((result), __FILE__, __LINE__)
#define cuB(result) cudaBlasCheck((result), __FILE__, __LINE__)
#define cuR(result) cudaRandCheck((result), __FILE__, __LINE__)

inline
cudaError_t cudaCheck(cudaError_t result, const char *file, int line)
{
  // Note that max total blockSize is 1024
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
curandStatus_t cudaRandCheck(curandStatus_t result, const char *file, int line) {
  if (result != CURAND_STATUS_SUCCESS) {
    exit(result);
  }
  return result;
}

inline
cublasStatus_t cudaBlasCheck(cublasStatus_t result, const char *file, int line) {
  // check for cuda errors
  // #ifdef DEBUG
  if (result != CUBLAS_STATUS_SUCCESS) {
    auto s = std::string();
    switch (result) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        s = "CUBLAS_STATUS_NOT_INITIALIZED";

      case CUBLAS_STATUS_ALLOC_FAILED:
        s = "CUBLAS_STATUS_ALLOC_FAILED";

      case CUBLAS_STATUS_INVALID_VALUE:
        s = "CUBLAS_STATUS_INVALID_VALUE";

      case CUBLAS_STATUS_ARCH_MISMATCH:
        s = "CUBLAS_STATUS_ARCH_MISMATCH";

      case CUBLAS_STATUS_MAPPING_ERROR:
        s = "CUBLAS_STATUS_MAPPING_ERROR";

      case CUBLAS_STATUS_EXECUTION_FAILED:
        s = "CUBLAS_STATUS_EXECUTION_FAILED";

      case CUBLAS_STATUS_INTERNAL_ERROR:
        s = "CUBLAS_STATUS_INTERNAL_ERROR";

      default:
        s = "<unknown>";
      }
    fprintf(stderr, "[%s:%d] cuBLAS Runtime Error: ", file, line);
    std::cout << s << '\n';
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

inline
__host__ __device__ double norm3d_host(double a, double b, double c) {
  // 2-norm for 3 dimensions
  // simplified and host & device-compatible version of norm3d from CUDA math,  without overflow protection
  return pow(a * a + b * b + c * c, 0.5);
}

inline
__device__ size_t randint(curandState *state, size_t max_range) {
  // excl. max_range
  // note, curand_uniform returns a value in (0,1], but we need the inverse [1,0), which is multiplied by the range and casted to the "previous" int
  return (size_t) max_range * (1. - curand_uniform(state));
}


// template <unsigned int blockSize, typename T>
// inline __device__ void warp_reduce(volatile T *s, unsigned int i) {
//   // example code from Nvidia
//   if (blockSize >= 64) s[i] += s[i + 32];
//   if (blockSize >= 32) s[i] += s[i + 16];
//   if (blockSize >= 16) s[i] += s[i +  8];
//   if (blockSize >=  8) s[i] += s[i +  4];
//   if (blockSize >=  4) s[i] += s[i +  2];
//   if (blockSize >=  2) s[i] += s[i +  1]; // TODO rm last line
// }

// template <unsigned int blockSize>
// inline __device__ void warp_reduce_complex(WAVE *s, const unsigned int i) {
//   // TODO assert size <= 2*WARP_SIZE
//   // TODO if (1 < size <= 64) for (n = size / 2;;)
// #pragma unroll
//   for (int n = 32; n >= 1; n/=2) {
//     if (blockSize >= n+n)
//       s[i] = cuCadd(s[i], s[i + n]);

//     __threadfence(); // TODO mv inside the prev if
//   }

//   // // example code from Nvidia
//   // if (size >= 64) s[i] = cuCadd(s[i], s[i + 32]);
//   // __threadfence();
//   // if (size >= 32) s[i] = cuCadd(s[i], s[i + 16]);
//   // __threadfence();
//   // if (size >= 16) s[i] = cuCadd(s[i], s[i +  8]);
//   // __threadfence();
//   // if (size >=  8) s[i] = cuCadd(s[i], s[i +  4]);
//   // __threadfence();
//   // if (size >=  4) s[i] = cuCadd(s[i], s[i +  2]);
//   // __threadfence();
//   // if (size >=  2) s[i] = cuCadd(s[i], s[i +  1]);
//   // __threadfence();
// }


inline __host__ __device__ void cos_sin(const double x, double *cos, double *sin) {
  // Save cosine(x), sine(x) to &cos, &sin.
  // Flipped arguments for readability.
  sincos(x, sin, cos);
}

inline __host__ __device__ double angle(const cuDoubleComplex phasor) {
  return atan2(phasor.y, phasor.x);
}

inline __host__ __device__ cuDoubleComplex from_polar(const double r, const double phi = 0.) {
  // Convert polar coordinates (r,phi) to Cartesian coordinates (re, im)
  // Using `r * e^(phi I) = r (cos phi + I sin phi)`
  // TODO rename => to_phasor?
  // Note that result = {amp,0} if phase = 0, but adding such a branch may slow down performance
  cuDoubleComplex result;
  cos_sin(phi, &result.x, &result.y);
  return {r * result.x, r * result.y};
}

inline __host__ __device__ cuDoubleComplex to_polar(const cuDoubleComplex x) {
  return {cuCabs(x), angle(x)};
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace kernel {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<bool transpose = false>
inline void sum_rows(const size_t width, const size_t n_rows, cublasHandle_t handle,
                     const WAVE *A, const ConstCUDAVector<WAVE> x,
                     WAVE *y, const WAVE beta = {0., 0.}) {
  /**
   * Sum all rows of matrix A. `A,x,y` must be device pointers.
   *
   * GEMV: GEneral Matrix Vector multiplication
   * `y = alpha * op(A)x + beta y`
   * Note, argument width = lda = stride of matrix
   */
  assert(width == x.size);
  const WAVE alpha = {1., 0.};
  if (transpose)
    cuB( cublasZgemv(handle, CUBLAS_OP_T, width, n_rows, &alpha, A, width, x.data, 1, &beta, y, 1) );
  else
    cuB( cublasZgemv(handle, CUBLAS_OP_N, n_rows, width, &alpha, A, n_rows, x.data, 1, &beta, y, 1) );
}

__global__ void init_rng(curandState *state, const unsigned int seed, const unsigned int i_stream) {
  // seed should be unique for each experiment and may have to be reset after each kernel launch
  const dim3
    tid (blockIdx.x * blockDim.x + threadIdx.x,
         blockIdx.y * blockDim.y + threadIdx.y),
    gridSize (blockDim.x * gridDim.x,
              blockDim.y * gridDim.y);
  const unsigned int
    global_tid = tid.x + tid.y * gridSize.x,
    sequence_number = global_tid + i_stream * gridSize.x * gridSize.y;
  curand_init(seed, sequence_number, 0, &state[sequence_number]);

  // test TODO rm
  auto x = curand_uniform(&state[sequence_number]);
  auto y = curand_uniform_double(&state[sequence_number]);
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
