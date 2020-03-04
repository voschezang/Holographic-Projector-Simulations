#ifndef KERNEL
#define KERNEL

#include "macros.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <complex.h>
#include <cuComplex.h>


inline
cudaError_t cu(cudaError_t result)
{
  // check for cuda errors
#ifdef DEBUG
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
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
__device__ double angle(cuDoubleComplex  z) {
  return atan2(cuCreal(z), cuCimag(z));
}

inline
__device__ cuDoubleComplex polar(double a, double phi) {
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
inline
__device__ WTYPE_cuda K(size_t i, size_t j,
                        WTYPE_cuda *x, STYPE *u, STYPE *v,
                        const char direction) {
  // TODO unpack input to u1,u2,3 v1,v2,v3?
  // TODO consider unguarded functions, intrinsic functions
#ifdef DEBUG
  assert(direction == -1 || direction == 1);
#endif

  size_t
    n = i * DIMS,
    m = j * DIMS; // TODO use struct?
  // TODO use softeningSquared?
  double
    distance = norm3d(v[m] - u[n], v[m+1] - u[n+1], v[m+2] - u[n+2]),
    amp = cuCabs(x[i]),
    phase = angle(x[i]);

#ifdef DEBUG
  if (distance == 0) { printf("ERROR: distance must be nonzero"); asm("trap;"); }
  // if (amp > 0) printf(">0 \ti: %i, abs: %0.4f, dis: %0.3f\n", i, amp, distance);
  // // TODO check overflows
  if (isnan(amp)) printf("found nan\n");
  if (isinf(amp)) printf("found inf\n");
  if (isnan(distance)) printf("found nan\n");
  if (isinf(distance)) printf("found inf\n");
  // if (amp > 0) printf("amp = %0.5f > 0\n", amp);
  // if (distance > 0) printf("dis: %0.4f\n\n", distance);
  cuDoubleComplex res = polar(amp, phase);
  if (amp > 0) assert(cuCabs(res) > 0);
#endif

  // TODO __ddiv_rd, __dmul_ru
  return polar(amp / distance, phase - distance * direction * TWO_PI_OVER_LAMBDA);
}

// TODO optimize memory / prevent Shared memory bank conflicts for x,u arrays
__global__ void kernel3(WTYPE_cuda *x, STYPE *u, WTYPE_cuda *y, STYPE *v, const size_t i_batch, const char direction)
{
  /** First compute local sum, then do nested aggregation
   *
   * out[BATCH_SIZE * blockDim] = array with output per block
   * v[BATCH_SIZE * DIM] = locations of y-datapoints
   */
  //
  __shared__ WTYPE_cuda tmp[THREADS_PER_BLOCK * BATCH_SIZE];
  // TODO use cuda.y-stride? - note the double for loop - how much memory fits in an SM?
  // TODO switch y-loop and x-loop and let sum : [BATCH_SIZE]? assuming y-batch is in local memory
  // printf("idx %i -", threadIdx.x);
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    // size_t j;
    WTYPE_cuda sum;

    // for each y-datapoint in current batch
    for(unsigned int m = 0; m < BATCH_SIZE; ++m) {
      sum = ZERO;
      // j = m + i_batch * BATCH_SIZE;
      // Usage of stride allows <<<1,1>>> kernel invocation
      for (size_t i = idx; i < N; i += stride) {
        sum = cuCadd(K(i, m, x, u, v, direction), sum);
        // TODO do this in separate func
        //TODO err: i_batch does not depend on x
        if (i == 0 && direction == -1) {
          // add single far away light source, with constant phase
          // TODO this causes a strange offset in z
          sum = cuCadd(polar(1, 0.4912), sum);
        }
      }
      tmp[m + threadIdx.x * BATCH_SIZE] = sum;
      // tmp[m * THREADS_PER_BLOCK + threadIdx.x] = sum;
#ifdef DEBUG
      cuCheck(sum);
#endif
    }
  }

  // sync all (incl non-aggregating cores)
  __syncthreads();

  // aggregate locally (within blocks)
  if (threadIdx.x == 0) {
    // for each y-datapoint in current batch
    for(unsigned int m = 0; m < BATCH_SIZE; ++m) {
      WTYPE_cuda sum;
      sum = ZERO;
      for (unsigned int k = 0; k < THREADS_PER_BLOCK; ++k)
        sum = cuCadd(sum, tmp[m + k * BATCH_SIZE]);
      // for (unsigned int k = 0; k < THREADS_PER_BLOCK; ++k)
      //   sum = cuCadd(sum, tmp[k + m * THREADS_PER_BLOCK]);

#ifdef DEBUG
      cuCheck(sum);
#endif

      // TODO foreach batch element
      // y[blockIdx.x + m * BLOCKDIM] = sum;
      y[m + blockIdx.x * BATCH_SIZE] = sum;
    }
  }

  // do not sync blocks, exit kernel and agg block results locally or in diff kernel
}

__global__ void kernel1(WTYPE_cuda *x, STYPE *u, WTYPE_cuda  *y, STYPE *v)
{
  // Single kernel, used in y_i = \sum_j K(y_i,x_j)
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  WTYPE_cuda sum = ZERO;

  for(int n = 0; n < N; ++n)
    sum = cuCadd(K(n, i, x, u, v, 1), sum);

  y[i] = sum;
}

#endif
