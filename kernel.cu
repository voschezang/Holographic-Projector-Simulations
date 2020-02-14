#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <complex.h>
#include <cuComplex.h>

#include "macros.h"


__device__ double angle(cuDoubleComplex  z) {
  return atan2(cuCreal(z), cuCimag(z));
}

__device__ cuDoubleComplex polar(double r, double i) {
  // return the complex number r * exp(i * I)
  cuDoubleComplex res;
  sincos(i, &res.x, &res.y);
  return cuCmul(make_cuDoubleComplex(r, 0), res);
}

// TODO consider non-complex types (double real, double imag)
// and check computational cost
__device__ WTYPE_cuda K(size_t i, size_t j, WTYPE_cuda *x, STYPE *u, STYPE *v) {
  // TODO unpack input to u1,u2,3 v1,v2,v3?
  // TODO consider unguarded functions
  size_t n = i * DIMS; // TODO use struct?
  size_t m = j * DIMS;
  double amp, phase, distance;
  int direction = 1;
  // DIM == 3
  distance = norm3d(v[m] - u[n], v[m+1] - u[n+1], v[m+2] - u[n+2]);
  amp = cuCabs(x[i]);
  phase = angle(x[i]);
  amp /= distance;
  phase -= direction * 2 * M_PI * distance / LAMBDA;
  return polar(amp, phase);
}

// TODO optimize memory / prevent Shared memory bank conflicts for x,u arrays
__global__ void kernel3(WTYPE_cuda *x, STYPE *u, WTYPE_cuda *out, STYPE *v, const size_t i_batch)
{
  /** First compute local sum, then do nested aggregation
   *
   * out[BATCH * blockDim] = array with output per block
   * v[BATCH_SIZE * DIM] = location of y
   */
  //
  __shared__ WTYPE_cuda tmp[THREADS_PER_BLOCK];
  WTYPE_cuda sum;
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // const size_t di_batch = i_batch + BATCH_SIZE;
  // for(int i = idx*THREADS_PER_BLOCK; n < (idx+1)*THREADS_PER_BLOCK; ++n) {

  // for each y-datapoint in current batch
  // TODO use cuda.y-stride?
  // TODO switch y-loop and x-loop and let sum : [BATCH_SIZE]? assuming y-batch is in local memory
  {
    size_t j;
    size_t stride = blockDim.x * gridDim.x;
    for(int m = 0; m < BATCH_SIZE; ++m) {
      sum = ZERO;
      j = m + i_batch * BATCH_SIZE; // y
      // Usage of stride allows <<<1,1>>> kernel invocation
      for (int i = idx; i < N; i += stride) {
        sum = cuCadd(K(i, j, x, u, v), sum);
      }
      tmp[m + threadIdx.x + BATCH_SIZE] = sum;
    }
  }

  // sync all (incl non-aggregating cores)
  __syncthreads();

  // aggregate locally (within blocks)
  // TODO multiple stages? if x % 2 == 0 -> ..
  if (threadIdx.x == 0) {
    sum = tmp[0];
    for (int k = 1; k < THREADS_PER_BLOCK; ++k)
      sum = cuCadd(sum, tmp[k]);

    // TODO foreach batch element
    out[blockIdx.x * BATCH_SIZE] = tmp[0];
  }

}

__global__ void kernel1(WTYPE_cuda *x, STYPE *u, WTYPE_cuda  *y, STYPE *v)
{
  // Single kernel, used in y_i = \sum_j K(y_i,x_j)
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  WTYPE_cuda sum = ZERO;

  for(int n = 0; n < N; ++n)
    sum = cuCadd(K(n, i, x, u, v), sum);

  y[i] = sum;
}
