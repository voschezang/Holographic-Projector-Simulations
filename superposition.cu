#ifndef KERNEL_SUPERPOSITION
#define KERNEL_SUPERPOSITION

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>

#include "macros.h"
#include "util.h"
#include "kernel.cu"

// TODO consider unguarded functions, intrinsic functions
// TODO use softeningSquared?


// a transformation from projector to projection is forwards, vice versa is backwards
enum class Direction {Forward, Backward};

template<Direction dir>
inline __device__ double value() {
  // manual conversion because <type_traits> lib is not yet supported
  if (dir == Direction::Forward) return double{1.0};
  else return double{-1.0};
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace superposition {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<const Direction direction>
inline __device__ WTYPE single(const size_t i, const size_t j,
                        const WTYPE *x, const STYPE *u, STYPE *v) {
  const size_t
    n = i * DIMS,
    m = j * DIMS;
  const double
    distance = norm3d(v[m] - u[n], v[m+1] - u[n+1], v[m+2] - u[n+2]),
    amp = cuCabs(x[i]),
    phase = angle(x[i]);

#ifdef DEBUG
  if (distance == 0) { printf("ERROR: distance must be nonzero\n"); asm("trap;"); }
#endif
  // TODO __ddiv_rd, __dmul_ru

  if (direction == Direction::Forward)
    return polar(amp / distance, phase - distance * TWO_PI_OVER_LAMBDA);
  else
    return polar(amp / distance, phase + distance * TWO_PI_OVER_LAMBDA);
}

template<Direction direction>
inline __device__ void per_thread(WTYPE *__restrict__ x, STYPE *__restrict__ u,
                                  WTYPE *__restrict__ y_local, STYPE *__restrict__ v) {
  // type WTYPE __restrict__ y_local[SHARED_MEMORY_SIZE]
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

#pragma unroll
  for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
    y_local[m].x = 0;
    y_local[m].y = 0;
  }

  // add single far away light source, with arbitrary (but constant) phase
  // assume threadIdx.x is a runtime constant
  if (BLOCKDIM >= KERNEL_BATCH_SIZE)
    if (idx < KERNEL_BATCH_SIZE)
      y_local[idx] = polar(1, 0.4912);

  // for each y-datapoint in current batch
  // outer loop for batch, inner loop for index is faster than vice versa
  for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
    for (size_t i = idx; i < N; i += stride)
      y_local[m] = cuCadd(y_local[m], single<direction>(i, m, x, u, v_cached));

#ifdef DEBUG
    cuCheck(y_local[m]);
#endif
  }
}

inline __device__ void copy_result(WTYPE *__restrict__ local, WTYPE *__restrict__ tmp) {
  // TODO rename tmp => shared
  const unsigned int tid = threadIdx.x;

#if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1)
  for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
    tmp[threadIdx.x + m * BLOCKDIM] = local[m];
    // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = local[m];
#else
  // write before first sync. note that integer division is used

  if (KERNEL_BATCH_SIZE < 2) {
    // use 1/2 a much shared memory

    const unsigned int halfBlockSize = BLOCKDIM / 2;
    // simple reduction, first half writes first, then second half add their result
    if (threadIdx.x >= halfBlockSize)
      for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
        tmp[threadIdx.x - halfBlockSize + m * BLOCKDIM] = local[m];
        // tmp[m + (threadIdx.x - halfBlockSize ) * KERNEL_BATCH_SIZE] = local[m];

    __syncthreads();
    if (threadIdx.x < halfBlockSize)
      for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
        tmp[threadIdx.x + m * KERNEL_BATCH_SIZE] = \
          cuCadd(local[m], tmp[threadIdx.x + m * KERNEL_BATCH_SIZE]);
        // tmp[m + threadIdx.x * KERNEL_BATCH_SIZE] = \
        //   cuCadd(local[m], tmp[m + threadIdx.x * KERNEL_BATCH_SIZE]);
  }
  else if (REDUCE_SHARED_MEMORY == 2) {
    // still use 1/2 of shared memory, but without idle threads
    // TODO consider another step, resulting in 1/4
    // dm  = threadIdx.x < halfBlockSize ? 0 : BLOCKSIZE * KERNEL_BATCH_SIZE / 2,

    const unsigned int size = BLOCKDIM / REDUCE_SHARED_MEMORY;
    // const unsigned int halfBlockSize = BLOCKDIM / 2;
    const unsigned int idx = tid < size ? tid : tid - size;
    {
      const unsigned int
        dm  = threadIdx.x < size ? 0 : KERNEL_BATCH_SIZE / 2;
      for (unsigned int m = 0; m < KERNEL_BATCH_SIZE / 2; ++m) {
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
  }
  else if (REDUCE_SHARED_MEMORY > 2) {
    const unsigned int size = BLOCKDIM / REDUCE_SHARED_MEMORY;
    for (unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
      if (tid < size)
        tmp[m * size + tid] = local[m];

      const unsigned int relative_tid = tid % size;
      for (unsigned int n = 1; n < REDUCE_SHARED_MEMORY; ++n) {
        if (n * size < tid && tid < (n+1) * size) {
          const unsigned int i = m * size + relative_tid;
          // const unsigned int i = (m + dm) + idx * KERNEL_BATCH_SIZE;
          tmp[i] = cuCadd(tmp[i], local[m]);
        }
      }
    }
  }

#endif
  __syncthreads();
}

inline __device__ void aggregate_blocks(WTYPE *__restrict__ tmp, double *__restrict__ y) {
  const unsigned int tid = threadIdx.x;
  const unsigned int size = BLOCKDIM / REDUCE_SHARED_MEMORY;

  // inter warp
  // TODO do this in parallel for the next warp in case of next batch
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
    // TOOD check for memory bank conflicts
    WTYPE *x = &tmp[m * size];
#pragma unroll
    for (unsigned int s = size / 2; s >= 2 * WARP_SIZE; s/=2) {
      if(tid < s)
        x[tid] = cuCadd(x[tid], x[tid + s]);

      __syncthreads();
    }

    // if(size == 512){ if(tid < 256){ x[tid] = cuCadd(x[tid], x[tid + 256]);} __syncthreads();}
    // if(size >= 256){ if(tid < 128){ x[tid] = cuCadd(x[tid], x[tid + 128]);} __syncthreads();}
    // if(size >= 128){ if(tid <  64){ x[tid] = cuCadd(x[tid], x[tid +  64]);} __syncthreads();}
  }


  // final intra warp aggregation
#ifdef PARALLEL_INTRA_WARP_AGG
  // let each warp aggregate a different batch
  const unsigned int n_warps = DIV(BLOCKDIM, WARP_SIZE);
  const unsigned int wid = tid / WARP_SIZE;
  const unsigned int lane = tid % 32;
  assert(n_warps != 0);
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; m+=n_warps)
    for(unsigned int w = 0; w < n_warps; ++w)
      if (wid == w
          && (m+w) < KERNEL_BATCH_SIZE
          && lane < size / 2)
        warp_reduce_c<size, WTYPE>(&tmp[(m+w) * size], lane);
#else
  for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m)
    if (tid < WARP_SIZE && tid < size / 2)
      warp_reduce_c<size, WTYPE>(&tmp[m * size], tid);
#endif


#if (BLOCKDIM >= KERNEL_BATCH_SIZE)
  if (tid < KERNEL_BATCH_SIZE) {
    unsigned int m = tid;
#else
    // for each y-datapoint in current batch
    for(unsigned int m = 0; m < KERNEL_BATCH_SIZE; ++m) {
#endif

    WTYPE sum;
    // sum = ZERO;
    sum = tmp[m * size];
#ifdef DEBUG
    cuCheck(sum);
#endif
    const unsigned int i = blockIdx.x + m * GRIDDIM;
    y[i] = sum.x;
    y[i + GRIDDIM * BATCH_SIZE] = sum.y; // note the use of stream batch size
  }

  // do not sync blocks, exit kernel and agg block results locally or in diff kernel
}

  // TODO template <direction>? icm kernel <<< >>> syntax?
template<Direction direction>
__global__ void per_block(WTYPE *__restrict__ x, STYPE *__restrict__ u,
                          double *__restrict__ y, STYPE *__restrict__ v) {
  __shared__ WTYPE tmp[SHARED_MEMORY_SIZE];
  // TODO transpose tmp array? - memory bank conflicts
  {
    WTYPE y_local[KERNEL_BATCH_SIZE];
    superposition::per_thread<direction>(x, u, y_local, v);
    superposition::copy_result(y_local, tmp);
  }
  superposition::aggregate_blocks(tmp, y);
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
