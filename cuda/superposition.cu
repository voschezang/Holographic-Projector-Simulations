#ifndef KERNEL_SUPERPOSITION
#define KERNEL_SUPERPOSITION

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>

#include "macros.h"
#include "hyper_params.h"
#include "util.h"
#include "kernel.cu"

// TODO consider unguarded functions, intrinsic functions
// TODO use softeningSquared?


// a transformation from projector to projection is forwards, vice versa is backwards
enum class Direction {Forward, Backward};

// template<Direction dir>
// inline __device__ double value() {
//   // manual conversion because <type_traits> lib is not yet supported
//   if (dir == Direction::Forward) return double{1.0};
//   else return double{-1.0};
// }

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace superposition {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<const Direction direction>
inline __device__ WTYPE single(const size_t i, const size_t j,
                        const WTYPE *x, const STYPE *u, const STYPE *v) {
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

template<Direction direction, bool add_constant_source>
inline __device__ void per_thread(const WTYPE *__restrict__ x, const size_t N_x,
                                  const STYPE *__restrict__ u,
                                  WTYPE *__restrict__ y_local,
                                  const STYPE *__restrict__ v) {
#if CACHE_BATCH
  // TODO don't fill array in case N_x < tid
  // cache v[batch] because it is read by every thread
  // v_cached is constant and equal for each block
  __shared__ STYPE v_cached[KERNEL_SIZE * DIMS];
  // use strides when BLOCKDIM < BATCH_SIZE * DIMS
  for (unsigned int i = threadIdx.x; i < KERNEL_SIZE * DIMS; i+=blockDim.x)
    v_cached[i] = v[i];

  __syncthreads();
  STYPE *v_ = v_cached;
#else
  STYPE *v_ = v;
#endif

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  // add single far away light source, with arbitrary (but constant) phase
  // assume threadIdx.x is a runtime constant
  if (add_constant_source)
    if (blockDim.x >= KERNEL_SIZE && idx < KERNEL_SIZE)
      y_local[idx] = polar(1, ARBITRARY_PHASE);

  // for each y-datapoint in current batch
  // outer loop for batch, inner loop for index is faster than vice versa
  // TODO consider transposing u, v to improve memory coalescing (w[i, j] is always read for each thread i, then for each dim j)
  for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
    for (size_t i = idx; i < N_x; i += stride)
      y_local[m] = cuCadd(y_local[m], single<direction>(i, m, x, u, v_));
}

inline __device__ void copy_result(WTYPE *__restrict__ local, WTYPE *__restrict__ y_shared) {
  const unsigned int tid = threadIdx.x;

  if (blockDim.x == 1 || REDUCE_SHARED_MEMORY == 1) {
    // #if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1)
    for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
      y_shared[threadIdx.x + m * blockDim.x] = local[m];
    // y_shared[m + threadIdx.x * KERNEL_SIZE] = local[m];
    // #else
  }
  else {
    // write before first sync. note that integer division is used

    if (KERNEL_SIZE < 2) {
      // use 1/2 a much shared memory

      const unsigned int halfBlockSize = blockDim.x / 2;
      // simple reduction, first half writes first, then second half add their result
      if (threadIdx.x >= halfBlockSize)
        for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
          y_shared[threadIdx.x - halfBlockSize + m * blockDim.x] = local[m];
      // y_shared[m + (threadIdx.x - halfBlockSize ) * KERNEL_SIZE] = local[m];

      __syncthreads();
      if (threadIdx.x < halfBlockSize)
        for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
          y_shared[threadIdx.x + m * KERNEL_SIZE] = \
            cuCadd(local[m], y_shared[threadIdx.x + m * KERNEL_SIZE]);
      // y_shared[m + threadIdx.x * KERNEL_SIZE] = \
      //   cuCadd(local[m], y_shared[m + threadIdx.x * KERNEL_SIZE]);
    }
    else if (REDUCE_SHARED_MEMORY == 2) {
      // still use 1/2 of shared memory, but without idle threads
      // TODO consider another step, resulting in 1/4
      // dm  = threadIdx.x < halfBlockSize ? 0 : BLOCKSIZE * KERNEL_SIZE / 2,

      const unsigned int size = blockDim.x / REDUCE_SHARED_MEMORY;
      // const unsigned int halfBlockSize = BLOCKDIM / 2;
      const unsigned int idx = tid < size ? tid : tid - size;
      {
        const unsigned int
          dm  = threadIdx.x < size ? 0 : KERNEL_SIZE / 2;
        for (unsigned int m = 0; m < KERNEL_SIZE / 2; ++m) {
          // first half writes first half of batch, idem for second half
          // y_shared[(m + dm) + idx * KERNEL_SIZE] = local[m + dm];
          y_shared[idx + (m + dm) * size] = local[m + dm];
        }
      }

      __syncthreads();
      const unsigned int dm = threadIdx.x < size ? KERNEL_SIZE / 2 : 0;
      for(unsigned int m = 0; m < KERNEL_SIZE / 2; ++m) {
        // first half adds+writes first half of batch, idem for second half
        const unsigned int i = idx + (m + dm) * size;
        // const unsigned int i = (m + dm) + idx * KERNEL_SIZE;
        y_shared[i] = cuCadd(y_shared[i], local[m + dm]);
      }
    }
    else if (REDUCE_SHARED_MEMORY > 2) {
      const unsigned int size = blockDim.x / REDUCE_SHARED_MEMORY;
      for (unsigned int m = 0; m < KERNEL_SIZE; ++m) {
        if (tid < size)
          y_shared[m * size + tid] = local[m];

        const unsigned int relative_tid = tid % size;
        for (unsigned int n = 1; n < REDUCE_SHARED_MEMORY; ++n) {
          if (n * size < tid && tid < (n+1) * size) {
            const unsigned int i = m * size + relative_tid;
            // const unsigned int i = (m + dm) + idx * KERNEL_SIZE;
            y_shared[i] = cuCadd(y_shared[i], local[m]);
          }
        }
      }
    }

  } // end if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1) {
  __syncthreads();
}

template <unsigned int blockSize>
inline __device__ void aggregate_blocks(WTYPE *__restrict__ y_shared, double *__restrict__ y_global) {
  const unsigned int tid = threadIdx.x;
  const unsigned int size = CEIL(blockSize, REDUCE_SHARED_MEMORY);

  // inter warp
  // TODO do this in parallel for the next warp in case of next batch
  for(unsigned int m = 0; m < KERNEL_SIZE; ++m) {
    // TOOD check for memory bank conflicts
    WTYPE *x = &y_shared[m * size];
#pragma unroll
    for (unsigned int s = size / 2; s >= 2 * WARP_SIZE; s/=2) {
      if(tid < s)
        x[tid] = cuCadd(x[tid], x[tid + s]);

      __syncthreads();
    }

    // alt
    // if(size == 512){ if(tid < 256){ x[tid] = cuCadd(x[tid], x[tid + 256]);} __syncthreads();}
    // if(size >= 256){ if(tid < 128){ x[tid] = cuCadd(x[tid], x[tid + 128]);} __syncthreads();}
    // if(size >= 128){ if(tid <  64){ x[tid] = cuCadd(x[tid], x[tid +  64]);} __syncthreads();}
  }

  if (size >= 2) {
    // final intra warp aggregation
    if (PARALLEL_INTRA_WARP_AGG && blockSize >= 2 * WARP_SIZE) {
      // TODO this is incorrect, output has matrix-like noise
      // let each warp aggregate a different batch
      const unsigned int n_warps = CEIL(blockSize, WARP_SIZE);
      const unsigned int wid = tid / WARP_SIZE;
      const unsigned int lane = tid % 32;
      static_assert(size < 2 || size / 2 != 0, "");
      // use 1 + lane < 1 + size / 2 to suppress warning
      for(unsigned int m = 0; m < KERNEL_SIZE; m+=n_warps)
        for(unsigned int w = 0; w < n_warps; ++w)
          if (wid == w
              && (m+w) < KERNEL_SIZE
              && 1 + lane < 1 + size / 2)
            warp_reduce_complex<size>(&y_shared[(m+w) * size], lane);
    }
    else {
      // use 1 + tid < 1 + size / 2 to suppress warning
      for(unsigned int m = 0; m < KERNEL_SIZE; ++m)
        if (tid < WARP_SIZE && 1+ tid < 1 + size / 2)
          warp_reduce_complex<size>(&y_shared[m * size], tid);
    }
  }

  // TODO check case of small Blockdim
  for(unsigned int m = tid; m < KERNEL_SIZE; m+=blockDim.x) {
    // Note that y_global[0] is relative to current batch and kernel
    const auto i = blockIdx.x + m * gridDim.x;
    const auto sum = y_shared[m * size];
    y_global[i] = sum.x;
    y_global[i + gridDim.x * BATCH_SIZE * KERNEL_SIZE] = sum.y; // note the use of stream batch size
  }

  // do not sync blocks, exit kernel and agg block results locally or in different kernel
}

template<Direction direction, bool add_constant_source, unsigned int blockSize>
__global__ void per_block(const Geometry p,
                          const WTYPE *__restrict__ x, const size_t N_x,
                          const STYPE *__restrict__ u,
                          double *__restrict__ y_global,
                          STYPE *__restrict__ v) {
  __shared__ WTYPE y_shared[SHARED_MEMORY_SIZE(blockSize)];
  // TODO transpose y_shared? - memory bank conflicts, but first simplify copy_result()
  // but, this would make warp redcuce more complex
  {
    // extern WTYPE y_local[]; // this yields; warning: address of a host variable "dynamic" cannot be directly taken in a device function
    WTYPE y_local[KERNEL_SIZE] = {}; // init to zero
    superposition::per_thread<direction, add_constant_source>(x, N_x, u, y_local, v);
    superposition::copy_result(y_local, y_shared);
  }
  superposition::aggregate_blocks<blockSize>(y_shared, y_global);
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
