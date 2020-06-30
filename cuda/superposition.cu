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
enum class Direction {Forwards, Backwards};

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
inline __host__ __device__ WAVE single(const WAVE x, const SPACE *u, const SPACE *v) {
  /**
   * Compute the "superposition" of a single input datapoint, for some location `v \in R^3`
   */
  // transposed shape (DIMS, N) for spatial data is not significantly faster than shape (N, DIMS)
  const double
    distance = NORM_3D(v[0] - u[0], v[1] - u[1], v[2] - u[2]),
    amp = cuCabs(x),
    phase = angle(x);

#ifdef DEBUG
  if (distance == 0) { printf("ERROR: distance must be nonzero\n"); asm("trap;"); }
#endif
  // TODO __ddiv_rd, __dmul_ru

#ifdef TEST_CONST_PHASE
  return from_polar(amp / distance, ARBITRARY_PHASE);
#endif

  if (direction == Direction::Forwards)
    return from_polar(amp / distance, phase + distance * TWO_PI_OVER_LAMBDA);
  else
    return from_polar(amp / distance, phase - distance * TWO_PI_OVER_LAMBDA);
}

template<Direction direction, bool cache_batch>
inline __device__ void per_thread(const WAVE *__restrict__ x, const size_t N_x,
                                  const SPACE *__restrict__ u,
                                  WAVE *__restrict__ y_local,
                                  const SPACE *__restrict__ v)
{
  const size_t
    idx = blockIdx.x * blockDim.x + threadIdx.x,
    stride = blockDim.x * gridDim.x;

  if (cache_batch) {
    // TODO don't fill array in case N_x < tid
    // cache v[batch] because it is read by every thread
    // v_cached is constant and equal for each block
    __shared__ SPACE v_cached[KERNEL_SIZE * DIMS];
    // use strides when BLOCKDIM < BATCH_SIZE * DIMS
    for (unsigned int i = threadIdx.x; i < KERNEL_SIZE * DIMS; i+=blockDim.x)
      v_cached[i] = v[i];

    __syncthreads();

    // for each y-datapoint in current batch
    // outer loop for batch, inner loop for index is faster than vice versa
    for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
      for (size_t i = idx; i < N_x; i += stride)
        y_local[m] = cuCadd(y_local[m], single<direction>(x[i], &u[i * DIMS], &v_cached[m * DIMS]));

  } else {
    // for each y-datapoint in current batch
    // outer loop for batch, inner loop for index is faster than vice versa
    for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
      for (size_t i = idx; i < N_x; i += stride)
        y_local[m] = cuCadd(y_local[m], single<direction>(x[i], &u[i * DIMS], &v[m * DIMS]));
  }
}

inline __device__ void copy_result(WAVE *__restrict__ local, WAVE *__restrict__ shared) {
  const unsigned int tid = threadIdx.x;

  if (blockDim.x == 1 || REDUCE_SHARED_MEMORY == 1) {
    // #if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1)
    // TODO #if SHARED_MEMORY_LAYOUT row-major #else column-major
    for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
      shared[SIdx(m, tid, blockDim.x)] = local[m];
  }
  else {
    // write before first sync. note that integer division is used

    if (KERNEL_SIZE < 2) {
      // use 1/2 a much shared memory

      const unsigned int halfBlockSize = blockDim.x / 2;
      // simple reduction, first half writes first, then second half add their result
      if (threadIdx.x >= halfBlockSize)
        for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
          shared[SIdx(m, tid - halfBlockSize, blockDim.x)] = local[m];

      __syncthreads();
      if (threadIdx.x < halfBlockSize)
        for (unsigned int m = 0; m < KERNEL_SIZE; ++m)
          shared[SIdx(m, tid, blockDim.x)] = cuCadd(shared[SIdx(m, tid, blockDim.x)], local[m]);
    }
    else if (REDUCE_SHARED_MEMORY == 2) {
      // still use 1/2 of shared memory, but without idle threads
      // TODO consider another step, resulting in 1/4
      // dm  = threadIdx.x < halfBlockSize ? 0 : BLOCKSIZE * KERNEL_SIZE / 2,

      const unsigned int size = blockDim.x / REDUCE_SHARED_MEMORY;
      const unsigned int idx = tid < size ? tid : tid - size;
      {
        const unsigned int
          dm  = threadIdx.x < size ? 0 : KERNEL_SIZE / 2;
        // first half writes first half of batch, idem for second half
        for (unsigned int m = 0; m < KERNEL_SIZE / 2; ++m)
          shared[SIdx(m+dm, idx, size)] = local[m + dm];
      }

      __syncthreads(); // block level sync
      const unsigned int dm = threadIdx.x < size ? KERNEL_SIZE / 2 : 0;
      // first half adds+writes first half of batch, idem for second half
      for(unsigned int m = 0; m < KERNEL_SIZE / 2; ++m)
        shared[SIdx(m+dm, idx, size)] = cuCadd(shared[SIdx(m+dm, idx, size)], local[m + dm]);
    }
    else if (REDUCE_SHARED_MEMORY > 2) {
      const unsigned int size = blockDim.x / REDUCE_SHARED_MEMORY;
      for (unsigned int m = 0; m < KERNEL_SIZE; ++m) {
        if (tid < size)
          shared[SIdx(m, tid, size)] = local[m];

        const unsigned int relative_tid = tid % size;
        for (unsigned int n = 1; n < REDUCE_SHARED_MEMORY; ++n) {
          __syncthreads(); // block level sync
          if (n * size < tid && tid < (n+1) * size) {
            const unsigned int i = SIdx(m, relative_tid, size);
            shared[i] = cuCadd(shared[i], local[m]);
          }
        }
      }
    }

  } // end if (BLOCKDIM == 1 || REDUCE_SHARED_MEMORY == 1) {
  __syncthreads(); // block level sync
}

template <unsigned int blockSize>
inline __device__ void aggregate_blocks(WAVE *__restrict__ y_shared, double *__restrict__ y_global) {
  const unsigned int tid = threadIdx.x;
  const unsigned int size = CEIL(blockSize, REDUCE_SHARED_MEMORY);

  // agg inter warp
  for(unsigned int m = 0; m < KERNEL_SIZE; ++m) {
    // TOOD check for memory bank conflicts

    #pragma unroll
        for (unsigned int s = size / 2; s >= 2 * WARP_SIZE; s/=2) {
          if(tid < s)
            y_shared[SIdx(m, tid, size)] = cuCadd(y_shared[SIdx(m, tid,     size)],
                                                  y_shared[SIdx(m, tid + s, size)]);

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
#if !SHARED_MEMORY_LAYOUT
      assert(0); // TODO
#endif
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
        if (tid < WARP_SIZE && 1 + tid < 1 + size / 2) {
#pragma unroll
          for (int n = WARP_SIZE; n >= 1; n/=2)
            if (size >= n+n) {
              // s[i] = cuCadd(s[i], s[i + n]);
              y_shared[SIdx(m, tid, size)] = cuCadd(y_shared[SIdx(m, tid,     size)],
                                                    y_shared[SIdx(m, tid + n, size)]);
              __threadfence(); // TODO mv inside the prev if
            }
        }
    }
  }

  // TODO check case of small Blockdim
  for(unsigned int m = tid; m < KERNEL_SIZE; m+=blockDim.x) {
    // Note that y_global[0] is relative to current batch and kernel
    const auto i = blockIdx.x + m * gridDim.x;
    const auto sum = y_shared[SIdx(m, 0, size)];
    y_global[i] = sum.x;
    y_global[i + gridDim.x * BATCH_SIZE * KERNEL_SIZE] = sum.y; // note the use of stream batch size
  }

  // do not sync blocks, exit kernel and agg block results locally or in different kernel
}

template<Direction direction, unsigned int blockSize>
__global__ void per_block(const Geometry p,
                          const WAVE *__restrict__ x, const size_t N_x,
                          const SPACE *__restrict__ u,
                          double *__restrict__ y_global,
                          const SPACE *__restrict__ v) {
  __shared__ WAVE y_shared[SHARED_MEMORY_SIZE(blockSize)];
  // TODO transpose y_shared? - memory bank conflicts, but first simplify copy_result()
  // but, this would make warp redcuce more complex
  {
    // extern WAVE y_local[]; // this yields; warning: address of a host variable "dynamic" cannot be directly taken in a device function
    WAVE y_local[KERNEL_SIZE] = {}; // init to zero
    superposition::per_thread<direction, CACHE_BATCH>(x, N_x, u, y_local, v);

    // printf("tid: %i \t y[0]: a: %f phi: %f (local)\n", threadIdx.x, cuCabs(y_local[0]), angle(y_local[0]));
    superposition::copy_result(y_local, y_shared);
    // {
    //   __syncthreads();
    //   printf("tid: %i \t y[0]: a: %f phi: %f (shared)\n", threadIdx.x, cuCabs(y_shared[0]), angle(y_shared[0]));
    // }
#ifdef TEST_CONST_PHASE
    for (size_t i = 0; i < KERNEL_SIZE; ++i) {
      // printf("y_local[%lu]: amp: %e, phase: %e, .y: %e\n", i, cuCabs(y_local[i]), angle(y_local[i]), y_local[i].y);
      assert(angle(y_local[i]) - ARBITRARY_PHASE < 1e-6);
    }
    __syncthreads();
    for (size_t i = 0; i < SHARED_MEMORY_SIZE(blockSize); ++i)
      assert(angle(y_shared[i]) - ARBITRARY_PHASE < 1e-6);
#endif
  }
  superposition::aggregate_blocks<blockSize>(y_shared, y_global);
  // {
  //   __syncthreads();
  //   size_t m = gridDim.x * BATCH_SIZE * KERNEL_SIZE;
  //   printf("N_x: %lu, m: %lu\n", N_x, m);
  //   WAVE c = {y_global[0], y_global[m]};
  //   printf("tid: %i \t y[0]: a: %f phi: %f (shared)\n", threadIdx.x, cuCabs(c), angle(c));
  // }
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
