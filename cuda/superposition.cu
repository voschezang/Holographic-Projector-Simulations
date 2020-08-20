#ifndef KERNEL_SUPERPOSITION
#define KERNEL_SUPERPOSITION

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <cuComplex.h>
#include <cub/cub.cuh>

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
inline __host__ __device__ WAVE phasor_displacement(const Polar x, const double *u, const double *v) {
  // inline __host__ __device__ WAVE phasor_displacement(const double a, const double phi, const SPACE *u, const SPACE *v) {
  /**
   * Compute the phasor displacement single source datapoint, for some target location `v \in R^3`
   * `amp / distance * exp(phi \pm distance * 2 * pi / lambda)`
   */
  // const auto distance  = NORM_3D(v[0] - u[0], v[1] - u[1], v[2] - u[2]);
  const double distance = NORM_3D(v[0] - u[0], v[1] - u[1], v[2] - u[2]);
#if DEBUG
  assert(distance > 1e-9);
#endif
  if (direction == Direction::Forwards)
    return from_polar(x.amp / distance, x.phase + distance * TWO_PI_OVER_LAMBDA);
  else
    return from_polar(x.amp / distance, x.phase - distance * TWO_PI_OVER_LAMBDA);
}

template<const Direction direction>
__global__ void phasor_displacement(const Polar x, const double *u, const double *v, WAVE *y) {
  // in place
  y[0] = phasor_displacement<direction>(x, u, v);
}

template<Direction direction, int blockDim_x, int blockDim_y, Algorithm algorithm, bool shared_memory = false>
__global__ void per_block(const size_t N, const size_t M, const size_t N_stride,
                          const Polar *__restrict__ x,
                          const double *__restrict__ u,
                          const double *__restrict__ v,
                          WAVE *__restrict__ y_global,
                          const bool append_result = false) {
  // Ideally N == width but in case of underutilized batches equality does not hold.
  // M_stride is omitted because it is always equal to M
#ifdef DEBUG
  assert(blockDim.x * blockDim.y * blockDim.z <= 1024); // max number of threads per block
#endif
  // const auto
  //   tid = blockIdx * blockDim + threadIdx,
  //   gridSize = blockDim * gridDim;
  const dim3
    tid (blockIdx.x * blockDim.x + threadIdx.x,
         blockIdx.y * blockDim.y + threadIdx.y),
    gridSize (blockDim.x * gridDim.x,
              blockDim.y * gridDim.y);

  if (algorithm == Algorithm::Naive) {
    for (size_t n = tid.x; n < N; n += gridSize.x) {
      // Note that "caching" x[n] per outer loop happens automatically by the compiler
      for (size_t m = tid.y; m < M; m += gridSize.y) {
#ifndef TEST_CONST_PHASE
        const WAVE y = phasor_displacement<direction>(x[n], &u[n * DIMS], &v[m * DIMS]);
#else
        const WAVE y = from_polar(1., 0.);
#endif
        const size_t i = Yidx(n, m, N_stride, M);
        // TODO add bool to template and use: y[] = y + int(append) y
        if (append_result)
          {y_global[i].x += y.x; y_global[i].y += y.y;}
        else
          y_global[i] = y;
        assert(!isinf(cuCabs(y_global[i])));
        assert(!isnan(cuCabs(y_global[i])));
      } }
  }
  else {
    // TODO compare enums cub::BlockReduceAlgorithm: cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY cub::BLOCK_REDUCE_WARP_REDUCTIONS
    typedef cub::BlockReduce<double, blockDim_x, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, 700> BlockReduce;
    // typedef cub::BlockReduce<double, blockDim_x> BlockReduce;
    __shared__ typename BlockReduce::TempStorage y_shared[shared_memory ? blockDim_y : 1];
    // __shared__ typename BlockReduce::TempStorage y_shared[blockDim_y];

#ifdef U_SHARED
    // TODO extern, depending on N / gridSize.x / thread_size.x
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
    // extern __shared__ double shared[];
    const size_t
      thread_size_x = 4,
      shared_len_x = shared_memory ? blockDim_x * thread_size_x : 0;
    __shared__ Polar x_shared[MAX(shared_len_x, 1)];
    __shared__ double u_shared[MAX(shared_len_x * DIMS, 1)];
    // WAVE *x_shared = (WAVE*) shared;
    // double *u_shared = (double*) &shared[thread_size.x];

    // save results to Y[m, x] instead of Y[m, n]

    // NOTE tid != threadIdx
    if (shared_memory) {
      for (size_t i = threadIdx.y; i < thread_size_x; i += blockDim_y) {
        const size_t
          n = tid.x + i * gridSize.x,
          k = threadIdx.x + i * blockDim_x;
        if (n < N) {
          x_shared[k] = x[n];
          for (int j = 0; j < DIMS; ++j)
            u_shared[k * DIMS + j] = u[n * DIMS + j];
        }
      }
      __syncthreads();
    }
#endif
#ifdef V_SHARED
    double *v_shared = (double *) &y_shared[0];
#endif

    for (size_t m = tid.y; m < M; m += gridSize.y) {

#ifdef V_SHARED
      if (shared_memory && blockDim_x >= DIMS && N >= DIMS && M >= gridSize.y) {
        // modulo % operation is very slow
        // if (shared_memory && blockDim_x >= DIMS && N >= DIMS && M >= gridSize.y && M % gridSize.y == 0 && N % gridSize.x == 0) {
        if (threadIdx.x < DIMS)
          v_shared[threadIdx.y * DIMS + threadIdx.x] = v[m * DIMS + threadIdx.x];

        __syncthreads();
      }
#endif

      // TODO mv condition to start of func
      if (tid.x < N) { // TODO this should cause deadlocks during BlockReduce (for certain geometry)
        // TODO add subfunctions for profiler
        WAVE y {0,0};

#ifdef U_SHARED
        for (size_t i = 0; i < thread_size_x; ++i) {
          const size_t
            n = tid.x + i * gridSize.x,
            k = threadIdx.x + i * blockDim_x;
          if (n < N)
            y = cuCadd(y, phasor_displacement<direction>(x_shared[k], &u_shared[k * DIMS], &v[m * DIMS]));
          // y = cuCadd(y, phasor_displacement<direction>(x[n], &u[n * DIMS], &v[m * DIMS]));
          // y = cuCadd(y, phasor_displacement<direction>(x[0], &u[0 * DIMS], &v[m * DIMS]));
        }
#else
        // ------------------------------------------------------------
        for (size_t n = tid.x; n < N; n += gridSize.x) {
#ifdef V_SHARED
          if (shared_memory && blockDim_x >= DIMS && N >= DIMS && M >= gridSize.y)
            y = cuCadd(y, phasor_displacement<direction>(x[n], &u[n * DIMS], &v_shared[threadIdx.y * DIMS]));
          else
#endif
            y = cuCadd(y, phasor_displacement<direction>(x[n], &u[n * DIMS], &v[m * DIMS]));
        }
        // ------------------------------------------------------------
#endif
#ifdef V_SHARED
        if (shared_memory && blockDim_x >= DIMS && N >= DIMS && M >= gridSize.y)
          __syncthreads();
#endif
#ifdef TEST_CONST_PHASE
        for (size_t n = tid.x; n < N; n += gridSize.x)
          y = cuCadd(y, from_polar(1., 0.));
#endif
        // alt // https://github.com/thrust/thrust/blob/master/examples/sum_rows.cu
        if (shared_memory) {
          // TODO don't save result to every thread, only thread 0
          // TODO what about unused thread in reduction? -> should cause deadlock
          // Real part .x
          y.x = BlockReduce(y_shared[threadIdx.y]).Sum(y.x);
          // TODO is a sync here required?
          // __syncthreads();
          // TODO mv first global mem acces here? -> hide memory latency
          // Imaginary part .y
          y.y = BlockReduce(y_shared[threadIdx.y]).Sum(y.y);
          // __syncthreads();

          if (threadIdx.x == 0) {
#ifdef TEST_CONST_PHASE
            assert(blockDim_y == blockDim.y);
            assert(y.x == blockDim_x);
            assert(y.y == 0.);
#endif
            const size_t i = Yidx(blockIdx.x, m, MIN(N_stride, gridDim.x), M);
            if (append_result)
              {y_global[i].x += y.x; y_global[i].y += y.y;}
            else
              y_global[i] = y;
          }
        }
        else {
          const size_t i = Yidx(tid.x, m, MIN(N_stride, gridSize.x), M);
          // printf("y[%i] or y[%i, %i]: amp = %e, \tangle = %e\n", i, tid.x, m, cuCabs(y), angle(y));
          if (append_result)
            {y_global[i].x += y.x; y_global[i].y += y.y;}
          else
            y_global[i] = y;
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
