#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <limits.h>
#include <numeric>
#include <numeric>
#include <time.h>
#include <iostream>
#include <iterator>     // std::ostream_iterator
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h> // unused in this file but causes error if omitted
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "macros.h"
#include "main.h"
#include "kernel.cu"
#include "util.h"
#include "superposition.cu"

// note -O3 is (default)
// nvcc -o uti test_utilization.cu -std=c++14 -arch=compute_70 -code=sm_70 && ./uti && rm uti
// nvcc -o uti test_utilization.cu -std=c++14 -arch=compute_70 -code=sm_70 -Xptxas -O3,-v maxrrregcount=1 && ./uti && rm uti


#define CHECK_NAN 0
#define UNROLL 2
#define VECTORIZE 0 // in theory, parallelize on thread-level
#define SUPERPOSITION 0

#define N_per_thread 1
#define M_per_thread 1024

#define FLOP_per_point_K 1


template<int operation = 2, typename T = double>
__global__ void dummy(T *out, size_t n = 1) {
  cuDoubleComplex result;
  const T c = 1.01, c1 = 0.001;
#if (VECTORIZE)
  T x[UNROLL]; // best performance for UNROLL = 2
  for (int j = 0; j < UNROLL; ++j) x[j] = c + j * 1e-1;
#else
  T x = c;
#endif
  for (int k = 0; k < n; ++k)
    for (long int i = N_per_thread * M_per_thread / UNROLL; i > 0; --i) {
#pragma unroll
      for (int j = 0; j < UNROLL; ++j) {
#if (VECTORIZE)
        if (operation == 0) x[j] += c;
        if (operation == 2) x[j] *= c;
        if (operation == 9) {x[j] = (x[j] * c) + c1;}
#else
        if (operation == 0) x += c;
        if (operation == 1) x -= c;
        if (operation == 2) x *= c;
        if (operation == 3) x /= c;
        if (operation == 4) x = pow(x, c);
        if (operation == 5) x = sqrt(x);
        if (operation == 6) cos_sin(x, &result.x, &result.y);
        if (operation == 7) {x += c; x -= c; x += c; x = sqrt(x);}
        if (operation == 8) {long long int x2 = x, diff = x2 / TWO_PI; cos_sin(x - diff * x2, &result.x, &result.y);} // modulo TWO_PI
        if (operation == 9) {x = (x * c) + c1;}
        if (operation == 10) {x = x * (c1 + x);}
        if (operation == 11) {
          // excl. norm3d
          x = c + (x * TWO_PI_OVER_LAMBDA) / c;
          cos_sin(x, &result.x, &result.y);
          x *= x / c;
        }
        // if (operation == 12) {x += x; x1 += x1;}
        // if (operation == 13) {x *= x; x1 *= x1;}
#endif
      }
    }
#if (CHECK_NAN)
  // for (int j = 0; j < UNROLL; ++j) {
  //   assert(!isinf(x[j]));
  //   assert(!isnan(x[j]));}
  assert(!isinf(x));
  assert(!isnan(x));
#endif
  // write something to memory (force side effect), otherwise CUDA won't execute properly
  // assume y-dim = 1
  if (blockIdx.x == 0 && threadIdx.x == 0) {
#if (VECTORIZE)
    // Note that all elements in array x must be potentially used
    for (int j = 0; j < UNROLL; ++j) out[j] = x[j];
#else
    out[0] = x;
#endif
  }
}


__global__ void superpos(double *out, size_t n = 1) {
  __shared__ double x[8];
  WAVE y;
  // assume y-dim = 1
  if (blockIdx.x == 0 && threadIdx.x < 8)
    x[threadIdx.x] = 1.00234 + threadIdx.x * 0.0531;

  __syncthreads();
  for (int k = 0; k < n; ++k)
    for (long int i = N_per_thread * M_per_thread / UNROLL; i > 0; --i) {
#pragma unroll
      for (int j = 0; j < UNROLL; ++j) {
        const double distance = norm3d(x[2] - x[3], x[4] - x[5], x[6] - x[8]);
        y = from_polar(x[0] / distance, x[1] - distance * TWO_PI_OVER_LAMBDA);
      } }

  // write something to memory (force side effect), otherwise CUDA won't execute properly
  if (blockIdx.x == 0 && threadIdx.x == 0)
    out[blockIdx.x] = y.x;
}


__global__ void superpos2(double *out, size_t n = 1,  size_t seed = 0) {
  // Runtime mean 	0.155 s, 	std: 0.000
  //   per kernel 	3.031e-05 s
  //   Efficiency 	0.276727 TFLOPS, 	 N: 4.295e+10
  //   Efficiency 	276.726891 GFLOPS, 	 N: 4.295e+10

  //   Runtime mean 	1.280 s, 	std: 0.000
  //   per kernel 	2.501e-04 s
  //   Efficiency 	0.335430 TFLOPS, 	 N: 4.295e+11
  //   Efficiency 	335.429869 GFLOPS, 	 N: 4.295e+11

  //   Runtime mean 	14.076 s, 	std: 0.523
  //   per kernel 	2.749e-03 s
  //   Efficiency 	0.305129 TFLOPS, 	 N: 4.295e+12
  //   Efficiency 	305.129071 GFLOPS, 	 N: 4.295e+12

  const dim3
    tid (blockIdx.x * blockDim.x + threadIdx.x,
         blockIdx.y * blockDim.y + threadIdx.y),
    gridSize (blockDim.x * gridDim.x,
              blockDim.y * gridDim.y);
  const unsigned int global_tid = tid.x + tid.y * gridSize.x;
  double c0 = 1 + 1. / (1 + global_tid + seed);
  Polar x {1, c0}; WAVE y {0,0};
  double args[6] = {1,2,3,4,5,6};
  for (size_t k = 0; k < n; ++k)
    for (size_t i = 0; i < N_per_thread; ++i) {
      for (size_t j = 0; j < M_per_thread; ++j) {
        // y = cuCadd(y, superposition::phasor_displacement<Direction::Forwards>(x, args+0, args+3));
        y = superposition::phasor_displacement<Direction::Forwards>(x, args+0, args+3);
      } }
  // write something to memory (force side effect), otherwise CUDA won't execute properly
  if (global_tid == 0)
    out[global_tid] = y.x;
}


std::string key(int operation) {
  if (operation == 0) return "add";
  if (operation == 1) return "sub";
  if (operation == 2) return "mul";
  if (operation == 3) return "div";
  if (operation == 4) return "pow";
  if (operation == 5) return "sqr";
  if (operation == 6) return "exp";
  if (operation == 7) return "nor";
  if (operation == 8) return "exx";
  if (operation == 9) return "fma";
  if (operation == 10) return "fam";
  if (operation == 11) return "sup";
  if (operation == 12) return "ad2";
  if (operation == 13) return "mu2";
  return "...";
}

template<int operation = 2, typename T = double>
double run_kernel(size_t n = 1, int v = 0) {
  // max: 1024 threads per block
  // const dim3 GridDim (16, 1), BlockDim (512, 1);
  const dim3 GridDim (64, 1), BlockDim (512, 1);

  const size_t
    n_trials = 3,
    n_streams = 5,
    // n_batches_per_stream = 1024,
    // n_batches_per_stream = 64,
    n_batches_per_stream = 256,
    kernel_size = GridDim.x * GridDim.y * BlockDim.x * BlockDim.y,
    n_results = kernel_size * n_streams,
    N = N_per_thread * M_per_thread * n * kernel_size * n_batches_per_stream * n_streams;

  cudaStream_t streams[n_streams];
  for (auto& stream : streams)
    cu( cudaStreamCreate(&stream) );

  T *d_x;
  cu( cudaMalloc( (void **) &d_x, n_results * sizeof(double) ) );
  // warm up
#if (SUPERPOSITION)
  superpos<<<GridDim, BlockDim, 0, streams[0]>>>(d_x, 1);
#else
  dummy<operation, T><<<GridDim, BlockDim, 0, streams[0]>>>(d_x, 1);
#endif
  cu( cudaDeviceSynchronize() );

  struct timespec t1, t2;
  auto dt = std::vector<double>(n_trials, 0.0);

  for (auto& i : range(n_trials)) {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (size_t b = 0; b < n_batches_per_stream; ++b)
      for (size_t s = 0; s < n_streams; ++s) {
#if (SUPERPOSITION)
        superpos<<<GridDim, BlockDim, 0, streams[s]>>>(d_x + s * kernel_size, n);
#else
        dummy<operation, T><<<GridDim, BlockDim, 0, streams[s]>>>(d_x + s * kernel_size, n);
#endif
        // seed = b + i * n_batches_per_stream
      }
    cu( cudaDeviceSynchronize() );
    clock_gettime(CLOCK_MONOTONIC, &t2);
    dt[i] = diff(t1, t2);
  }

  for (auto& i : range(n_streams))
    cudaStreamDestroy(streams[i]);

  cu( cudaFree(d_x) );
  cu( cudaDeviceSynchronize() );

  auto mu = mean(dt), var = sample_variance(dt);
  auto flops = std::vector<double>(n_trials, 0.0);
  for (auto& i : range(n_trials))
    flops[i] = N * FLOP_per_point_K / dt[i];

  if (v) {
    std::cout << key(operation);
    std::cout << "," << n << "," << N << "," << mu << "," << var;
    std::cout << "," << mean(flops) << "," << sample_variance(flops) << '\n';
  }
  if (v > 1) {
    // Note,
    printf("Runtime mean \t%0.3f s, \tstd: %.3f\n", mu, var);
    printf("\tper kernel \t%0.3e s\n", mu / (double) (n_streams * n_batches_per_stream) );
    printf("Efficiency \t%0.6f TFLOPS, \t N: %lu\n", mean(flops) * 1e-12, N);
    printf("Efficiency \t%0.6f GFLOPS, \t N: %.3e\n", mean(flops) * 1e-9, (double) N);
  }
  return mean(flops) * 1e-12;
}

int main(void)
{
  int v = 1;
  size_t n = 1;
  // single precision is ~ 2x as fast as double precision
  run_kernel<2>(n, 2);
  // run_kernel<2, float>(n, 1);
  for (n = 1; n <= 32; n *= 32) {
    // run_kernel<2>(n, 1);
    // run_kernel<0>(n,v); // add
    // run_kernel<1>(n,v); // sub
    // run_kernel<2>(n,v); // mul
    // run_kernel<3>(n,v); // div
    // // run_kernel<4>(n,v); // pow
    // run_kernel<5>(n,v); // sqr
    // run_kernel<6>(n,v); // exp
    // // run_kernel<7>(n,v); // nor
    // // run_kernel<8>(n,v); // exx
    // run_kernel<9>(n,v); // fma
    // run_kernel<10>(n,v); // fam
    // run_kernel<11>(n,v); // sup
    // run_kernel<12>(n,v); // ad2 add twice (concurrent)
    // run_kernel<13>(n,v); // mu2 multiply (concurrent)
    // std::cout << "add," << n << "," << run_kernel<0>(n, v) << '\n';
    // std::cout << "sub," << n << "," << run_kernel<1>(n, v) << '\n';
    // std::cout << "mul," << n << "," << run_kernel<2>(n, v) << '\n';
    // std::cout << "div," << n << "," << run_kernel<3>(n, v) << '\n';
    // std::cout << "pow," << n << "," << run_kernel<4>(n, v) << '\n';
    // std::cout << "sqr," << n << "," << run_kernel<5>(n, v) << '\n';
    // std::cout << "exp," << n << "," << run_kernel<6>(n, v) << '\n';
    // std::cout << "nor," << n << "," << run_kernel<7>(n, v) << '\n';
    // std::cout << "fma," << n << "," << run_kernel<9>(n, v) << '\n';
    // std::cout << "fam," << n << "," << run_kernel<10>(n, v) << '\n';
    // std::cout << "sup," << n << "," << run_kernel<11>(n, v) << '\n';
    std::cout << "\n";
  }
  return 0;
}
