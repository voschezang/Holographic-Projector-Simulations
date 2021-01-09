#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <stdlib.h>
#include <numeric>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>     // std::ostream_iterator
#include <thrust/host_vector.h> // causes error if omitted
#include <thrust/device_vector.h>

#include "macros.h"
#include "util.h"
#include "kernel.cu"
#include "init.h"
#include "superposition.cu"
#include "transform.cu"

void test_sum_rows() {
  // `y = alpha * op(A)x + beta y`
  const size_t width = 1024, n_rows = 512;
  WAVE beta = {1,0};
  thrust::host_vector<WAVE>
    A (width * n_rows, {1,0}),
    x(width, {1,0}),
    y(n_rows, {3.21,0});
  thrust::device_vector<WAVE> d_A = A, d_x = x, d_y = y;
  const auto
    d_A_ptr = thrust::raw_pointer_cast(&d_A[0]),
    d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);
  const auto d_x_vec = ConstCUDAVector<WAVE> {thrust::raw_pointer_cast(&d_x[0]),
                                                 d_x.size()};
  cublasHandle_t handle;
  cuB( cublasCreate(&handle) );
  kernel::sum_rows<false>(width, n_rows, handle, d_A_ptr, d_x_vec, d_y_ptr, {0,0});
  y = d_y;
  // for (size_t i = 0; i < n_rows; ++i) {
  //   printf("y[%lu]: %0.1f + width = %lu \t%0.1f\n", i, y[i].x - width, y[i].x, y[i].y);
  //   assert(equals(y[i].x, width));
  // }
  for (size_t i = 0; i < n_rows; ++i)
    assert(equals(y[i].x, width));
  kernel::sum_rows<false>(width, n_rows, handle, d_A_ptr, d_x_vec, d_y_ptr, beta);
  y = d_y;
  for (size_t i = 0; i < n_rows; ++i)
    assert(equals(y[i].x, 2*width));
  cuB( cublasDestroy(handle) );
}

void test_superposition() {
  const int blockDim_x = 8, blockDim_y = 4, // template requires constant lvalue
    n_batches = 4;
  const dim3
    blockDim (blockDim_x, blockDim_y),
    gridDim (4, 4),
    gridSize (gridDim.x * blockDim.x, gridDim.y * blockDim.y),
    thread_size (2, 2); // datapoints per thread
  const size_t
    N_max = gridSize.x * thread_size.x * n_batches,
    M = gridSize.y * thread_size.y * n_batches + 3;

  double amp = LAMBDA, phi = 0.3456, delta = amp;

  auto d_x = thrust::device_vector<Polar> (N_max, {amp, phi});
  auto d_y = thrust::device_vector<WAVE> (M * N_max, {0.,0.});
  thrust::host_vector<Polar> x = d_x;
  thrust::host_vector<WAVE> y = d_y;
  std::vector<double>
    u (DIMS * N_max, 0.),
    v (DIMS * M, 0.);
  for (size_t n = 0; n < N_max; ++n)
    u[n*DIMS+2] = delta; // maintain amp
  thrust::device_vector<double> d_u = u, d_v = v;

  const auto d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);
  const auto d_u_ptr = thrust::raw_pointer_cast(&d_u[0]);
  const auto d_v_ptr = thrust::raw_pointer_cast(&d_v[0]);

  superposition::phasor_displacement<Direction::Forwards><<<1,1>>>(d_x[0], d_u_ptr, d_v_ptr, d_y_ptr);
  cudaDeviceSynchronize();
  // printf("y[0]: amp = %e, \tangle = %e\n", cuCabs(y[0]), angle(y[0]));
  // printf("y[1]: amp = %e, \tangle = %e\n", cuCabs(y[1]), angle(y[1]));
  y = d_y;
  // printf("y[0]: amp = %e, \tangle = %e\n", cuCabs(y[0]), angle(y[0]));
  assert(equals(cuCabs(y[0]), 1.));
  assert(equals(angle(y[0]), phi));

  for (auto& N : std::array<size_t, 2> {1, N_max}) {
    // Naive
    superposition::per_block<Direction::Forwards, blockDim_x, blockDim_y, Algorithm::Naive, false><<<gridDim,blockDim>>>(N, M, N, d_x_ptr, d_u_ptr, d_v_ptr, d_y_ptr, false);
    y = d_y;
    // printf("x[0]: amp = %e, \tangle = %e\n", cuCabs(x[0]), angle(x[0]));
    for (size_t i = 0; i < N; ++i) {
      const size_t j = Yidx(0, i, N, M);
      assert(equals(cuCabs(y[j]), 1.));
      assert(equals(angle(y[i]), phi));
    }
    superposition::per_block<Direction::Forwards, blockDim_x, blockDim_y, Algorithm::Naive, false><<<gridDim,blockDim>>>(N, M, N, d_x_ptr, d_u_ptr, d_v_ptr, d_y_ptr, true);
    y = d_y;
    for (size_t i = 0; i < N; ++i) {
      const size_t j = Yidx(0, i, N, M);
      assert(equals(cuCabs(y[j]), 2.));
      assert(equals(angle(y[i]), phi));
    }
    // Alt, no shared memory
    superposition::per_block<Direction::Forwards, blockDim_x, blockDim_y, Algorithm::Reduced, false><<<gridDim,blockDim>>>(N, M, N, d_x_ptr, d_u_ptr, d_v_ptr, d_y_ptr, false);
    y = d_y;
    // printf("x[0]: amp = %e, \tangle = %e\n", cuCabs(x[0]), angle(x[0]));
    size_t N_out = MIN(N, gridSize.x);
    // printf("N: %i, N_out: %i, thread: %i\n", N, N_out, N / N_out);
    for (size_t n = 0; n < N_out; ++n) {
      for (size_t m = 0; m < M; ++m) {
        const size_t i = Yidx(n, m, N_out, M);
        // printf("y[%i] or y[%i, %i]: amp = %e, \tangle = %e\n", i, n, m, cuCabs(y[i]), angle(y[i]));
        assert(equals(cuCabs(y[i]), N / N_out));
        assert(equals(angle(y[i]), phi));
      } }
    // Alt, shared memory
    // TODO
  }

  // parallel superposition
  Geometry p;
  p.algorithm = 1;
  p.blockDim = blockDim;
  p.gridDim = gridDim;
  const bool allow_random = 1;
  for (auto& n_streams : std::array<int, 2> {1, n_batches / 2}) {
    p.n_streams = n_streams;
    for (auto& thread_size_x : std::array<size_t, 2> {16, 1}) {
      for (auto& thread_size_y : std::array<size_t, 2> {16, 1}) {
        p.thread_size = {thread_size_x, thread_size_y};
        // Note the underutilized batches for N = non powers of 2
        for (auto& N : std::array<size_t, 3> {1, 30, N_max - 9}) {
          assert(N <= N_max);
          p.n = {N, M};
          init::derive_secondary_geometry(p);
          if (N <= p.batch_size.x) assert(p.n_batches.x == 1);
          if (M <= p.batch_size.y) assert(p.n_batches.y == 1);
          const std::vector<Polar> X (x.begin(), x.begin() + N);
          auto z = transform_full<Direction::Forwards, Algorithm::Naive, false>(X, u, v, p);
          printf("\n\tparams: thread_size: %lu, %lu \t n_streams: %i, batch size: %lu, %lu\n",
                 thread_size_x, thread_size_y, n_streams, p.batch_size.x, p.batch_size.y);
          for (size_t i = 0; i < M; ++i) {
            printf("z[%i]: %f, %f - N: %lu\n", i, cuCabs(z[i]), angle(z[i]), N);
            assert(equals(angle(z[i]), phi));
            if (!allow_random)
              assert(equals(cuCabs(z[i]), N));
          }
          z = transform_full<Direction::Forwards, Algorithm::Reduced, false>(X, u, v, p);
          for (size_t i = 0; i < M; ++i) {
            assert(equals(angle(z[i]), phi));
            if (!allow_random)
              assert(equals(cuCabs(z[i]), N));
          }
          z = transform_full<Direction::Forwards, Algorithm::Reduced, true>(X, u, v, p);
          for (size_t i = 0; i < M; ++i) {
            assert(equals(angle(z[i]), phi));
            if (!allow_random)
              assert(equals(cuCabs(z[i]), N));
          }
        }
      }
    }
  }
}

int main() {
  test_sum_rows();
  test_superposition(); // TODO add better tests
  std::cout << "GPU DONE\n";
}
