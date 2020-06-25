
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
#include "hyper_params.h"
#include "util.h"
#include "kernel.cu"
#include "init.h"
#include "superposition.cu"
#include "functions.cu"

inline
bool equals(double a, double b, double max_rel_error=1e-6) {
  // T : float or double
  // ignore near-zero values
  const double non_zero_scalar = abs(a) >= 1e-9 ? abs(a): 1.;
  return abs(a - b) < max_rel_error * non_zero_scalar;
}

void test_complex(cuDoubleComplex c, int v=0) {
  assert(c.x == cuCreal(c));
  assert(c.y == cuCimag(c));
  double
    r = cuCabs(c),
    phi = angle(c);
  auto c2 = from_polar(r, phi);
  if (v) printf("r: %f phi: %f\n", r, phi);
  if (v) printf("c: %f+%fj v.s. %f+%fj\n", c.x, c.y, c2.x, c2.y);
  assert(equals(c.x, c2.x));
  assert(equals(c.y, c2.y));
}

void test_complex_multiple() {
  test_complex({0, 0});
  test_complex({3, 0});
  test_complex({0, 2});
  test_complex({-1.1, -0.99});
  test_complex({0.123, 4.3231});
  test_complex({-0.823, 4.75});
  test_complex({9.99, -6.1});
  assert(from_polar(-1, 0.82).x + 0.6822212 < 1e-3);
  assert(from_polar(-1, 0.82).y + 0.7311458 < 1e-3);
  auto c = cuDoubleComplex{1.8, 4.2};
  assert(cuCabs(c) - 4.56946 < 1e-3);
  assert(angle(c) - 1.16590 < 1e-3);
}

void test_normalize_amp() {
  auto x = std::vector<WTYPE> {{1,2}, {-1,4}, {8,12}, {11,-3}};
  for (double to = 0.5; to < 5; to*=3.14) {
    normalize_amp<false>(x, to);
    for (int i = 0; i < x.size(); ++i)
      assert(cuCabs(x[i]) - to <= 1e-6);
  }
}

template<const Direction direction = Direction::Forwards>
void test_superposition_single(std::vector<WTYPE> &x, std::vector<STYPE> &u, std::vector<STYPE> &v,
                               double amp, double phi) {
  size_t n = x.size();
  // TODO test function on device

  // init device memory
  const thrust::device_vector<WTYPE> d_x = x;
  const thrust::device_vector<STYPE>
    d_u = u,
    d_v = v;
  thrust::device_vector<double> d_y(n * 2);
  // cast to pointers to allow usage in non-thrust kernels
  const auto d_x_ptr = thrust::raw_pointer_cast(&d_x[0]);
  const auto
    d_u_ptr = thrust::raw_pointer_cast(&d_u[0]),
    d_v_ptr = thrust::raw_pointer_cast(&d_v[0]);
  auto d_y_ptr = thrust::raw_pointer_cast(&d_y[0]);

  // test conversion host <-> device
  thrust::host_vector<WTYPE> x2 = d_x;
  for (int i = 0; i < n; ++i)
    assert(x[i].x == x2[i].x && x[i].y == x2[i].y);

  // test superposition functions
  // on host
  WTYPE y0 = superposition::single<direction>(0, 0, &x[0], &u[0], &v[0]);

  // auto p = init::simple_geometry(1);
  // superposition::per_block<Direction::Forwards, 1><<< p.gridSize, p.blockSize, 0 >>> \
  //   (p, d_x_ptr, n, d_u_ptr, d_y_ptr, d_v_ptr );

  // thrust::host_vector<double> y_tmp = d_y;
  // WTYPE y0 {y_tmp[0], y_tmp[1]};
  // printf("amp: %f, phase %f\n", cuCabs(y0), angle(y0));
  assert(equals(amp, cuCabs(y0)));
  assert(equals(phi, angle(y0)));
  // auto a = cuCabs(y0), phi = angle(y0);
  // // printf("a: %f, phi: %f\n", a, phi);
  // assert(equals(a, 1538461.5384615385));
  // assert(equals(phi, 0.)); // ~1e-12 in python

  // for (int i = 0; i < n; ++i) {
  //   const unsigned int j = i * p.gridSize * p.kernel_size; // * 2
  //   const unsigned int k = i * p.kernel_size;
  //   superposition::per_block<Direction::Forwards, 1><<< p.gridSize, p.blockSize, 0 >>> \
  //     (p, d_x, n, d_u, &d_y_ptr[j], &d_v[k * DIMS] );
  // }
}

void test_superposition() {
  auto x = std::vector<WTYPE>{{1,0}};
  auto
    u = std::vector<STYPE>{0,0,0},
    v = std::vector<STYPE>{0,0,LAMBDA};

  assert(equals(norm3d_host(u[0] - v[0], u[1] - v[1], u[2] - v[2]) / LAMBDA, 1.));
  test_superposition_single(x, u, v, 1538461.5384615385, 0.);
  x[0].x = 1.31; x[0].y = -2.1;
  v[1] = 2.22e9 * LAMBDA;
  v[2] = 12.1 * LAMBDA;
  double
    distance = norm3d_host(u[0] - v[0], u[1] - v[1], u[2] - v[2]),
    phi_next = angle(x[0]) + distance  * TWO_PI_OVER_LAMBDA;
  assert(equals(cuCabs(x[0]), 2.4750959577357805));
  assert(equals(cuCabs(x[0]) / distance / LAMBDA, 2638.8357137755534));
  assert(equals(distance / LAMBDA, 2.22e9));
  assert(equals(phi_next, 13948671379.83868));
  test_superposition_single<Direction::Backwards>(x, u, v, 0.0017152432139541098, -1.0130525640725432);
  x[0].x = 3.33; x[0].y = 4.44;
  test_superposition_single<Direction::Backwards>(x, u, v, 0.003846153846153846, 0.9272970147848789);
  v[1] = 5.51e3 * LAMBDA;
  test_superposition_single(x, u, v, 1549.6263067864088, 1.0107725366705158);
}

int main() {
  algebra::test();
  test_normalize_amp();
  test_complex_multiple();
  // TODO add lin_algebra tests
  test_superposition();
  std::cout << "DONE\n";
}
