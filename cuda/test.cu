
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
#include <thrust/host_vector.h> // unused in this file but causes error if omitted

#include "macros.h"
#include "hyper_params.h"
#include "util.h"
#include "kernel.cu"

void test_complex(cuDoubleComplex c, double max_rel_error=1e-6, int v=0) {
  assert(c.x == cuCreal(c));
  assert(c.y == cuCimag(c));
  double
    r = cuCabs(c),
    phi = angle(c);
  auto c2 = from_polar(r, phi);
  if (v) printf("r: %f phi: %f\n", r, phi);
  if (v) printf("c: %f+%fj v.s. %f+%fj\n", c.x, c.y, c2.x, c2.y);
  if (c.x != 0) assert(abs(c.x - c2.x) < max_rel_error * abs(c.x));
  else          assert(abs(c.x - c2.x) < max_rel_error);
  if (c.y != 0) assert(abs(c.y - c2.y) < max_rel_error * abs(c.y));
  else          assert(abs(c.y - c2.y) < max_rel_error);
}

int main() {
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
  std::cout << "DONE\n";
}
