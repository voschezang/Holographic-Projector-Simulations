#ifndef ALGEBRA
#define ALGEBRA

#include <stdlib.h>
#include <algorithm>
#include <limits.h>
#include <numeric>

/**
 * Simple math & linear algebra functions, vector operations (not fully optimized)
 */

// Vector operations

template<typename T = double>
void add(std::vector<T> &x, const T value) {
  // add constant to vector (in place)
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] += value;
}

template<typename T = WAVE>
void add_complex(std::vector<T> &x, const T value) {
  // add constant to vector (in place)
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = cuCadd(x[i], value);
}

template<typename T = WAVE>
void add_complex(std::vector<T> &x, const std::vector<T> &y) {
  // elementwise sum of vectors (in place)
  assert(x.size() == y.size());
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = cuCadd(x[i], y[i]);
}

template<typename T = double>
void normalize(std::vector<T> &x, T to = 1) {
  if (x.size() == 0) return;
  const auto max_inv = to / (T) std::max_element(x.begin(), x.end())[0];
  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] *= max_inv;
}

// Generate vectors

std::vector<int> range(size_t len) {
  // similar to numpy.arrange
  assert(len < INT_MAX);
  auto values = std::vector<int>(len);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

std::vector<double> linspace(size_t len, double min = 0., double max = 1.) {
  // similar to Armadillo linspace
  assert(len > 0);
  auto values = std::vector<double>(len, min); // init to `min`
  if (len == 1)
    return values;

  const auto
    range = max - min,
    delta = range / (len - 1.0);

  // start at 0 to maintain alignment
  for (size_t i = 0; i < len; ++i)
    values[i] += delta * i;

  // test relative error
  if (max != 0.)
    assert(abs((values[len - 1] - max) / max) < 1e-3);

  return values;
}

std::vector<double> logspace(size_t len, double a, double b, double base = 10.) {
  // Return a sequence from 10^a to 10^b, spaced evenly over a logarithmic scale.
  // TODO change argument order to (a,b,len,base), idem for linspace, geomspace
  assert(len > 0);
  auto values = linspace(len, a, b);
  for (size_t i = 0; i < len; ++i)
    values[i] = pow(base, values[i]);

  return values;
}

std::vector<double> geomspace(size_t len, double a, double b) {
  // Return a sequence from a to b, spaced evenly over a logarithmic scale.
  return logspace(len, log10(a), log10(b), 10.);
}


// Statistics

template<typename T = double>
inline T sum(const T* x, const size_t len) {
  return std::accumulate(x, x + len, (T) 0);
}

template<typename T = double>
inline T sum(const std::vector<T> &x) {
  return std::accumulate(x.begin(), x.end(), (T) 0);
}

template<typename T = cuDoubleComplex>
inline double transform_reduce(const std::vector<T> &x, double (*transform)(T)) {
  // transform input vector with function f and sum the result
  // will be included in c++17
  auto op = [transform](double acc, T next) { return acc + transform(next); };
  return std::accumulate(x.begin(), x.end(), (double) 0., op);
}

template<typename T = double>
inline double mean(const T* x, const size_t len) {
  assert(len > 0);
  // note that mean of int vector is a double
  return sum(x, len) / (double) len;
}

template<typename T = double>
inline double mean(const std::vector<T> &x) {
  return mean<T>(x.data(), x.size());
}

template<typename T = double>
inline double sample_variance(const std::vector<T> &x) {
  // Return the (uncorrected) sample variance = E[(x - E[x])^2]
  assert(x.size() > 1);
  const double mu = mean(x);
  double acc = 0.0;

  for (auto& value : x)
    acc += (value - mu) * (value - mu);

  return acc / (double) x.size();
}

template<typename T = double>
inline double variance(const std::vector<T> &x) {
  // Estimate population variance based on sample.
  // Defined as n/(n-1) Var[x]
  return x.size() / (x.size() - 1.) * sample_variance(x);
}


// Util

inline double lerp(double a, double b, double ratio) {
  // Linear interpolation of two numbers a, b. Lazy version of linspace
  return (1 - ratio) * a + ratio * b;
}
inline double lerp(Range<double> x, double ratio) {
  return lerp(x.min, x.max, ratio);
}

inline double gerp(double a, double b, double ratio) {
  // Geometric interpolation of two numbers a, b. Lazy version of geomspace
  if (a == b) return a;
  if (a == 0) a = 1e-12;
  if (b == 0) b = 1e-12;
  return pow(10, lerp(log10(a), log10(b), ratio));
}
inline double gerp(Range<double> x, double ratio) {
  return gerp(x.min, x.max, ratio);
}


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace algebra {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void test() {
  // TODO check nan + inf values

  // empty vector has zero sum
  assert(sum(std::vector<int>{}) == 0);

  assert(sum(std::vector<int>{0, 1}) == 1);
  assert(sum(std::vector<double>{1.0, 0.5}) == 1.5);
  assert(mean(std::vector<double>{0,1}) == 0.5);
  assert(sample_variance(std::vector<double>{0,1}) == 0.25);

  assert(transform_reduce({{1,0}, {2.5,0}, {6.5,0}}, cuCabs) == 10.0);

  assert(lerp(0, 1, 0.9) == 0.9);
  assert(lerp(0, 2, 1) == 2);
  assert(gerp(1, 2, 0.5) - 1.41421 <= 1e-5);
  assert(gerp(1, 2, 0) == 1);
  assert(gerp(1, 2, 1) == 2);
  assert(lerp(3, 3, 0.3) - 3 < 1e-6);
  assert(lerp(4, 4, 0.8) - 4 < 1e-6);
  assert(gerp(3, 3, 0.3) - 3 < 1e-6);
  assert(gerp(4, 4, 0.8) - 4 < 1e-6);
  assert(gerp(0, 0, 0.8) < 1e-6);
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
