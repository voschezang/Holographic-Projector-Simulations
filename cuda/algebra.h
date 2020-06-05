#ifndef ALGEBRA
#define ALGEBRA

#include <stdlib.h>
#include <limits.h>
#include <numeric>

/**
 * Simple math & linear algebra functions, vector operations (not fully optimized)
 */


// Statistics

template<typename T>
inline T sum(std::vector<T> x) {
  return std::accumulate(x.begin(), x.end(), (T) 0);
}

template<typename T>
inline double mean(std::vector<T> x) {
  // note that mean of int vector is a double
  return sum(x) / (double) x.size();
}

template<typename T>
inline double sample_variance(std::vector<T> x) {
  // Return the (uncorrected) sample variance = E[(x - E[x])^2]
  assert(x.size() > 1);
  const double mu = mean(x);
  double acc = 0.0;

  for (auto& value : x)
    acc += (value - mu) * (value - mu);

  return acc / (double) x.size();
}

template<typename T>
inline double variance(std::vector<T> x) {
  // Estimate population variance based on sample.
  // Defined as n/(n-1) Var[x]
  return x.size() / (x.size() - 1.) * sample_variance(x);
}

// Generate vectors

std::vector<int> range(size_t len) {
  // similar to numpy.arrange
  assert(len < INT_MAX);
  auto values = std::vector<int>(len);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

std::vector<double> linspace(size_t len, double min, double max) {
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

std::vector<double> logspace(size_t len, double a, double b, double base) {
  // Return a sequence from 10^a to 10^b, spaced evenly over a logarithmic scaling.
  // TODO change argument order to (a,b,len,base), idem for linspace, geomspace
  assert(len > 0);
  auto values = linspace(len, a, b);
  for (size_t i = 0; i < len; ++i)
    values[i] = pow(base, values[i]);

  return values;
}

std::vector<double> logspace(size_t len, double a, double b) {
  return logspace(len, a, b, 10.);
}

std::vector<double> geomspace(size_t len, double a, double b) {
  // Return a sequence from a to b, spaced evenly over a logarithmic scale.
  return logspace(len, log10(a), log10(b), 10.);
}



///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace algebra {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void test() {
  double a = sum(std::vector<int>{0, 1});
  print(a);
  assert(sum(std::vector<int>{0, 1}) == 1);
  assert(sum(std::vector<double>{1.0, 0.5}) == 1.5);
  assert(mean(std::vector<double>{0,1}) == 0.5);
  assert(sample_variance(std::vector<double>{0,1}) == 0.25);
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
