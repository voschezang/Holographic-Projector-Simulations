#ifndef UTIL
#define UTIL

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
/* #include <type_traits> */

#include "macros.h"
#include "kernel.cu"

enum class FileType {TXT, DAT, GRID};
enum class Shape {Line, Circle};
enum class Variable {Offset, Width};

/* Geometry Hierarchy (parameters)
 * thread < block < grid < kernel < batch < stream
 * (hyper parameters are defined using macro's, to avoid dynamic memory)
 *
 * note that 1 kernel computes 1 or more superpositions w.r.t all input datapoints.
 */
struct Geometry {
  size_t blockSize;   // prod(blockDim.x,y,z), i.e. threads per block
  size_t gridSize;    // prod(gridDim.x,y,z), i.e. n blocks per kernel
  size_t kernel_size; // n output datapoints per kernel
  size_t batch_size;  // n kernels per batch
  size_t stream_size; // n batches per stream
  size_t n_streams;

  // secondary
  // "getters"
  // TODO use class with lazy methods

  // total number of ..
  size_t n_batches; // total n stream batches
  size_t n_kernels; // total n kernel calls (excl. agg)

  // n datapoints per ..
  size_t n_per_stream;
  size_t n_per_batch;
  size_t n_per_kernel; // i.e. per grid
  double n_per_block;
  double n_per_thread;

  // TODO differentiate between input and output datapoints
  /* double n_per_block; // can be < 1, i.e. not all blocks are used */
  /* double n_per_thread; // can be < 1, i.e. not all threads are used */

  size_t kernels_per_stream;
};

struct Plane {
/* Plane() : width(1), z_offset(0), randomize(false) {}; */
  char name;
  double width;
  double z_offset;
  bool randomize;
};

struct Params {
  // Simulation parameters, used to init plane distributions
  Plane input; // ground truth
  Plane projector;
  std::vector<Plane> projections;
  /* Geometry g; */
};

// TODO
/* class Params { */
/*  private: */
/*   // hierarchy */
/*   size_t */
/*     n_streams; */
/*     /\* _stream_size, *\/ */
/*     /\* _stream_batch_size, *\/ */
/*     /\* _batches_per_stream, *\/ */
/*     /\* _kernel_batch_size, *\/ */
/*     /\* _kernels_per_stream_batch; *\/ */

/*   // kernel */
/*   size_t blockSize; */
/*   size_t gridSize; */

/*  public: */
/*   size_t N; // number of "output" datapoints */
/*   Params(size_t N); */
/* } */

/* Params::Params(size_t n) { */
/*   n_streams = N_STREAMS; */
/* } */

/* Params::n_streams(size_t n) { */
/*   n_streams = n; */
/* } */


double flops(double runtime, size_t n, size_t m) {
  // Tera or Giga FLOP/s
  // :lscpu: 6 cores, 2x32K L1 cache, 15MB L3 cache
  // Quadro GV100: peak 7.4 TFLOPS (dp), 14.8 (sp), 59.3 (int)
  // bandwidth 870 GB/s
  //  cores: 5120, tensor cores 640, memory: 32 GB
  // generation Volta, compute capability 7.0
  // max size: 49 152
  // printf("fpp %i, t %0.4f, N*N %0.4f\n", FLOP_PER_POINT, t, N*N * 1e-9);
  return 1e-12 * n * m * (double) FLOP_PER_POINT / runtime;
}

double bandwidth(double runtime, const int n_planes, const char include_tmp) {
  // input phasor  + input space + output space
  const double unit = 1e-6; // MB/s
  double input = n_planes * N * (sizeof(WTYPE) + 2 * sizeof(STYPE));
  double output = n_planes * N * sizeof(WTYPE);
  if (!include_tmp)
    return unit * (input + output) / runtime;

  double tmp = GRIDDIM * SHARED_MEMORY_SIZE(BLOCKDIM) * sizeof(WTYPE);
  return unit * (input + output + tmp) / runtime;
}

void check(WTYPE  z) {
  /* double a = creal(z), b = cimag(z); */
  if (isnan(z.x)) printf("found nan re\n");
  if (isinf(z.x)) printf("found inf re\n");
  if (isnan(z.y)) printf("found nan I\n");
  if (isinf(z.y)) printf("found inf I\n");
  if (isinf(z.x)) exit(1);
}

void check_hyper_params(Geometry p) {
  assert(DIMS == 3);
  assert(SHARED_MEMORY_SIZE(p.blockSize) > 0);
  assert(REDUCE_SHARED_MEMORY >= 1);
  assert(REDUCE_SHARED_MEMORY <= CEIL(p.blockSize, 2));
}

void check_cvector(std::vector<WTYPE> x) {
#ifdef DEBUG
  for (size_t i = 0; i < x.size(); ++i)
    assert(cuCabs(x[i]) < DBL_MAX);
#endif
}

double memory_in_MB() {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = 2 * N * sizeof(WTYPE) + 2 * DIMS * N * sizeof(STYPE);
  return bytes * 1e-6;
}

std::vector<int> range(size_t len) {
  // similar to numpy.arrange
  auto values = std::vector<int>(len);
  /* for (size_t i = 0; i < len; ++i) */
  /*   values[i] += i; */
  std::iota(values.begin(), values.end(), 0);
  return values;
}

std::vector<double> linspace(size_t len, double min, double max) {
  // similar to Armadillo linspace
  assert(len > 0);
  assert(len > 1 || min == max);
  assert(max >= min);
  auto values = std::vector<double>(len, min); // init to `min`
  const auto range = max - min;
  const auto delta = range / len;
  for (size_t i = 1; i < len; ++i)
    values[i] += delta * i;

  assert(values[len - 1] - max < 0.001);
  return values;
}

void summarize_c(char name, std::vector<WTYPE> &x) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0, sum = 0;
  /* for (const auto& x : X) { */
  for (size_t i = 0; i < x.size(); ++i) {
    max_amp = fmax(max_amp, cuCabs(x[i]));
    min_amp = fmin(min_amp, cuCabs(x[i]));
    max_phase = fmax(max_phase , angle(x[i]));
    sum += cuCabs(x[i]);
  }
  double mean = sum / (double) x.size();
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f, mean: %f\n", name, min_amp, max_amp, max_phase, mean);
}

void normalize_amp(std::vector<WTYPE> &x, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < x.size(); ++i)
    max_amp = fmax(max_amp, cuCabs(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < x.size(); ++i) {
      x[i].x /= max_amp;
      x[i].y /= max_amp;
    }

  if (log_normalize)
    for (size_t i = 0; i < x.size(); ++i) {
      if (x[i].x > 0) x[i].x = -log(x[i].x);
      if (x[i].y > 0) x[i].y = -log(x[i].y);
    }
}

void summarize_double(char name, std::vector<double> &x) {
  double max = DBL_MIN, min = DBL_MAX;
  for (size_t i = 0; i < x.size(); ++i) {
    max = fmax(max, x[i]);
    min = fmin(min , x[i]);
  }
  printf("%c)  range: [%0.3f , %0.3f]\n", name, min, max);
}

void print_complex(WTYPE x, std::ofstream& out) {
  // check(x); //TODO uncomment
  // e.g. print as "1.2+3.4j" or "1.2-3.4j"
  if (x.y >= 0) {
    out << x.x << '+' << x.y << 'j';
  } else {
    out << x.x << x.y << 'j';
  }
}

template<typename T>
void map_to_and_write_array(std::vector<T> &x, double (*f)(T), const char sep, std::ofstream& out) {
  out << f(x[0]);
  for (size_t i = 1; i < x.size(); ++i)
    out << sep << f(x[i]);
}

template<typename T>
void write_array(std::vector<T> &x, const char sep, std::ofstream& out) {
  // TODO optimize
  out << x[0];
  for (size_t i = 1; i < x.size(); ++i)
    out << sep << x[i];
}

void write_complex_array(std::vector<WTYPE> &x, const char sep, std::ofstream& out) {
  out << x[0].x << sep << x[0].y;
  for (size_t i = 0; i < x.size(); ++i)
    out << sep << x[i].x << sep << x[i].y;
}

inline std::string quote(std::string s) {
  return "\"" + s + "\"";
}

inline std::string quote(char c) {
  return "\"" + std::string{c} + "\"";
}

/**
 * Parse as `{"k" <sep1> v <sep2>
 *            "k" <sep1> v <sep2>
 *            "k" <sep1> v}`
 */
void write_metadata(std::string phasor, std::string pos, Plane p, size_t len, std::ofstream& out) {
  // Use JSON-like separators with spaces for readiblity.
  const auto
    sep1 = ": ",
    sep2 = ", ";

  out << "{" \
      << quote("phasor")    << sep1 << quote(phasor) << sep2 \
      << quote("pos")       << sep1 << quote(pos)    << sep2 \
      << quote("z_offset")  << sep1 << p.z_offset    << sep2 \
      << quote("len")       << sep1 << len           << sep2 \
      << quote("precision") << sep1 << IO_PRECISION  << sep2 \
      << quote("dims")      << sep1 << DIMS          << "}\n";
}

void write_dot(char name, WTYPE *x, STYPE *u, size_t len) {
  char fn[] = "tmp/out-x.dat";
  fn[8] = name;
  FILE *out = fopen(fn, "wb");
  fprintf(out, "#dim 1 - dim 2 - dim 3 - Amplitude - Phase\n");
  for (size_t i = 0; i < len; ++i) {
    size_t j = i * DIMS;
    fprintf(out, "%f %f %f %f %f\n", u[j], u[j+1], u[j+2], cuCabs(x[i]), angle(x[i]));
  }
  fclose(out);
}

template <FileType type>
void write_arrays(std::vector<WTYPE> &x, std::vector<STYPE> &u,
                  std::string k1, std::string k2, bool overwrite, Plane p) {
  if (type != FileType::TXT) return;
  if (overwrite)
    print("Save results as txt");

  auto dir = std::string{"../tmp/"};
  auto mode = overwrite ? std::ofstream::binary : std::ofstream::app;
  std::ofstream out;
  out.open(dir + "out.json", mode);
  write_metadata(k1, k2, p, x.size(), out);
  out.close();

  /* out.open(dir + k1 + ".dat", mode); */
  /* out << std::scientific; // to allow e.g. 1.0e-50 */
  /* out << std::setprecision(IO_PRECISION); */
  /* write_complex_array(x, ',', out); */

  out << std::scientific; // to allow e.g. 1.0e-50
  out << std::setprecision(IO_PRECISION);

  out.open(dir + k1 + "_amp.dat", std::ofstream::binary);
  map_to_and_write_array(x, cuCabs, ',', out);
  out.close();

  out.open(dir + k1 + "_phase.dat", std::ofstream::binary);
  map_to_and_write_array(x, angle, ',', out);
  out.close();

  out.open(dir + k2 + ".dat", std::ofstream::binary);
  write_array(u, ',', out);
  out.close();
}

double dt(struct timespec t0, struct timespec t1) {
  return (double) (t1.tv_sec - t0.tv_sec) + \
    ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}

#endif
