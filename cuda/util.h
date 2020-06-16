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
#include <iterator>     // std::ostream_iterator
/* #include <type_traits> */

#include "macros.h"
#include "algebra.h"
#include "kernel.cu"

/* enum class FileType {TXT, DAT, GRID}; */ // TODO rm unused file io funcs
enum class Shape {Line, Cross, Circle, DottedCircle};
enum class Variable {Offset, Width};
enum class Transformation {Full, Amplitude}; // Full: keep phase+amp, Amplitude: rm phase

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
  /* double offset[DIMS]; // TODO */
  double z_offset;
  bool randomize;
  bool hd; // TODO
  /* size_t x, y; // effective width, height, upperbounded by .size() */
};

struct Params {
  // Simulation parameters, used to init plane distributions
  Plane input; // ground truth
  /* std::vector<Plane> inputs; // ground truth */
  Plane projector;
  std::vector<Plane> projections; // approximation of input
  /* Geometry g; */
};

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

double bandwidth(double runtime, size_t n, size_t m, bool include_tmp) {
  // input phasor + input space + output space
  // n,m : number of input, output datapoints
  const double unit = 1e-6; // MB/s
  double input = n * (sizeof(WTYPE) + 3 * sizeof(STYPE));
  double output = m * 3 * sizeof(STYPE);
  if (include_tmp) {
    double tmp = GRIDDIM * SHARED_MEMORY_SIZE(BLOCKDIM) * sizeof(WTYPE);
    return unit * (input + output + tmp) / runtime;
  }
  return unit * (input + output) / runtime;
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

double memory_in_MB(size_t n) {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = n * sizeof(WTYPE) + DIMS * n * sizeof(STYPE);
  return bytes * 1e-6;
}

void print_info(Geometry p, size_t Nx, size_t Ny, size_t Nz) {
  printf("\nHyperparams:");
  printf("\n CUDA geometry: <<<%i,%i>>>", p.gridSize, p.blockSize);
  printf("\t(%.3fk threads)", p.gridSize * p.blockSize * 1e-3);

  printf("\n Input size (datapoints): x: %i, y: %i, z: %i", Nx, Ny, Nz);
  printf("\n E[N_x / thread]: %.4fk", Nx / (double) p.gridSize * p.blockSize * 1e-3);
  printf("\tE[N_y / thread]: %.4fk", Ny / (double) p.gridSize * p.blockSize * 1e-3);

  printf("\n n streams: %4i", p.n_streams);
  printf("\tbatch size: \t%6i", p.stream_size);
  printf("\tkernel size: \t%4i", p.kernel_size);

  printf("\n"); printf("Memory lb: %0.2f MB\n", memory_in_MB(Ny));
  {
    size_t n = SHARED_MEMORY_SIZE(p.blockSize);
    double m = n * sizeof(WTYPE) * 1e-3;
    printf("Shared data (per block) (tmp): %i , i.e. %0.3f kB\n", n, m);
  }
}

void print_result(std::vector<double> dt, size_t n = 1, size_t m = 1) {
  // n,m : number of input, output datapoints per transformation
  const double mu = mean(dt);
  printf("TFLOPS:   \t%0.5f \t (%i FLOP_PER_POINT)\n",  \
         flops(mu, n, m), FLOP_PER_POINT);
  // TODO correct bandwidth datasize
  // TODO multiply dt.size() with n,m inside func instead of outside?
  printf("Bandwidth: \t%0.5f Mb/s (excl. shared memory)\n", bandwidth(mu, n, m, false));
  printf("Bandwidth: \t%0.5f MB/s (incl. shared memory)\n", bandwidth(mu, n, m, true));
  if (dt.size() > 1) {
    double var = variance(dt);
    printf("Var[dt]: %e, Var[dt]/E[dt]: %e\n", var, var / mu);
  }
}

bool abs_of_is_positive(WTYPE x) {
  return cuCabs(x) > 0;
}

void summarize_c(char name, std::vector<WTYPE> &x) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0, sum = 0;
  /* for (const auto& x : X) { */
  for (size_t i = 0; i < x.size(); ++i) {
    max_amp = fmax(max_amp, cuCabs(x[i]));
    min_amp = fmin(min_amp, cuCabs(x[i]));
    max_phase = fmax(max_phase, angle(x[i]));
    sum += cuCabs(x[i]);
  }
  double mean = sum / (double) x.size();
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f, mean: %f\n", name, min_amp, max_amp, max_phase, mean);
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

template<typename T = double>
std::vector<T> read_bytes(std::string fn) {
  /* std::istream_iterator<unsigned char> in_iterator (out, ""); */
  /* const unsigned char *a = (unsigned char *) &x[0]; */
  /* std::copy( a, &a[sizeof(double) * x.size()], out_iterator ); */

  /* auto f = std::ifstream(fn, std::ios::in | std::ios::binary); */
  /* std::vector<uint8_t> data ((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<T>()); */
  /* ifstream myfile ("example.txt"); */
  std::streampos size;
  char * memblock;

  // init stream at end of stream to obtain size
  std::ifstream file (fn, std::ios::in|std::ios::binary|std::ios::ate);
  if (!file.is_open()) exit(1);;
  size = file.tellg();
  if (size == 0) exit(2);
  /* printf("sizes: %u, %i; %i %u\n", size, (int) size, sizeof(double)); */
  /* printf("sizes: %i / %i\n", (int) size, sizeof(double)); */
  memblock = new char [size];
  /* printf("memblock: %i\n", sizeof(memblock) / sizeof(char)); */
  file.seekg (0, std::ios::beg);
  file.read (memblock, size);
  file.close();
  std::cout << "the entire file content is in memory\n";

  T *ptr = (T*) memblock;
  printf("data: %f, %f \n", ptr[0], ptr[1]);

  // TODO use proper memory handling
  /* delete[] memblock; */
  /* return unique_ptr<T>  */
  return std::vector<T>(ptr, ptr + size / sizeof(T));
}
/* printf("len: %i\n", *len); */
/* for(auto i: contents) { */
/*   int value = i; */
/*   std::cout << "data: " << value << std::endl; */
/* } */

/* std::cout << "file size: " << contents.size() << std::endl; */

/* struct {size_t len; std::string amp, pos; } read_meta(std::string fn) { */
/*   size_t len; */
/*   std::string amp, pos; */

/**
 * (With lazy evaluation) Map/apply function `f` to each array element and write the result to file `out`.
 */
template<typename T = WTYPE>
void map_to_and_write_array(std::vector<T> &x, double (*f)(T), const char sep, std::ofstream& out) {
  out << f(x[0]);
  for (size_t i = 1; i < x.size(); ++i)
    out << sep << f(x[i]);
}

template<typename T = WTYPE>
void map_to_and_write_bytes(std::vector<T> &x, double (*f)(T), std::ofstream& out) {
  const unsigned int buffer_size = x.size() > 128 ? 8 : 1;
  double buffer[buffer_size];
  if (x.size() > 128)
    assert(x.size() == (x.size() / buffer_size) * buffer_size);

  const auto bytes = (unsigned char *) &buffer;
  auto iter = std::ostream_iterator<unsigned char>(out, "");

  for (size_t i = 0; i < x.size(); i+=buffer_size) {
    // split inner loop to allow vectorization
    size_t j;
    for ( j = 0; j < buffer_size; ++j) {
      buffer[j] = f(x[i+j]);
    }
    std::copy(bytes, &bytes[buffer_size * sizeof(double)], iter);
  }
}

template<typename T = double>
void write_array(std::vector<T> &x, const char sep, std::ofstream& out) {
  // TODO optimize
  out << x[0];
  for (size_t i = 1; i < x.size(); ++i)
    out << sep << x[i];
}

template<typename T = double>
void write_bytes(std::vector<T> &x, std::ofstream& out) {
  std::ostream_iterator<unsigned char> out_iterator (out, "");
  const unsigned char *a = (unsigned char *) &x[0];
  std::copy( a, &a[sizeof(double) * x.size()], out_iterator );
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

  // TODO replace phasor by amp and phase
  out << "{" \
      << quote("phasor")    << sep1 << quote(phasor) << sep2 \
      << quote("pos")       << sep1 << quote(pos)    << sep2 \
      << quote("z_offset")  << sep1 << p.z_offset    << sep2 \
      << quote("len")       << sep1 << len           << sep2 \
      << quote("precision") << sep1 << IO_PRECISION  << sep2 \
      << quote("dims")      << sep1 << DIMS          << sep2 \
      << quote("hd")        << sep1 << p.hd          << "}\n";
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

void write_arrays(std::vector<WTYPE> &x, std::vector<STYPE> &u,
                  std::string k1, std::string k2, Plane p) {
  static bool overwrite = true;
  if (overwrite)
    print("Save results as txt");

  auto dir = std::string{"../tmp/"};
  auto mode = overwrite ? std::ofstream::binary : std::ofstream::app;
  std::ofstream out;

  out.open(dir + "out.json", mode);
  write_metadata(k1, k2, p, x.size(), out);
  out.close();

  /* out << std::scientific; // to allow e.g. 1.0e-50 */
  out << std::fixed;
  out << std::setprecision(IO_PRECISION);

  out.open(dir + k1 + "_amp.dat", std::ofstream::binary);
  /* map_to_and_write_array(x, cuCabs, ',', out); */
  map_to_and_write_bytes(x, cuCabs, out);
  out.close();

  out.open(dir + k1 + "_phase.dat", std::ofstream::binary);
  /* map_to_and_write_array(x, angle, ',', out); */
  map_to_and_write_bytes(x, angle, out);
  out.close();

  out.open(dir + k2 + ".dat", std::ofstream::binary);
  /* write_array(u, ',', out); */
  write_bytes(u, out);
  out.close();
  overwrite = false;
}

double diff(struct timespec t0, struct timespec t1) {
  return (double) (t1.tv_sec - t0.tv_sec) + \
    ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}

#endif
