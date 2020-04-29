#ifndef UTIL
#define UTIL

#include <assert.h>
#include <stdio.h>
#include <time.h>
/* #include <type_traits> */

#include "macros.h"
#include "kernel.cu"

enum class FileType {TXT, DAT, GRID};

double flops(double runtime) {
  // Tera or Giga FLOP/s
  // :lscpu: 6 cores, 2x32K L1 cache, 15MB L3 cache
  // Quadro GV100: peak 7.4 TFLOPS (dp), 14.8 (sp), 59.3 (int)
  // bandwidth 870 GB/s
  //  cores: 5120, tensor cores 640, memory: 32 GB
  // generation Volta, compute capability 7.0
  // max size: 49 152
  // printf("fpp %i, t %0.4f, N*N %0.4f\n", FLOP_PER_POINT, t, N*N * 1e-9);
  return 1e-12 * N * N * (double) FLOP_PER_POINT / runtime;
}

double bandwidth(double runtime, const int n_planes, const char include_tmp) {
  // input phasor  + input space + output space
  const double unit = 1e-6; // MB/s
  double input = n_planes * N * (sizeof(WTYPE) + 2 * sizeof(STYPE));
  double output = n_planes * N * sizeof(WTYPE);
  if (!include_tmp)
    return unit * (input + output) / runtime;

  double tmp = GRIDDIM * SHARED_MEMORY_SIZE * sizeof(WTYPE);
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

void check_params() {
#if (N_STREAMS < 1)
  printf("Invalid param: N_STREAMS < 1\n"); assert(0);
#elif (BATCHES_PER_STREAM < 1)
  printf("Invalid param: BATCHES_PER_STREAM < 1\n"); assert(0);
#elif (N_STREAMS * BATCHES_PER_STREAM != N_BATCHES)
  printf("Invalid param: incompatible N_STREAMS and N\n"); assert(0);
#endif
  assert(STREAM_SIZE > 0);
  assert(N > 0); assert(N_STREAMS > 0); assert(STREAM_SIZE > 0);
  assert(N_BATCHES > 0); assert(BATCH_SIZE > 0);
  assert(REDUCE_SHARED_MEMORY <= BLOCKDIM);
  assert(KERNEL_BATCH_SIZE <= BATCH_SIZE);
  assert(KERNEL_BATCH_SIZE * KERNELS_PER_BATCH == BATCH_SIZE);
  assert(BLOCKDIM / REDUCE_SHARED_MEMORY);
  assert(N_PER_THREAD > 0);
  assert(N == N_STREAMS * STREAM_SIZE);
  assert(N == BATCH_SIZE * BATCHES_PER_STREAM * N_STREAMS);
  assert(N_PER_THREAD * BLOCKDIM * GRIDDIM == N);
}

double memory_in_MB() {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = 2 * N * sizeof(WTYPE) + 2 * DIMS * N * sizeof(STYPE);
  return bytes * 1e-6;
}

void summarize_c(char name, WTYPE *x, size_t len) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0, sum = 0;
  /* for (const auto& x : X) { */
  for (size_t i = 0; i < len; ++i) {
    max_amp = fmax(max_amp, cuCabs(x[i]));
    min_amp = fmin(min_amp, cuCabs(x[i]));
    max_phase = fmax(max_phase , angle(x[i]));
    sum += cuCabs(x[i]);
  }
  double mean = sum / (double) N;
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f, mean: %f\n", name, min_amp, max_amp, max_phase, mean);
}

void normalize_amp(WTYPE *x, size_t len, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < len; ++i)
    max_amp = fmax(max_amp, cuCabs(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < len; ++i) {
      x[i].x /= max_amp;
      x[i].y /= max_amp;
    }

  if (log_normalize)
    for (size_t i = 0; i < len; ++i) {
      if (x[i].x > 0) x[i].x = -log(x[i].x);
      if (x[i].y > 0) x[i].y = -log(x[i].y);
    }
}

void summarize_double(char name, double *x, size_t n) {
  double max = DBL_MIN, min = DBL_MAX;
  for (size_t i = 0; i < n; ++i) {
    max = fmax(max, x[i]);
    min = fmin(min , x[i]);
  }
  printf("%c)  range: [%0.3f , %0.3f]\n", name, min, max);
}

void print_c(WTYPE x, FILE *out) {
  // check(x); //TODO uncomment
  if (x.y >= 0) {
    fprintf(out, "%f+%fj", x.x, x.y);
  } else {
    fprintf(out, "%f%fj", x.x, x.y);
  }
}

void write_array(char c, std::vector<STYPE> &x, FILE *out) {
  // key
  fprintf(out, "%c:", c);

  // first value
  fprintf(out, "%e", x[0]);
  // other values, prefixed by ','
  for (size_t i = 1; i < x.size(); ++i) {
    fprintf(out, ",%e", x[i]);
  }
  // newline / return
  fprintf(out, "\n");
}

void write_carray(char c, std::vector<WTYPE> &x, FILE *out) {
  // key
  fprintf(out, "%c:", c);
  // first value
  print_c(x[0], out);
  // other values, prefixed by ','
  for (size_t i = 1; i < x.size(); ++i) {
    fprintf(out, ",");
    print_c(x[i], out);
  }
  // newline / return
  fprintf(out, "\n");
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
void write_arrays(std::vector<WTYPE> &x, std::vector<WTYPE> &y, std::vector<WTYPE> &z,
                  std::vector<STYPE> &u, std::vector<STYPE> &v, std::vector<STYPE> &w) {
  printf("Save results as ");
  // TODO use csv for i/o, read python generated x
  if (type != FileType::TXT) return;
  char fn[] = "../tmp/out.txt";
  remove(fn); // fails if file does not exist
  FILE *out = fopen(fn, "wb");
  write_carray('x', x, out); write_carray('y', y, out); write_carray('z', z, out);
  write_array ('u', u, out); write_array ('v', v, out); write_array ('w', w, out);
  fclose(out);
}

double dt(struct timespec t0, struct timespec t1) {
  return (double) (t1.tv_sec - t0.tv_sec) + \
    ((double) (t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}

#endif
