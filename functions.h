#include <assert.h>
#include <complex.h>
#include <cuComplex.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "macros.h"


double flops(double t) {
  // Tera or Giga FLOP/s
  // :lscpu: 6 cores, 2x32K L1 cache, 15MB L3 cache
  // Quadro GV100: peak 7.4 TFLOPS, bandwidth 870 GB/s
  //  cores: 5120, tensor cores 640, memory: 32 GB
  // max size: 49 152
  // printf("fpp %i, t %0.4f, N*N %0.4f\n", FLOPS_PER_POINT, t, N*N * 1e-9);
  return 1e-12 * N * N * (double) FLOPS_PER_POINT / t;
}

void check(double complex  z) {
  double a = creal(z), b = cimag(z);
  if (isnan(a)) printf("found nan re\n");
  if (isinf(a)) printf("found inf re\n");
  if (isnan(b)) printf("found nan I\n");
  if (isinf(b)) printf("found inf I\n");
  if (isinf(a)) exit(1);
}

double memory_in_MB() {
  // Return lower-bound of memory use
  // complex arrays x,y \in C^N
  // real (double precision) arrays u,v \in C^(DIMS * N)
  unsigned int bytes = 2 * N * sizeof(WTYPE) + 2 * DIMS * N * sizeof(STYPE);
  return bytes * 1e-6;
}

void summarize_c(char name, WTYPE *x, size_t len) {
  double max_amp = 0, min_amp = DBL_MAX, max_phase = 0;
  for (size_t i = 0; i < len; ++i) {
    max_amp = fmax(max_amp, cabs(x[i]));
    min_amp = fmin(min_amp, cabs(x[i]));
    max_phase = fmax(max_phase , carg(x[i]));
  }
  printf("%c) amp: [%0.3f - %0.6f], max phase: %0.3f\n", name, min_amp, max_amp, max_phase);
}

void normalize_amp(WTYPE *x, size_t len, char log_normalize) {
  double max_amp = 0;
  for (size_t i = 0; i < len; ++i)
    max_amp = fmax(max_amp, cabs(x[i]));

  if (max_amp < 1e-6)
    printf("WARNING, max_amp << 1\n");

  if (max_amp > 1e-6)
    for (size_t i = 0; i < len; ++i)
      x[i] /= max_amp;

  if (log_normalize)
    for (size_t i = 0; i < len; ++i)
      if (cabs(x[i]) > 0)
        x[i] = -clog(x[i]);
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
  check(x);
  if (cimag(x) >= 0) {
    fprintf(out, "%f+%fj", creal(x), cimag(x));
  } else {
    fprintf(out, "%f%fj", creal(x), cimag(x));
  }
}

void write_array(char c, STYPE *x, size_t len, FILE *out) {
  unsigned int i = 0;
  // key
  fprintf(out, "%c:", c);
  // first value
  fprintf(out, "%f", x[0]);
  // other values, prefixed by ','
  for (i = 1; i < len; ++i) {
    fprintf(out, ",%f", x[i]);
  }
  // newline / return
  fprintf(out, "\n");
}

void write_carray(char c, WTYPE *x, size_t len, FILE *out) {
  unsigned int i = 0;
  // key
  fprintf(out, "%c:", c);
  // first value
  print_c(x[0], out);
  // other values, prefixed by ','
  for (i = 1; i < len; ++i) {
    fprintf(out, ",");
    print_c(x[i], out);
  }
  // newline / return
  fprintf(out, "\n");
}
