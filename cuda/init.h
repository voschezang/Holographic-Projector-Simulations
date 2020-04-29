#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "macros.h"

void init_random(curandGenerator_t *gen, unsigned int seed) {
  curandCreateGenerator(gen, CURAND_RNG_PSEUDO_XORWOW);
  /* curandCreateGenerator(gen, CURAND_RNG_PSEUDO_MT19937); */
  curandSetPseudoRandomGeneratorSeed(*gen, seed);
}

void gen_random(float *x, float *d_x, size_t len, curandGenerator_t gen) {
  // TODO do this on cpu or at runtime (per batch)
  curandGenerateUniform(gen, d_x, len);
  cudaMemcpy(x, d_x, len * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

template<bool randomize>
void init_plane(std::vector<STYPE> &v, const double width, const STYPE z_offset) {
  const size_t n_sqrt = round(sqrt(v.size() / DIMS));
  const double dS = width * SCALE / (double) n_sqrt; // actually dS^(1/DIMS)
  const double offset = 0.5 * width;
  const size_t n_random = 2 * n_sqrt;
  const double margin = 0.;
  const double random_range = dS - 0.5 * margin;
  curandGenerator_t generator;
  float *d_random, random[n_random];

  if (randomize) {
    init_random(&generator, 11235);
    cu( cudaMalloc( (void **) &d_random, n_random * sizeof(float) ) );
  }

  for (unsigned int i = 0; i < n_sqrt; ++i) {
    if (randomize)
      gen_random(random, d_random, n_random, generator);

    for (unsigned int j = 0; j < n_sqrt; ++j) {
      v[Ix(i,j,0)] = i * dS - offset;
      v[Ix(i,j,1)] = j * dS - offset;
      v[Ix(i,j,2)] = z_offset;

      if (randomize) {
        v[Ix(i,j,0)] += random_range * (random[j*2] - 0.5);
        v[Ix(i,j,1)] += random_range * (random[j*2+1] - 0.5);
      }
    }
  }

  if (randomize) {
    curandDestroyGenerator(generator);
    cu( cudaFree( d_random ) );
  }
}

void init_planes(std::vector<STYPE> &u, std::vector<STYPE> &v, std::vector<STYPE> &w) {
  // each plane x,u y,v z,w is a set of points in 3d space
  const double width = 0.0005; // m
  /* const double dS = width * SCALE / (double) N_sqrt; // actually dS^(1/DIMS) */
  /* const double offset = 0.5 * width; */

  const auto
    du = 1.0 / (double) u.size(), // TODO use SCALE?
    half_width = width / 2.0;
  for (unsigned int i = 0; i < u.size(); i+=DIMS) {
    u[i] = i * du - half_width;
    u[i+1] = 0;
    u[i+2] = 0;
  }

  init_plane<RANDOM_Y_SPACE>(v, width, -0.02);
  init_plane<RANDOM_Z_SPACE>(w, width, 0.0);
}
