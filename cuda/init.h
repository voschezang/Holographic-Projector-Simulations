#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "macros.h"

void init_random(curandGenerator_t *gen, unsigned int seed) {
  curandCreateGenerator(gen, CURAND_RNG_PSEUDO_XORWOW);
  curandCreateGenerator(gen ,CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(*gen, seed);
}

void gen_random(float *x, float *d_x, size_t len, curandGenerator_t gen) {
  curandGenerateUniform(gen, d_x, len);
  cudaMemcpy(x, d_x, len * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

void init_planes(WTYPE *x, STYPE *u, STYPE *v, STYPE *w) {
  // each plane x,u y,v z,w is a set of points in 3d space
  const double width = 0.0005; // m
  const double dS = width * SCALE / (double) N_sqrt; // actually dS^(1/DIMS)
  const double offset = 0.5 * width;
#if defined(RANDOM_X_SPACE) || defined(RANDOM_Y_SPACE) || defined(RANDOM_Z_SPACE)
  const double margin = 0.;
  const double random_range = dS - 0.5 * margin;
  curandGenerator_t generator;
  init_random(&generator, 11235);
#endif
  printf("Domain: X : %f x %f, dS: %f\n", width, width, dS);
  float *d_random, random[4 * N_sqrt];
  cu( cudaMalloc( (void **) &d_random, 4 * N_sqrt * sizeof(float) ) );
  for(unsigned int i = 0; i < N_sqrt; ++i) {
    gen_random(random, d_random, 4 * N_sqrt, generator);
    for(unsigned int j = 0; j < N_sqrt; ++j) {
      size_t idx = i * N_sqrt + j;
      x[idx].x = 0; // real part
      x[idx].y = 0; // imag part
      if (i == N_sqrt * 1/2 && j == N_sqrt / 2) x[idx].x = 1;
      // if (i == N_sqrt * 1/3 && j == N_sqrt / 2) x[idx] = 1;
      // if (i == N_sqrt * 2/3 && j == N_sqrt / 2) x[idx] = 1;
      // if (i == N_sqrt * 1/4 && j == N_sqrt / 4) x[idx] = 1;
      // if (i == N_sqrt * 3/4 && j == N_sqrt / 4) x[idx] = 1;

      u[Ix(i,j,0)] = i * dS - offset;
      u[Ix(i,j,1)] = j * dS - offset;
      u[Ix(i,j,2)] = 0;

#ifdef RANDOM_Y_SPACE
      v[Ix(i,j,0)] = i * dS - offset + random_range * \
        (random[j*4+0] - 0.5);
      v[Ix(i,j,1)] = j * dS - offset + random_range * \
        (random[j*4+1] - 0.5);
      if (i == 2 && j == 2) {
        printf("rand: %f, %f; %f, %f\n", v[Ix(i,j,0)], v[Ix(i-1,j,0)], v[Ix(i,j-1,0)], v[Ix(i-1,j-1,0)]);
      }
#else
      v[Ix(i,j,0)] = i * dS - offset;
      v[Ix(i,j,1)] = j * dS - offset;
#endif
      v[Ix(i,j,2)] = -0.02;

#ifdef RANDOM_Z_SPACE
      w[Ix(i,j,0)] = i * dS - offset + random_range * \
        (random[j*4+2] - 0.5);
      w[Ix(i,j,1)] = j * dS - offset + random_range * \
        (random[j*4+3] - 0.5);
#else
      w[Ix(i,j,0)] = i * dS - offset;
      w[Ix(i,j,1)] = j * dS - offset;
#endif
      w[Ix(i,j,2)] = 0;
    }
  }

  curandDestroyGenerator(generator);
  cu( cudaFree( d_random ) );
}
