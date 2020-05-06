#ifndef INIT
#define INIT

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "macros.h"
#include "hyper_params.h"
#include "util.h"

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

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace init {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/* Plane plane_params(bool randomize, double width, double z_offset) { */
/*   Plane p; */
/*   p.width = width; */
/*   p.randomize = randomize; */
/*   p.z_offset = z_offset; */
/*   return p; */
/* } */

Params params() {
  const double width = 0.0005;
  const bool randomize = true;
  auto offsets = std::vector<double>{0, 0.001, 0.01};
  auto projections = std::vector<Plane>{};
  std::cout << "size " << projections.size() << '\n';

  for (auto& offset : offsets)
    projections.push_back({width: width, z_offset: offset, randomize: randomize});

  return
    {   input       : {width: width, z_offset : 0,     randomize : randomize},
        projector   : {width: width, z_offset : -0.02, randomize : randomize},
        projections : projections};
}

Geometry geometry (const size_t n) {
  Geometry p;
  p.blockSize = BLOCKDIM;
  p.gridSize = GRIDDIM;
  p.kernel_size = KERNEL_SIZE;
  p.batch_size = BATCH_SIZE;
  p.n_streams = N_STREAMS;
  p.stream_size = n / (p.n_streams * p.batch_size * p.kernel_size);

  assert(p.blockSize   > 0); assert(p.gridSize   > 0);
  assert(p.kernel_size > 0); assert(p.batch_size > 0);
  assert(p.stream_size > 0); assert(p.n_streams  > 0);
  assert(n == p.kernel_size * p.batch_size * p.stream_size * p.n_streams);

  // secondary

  p.n_batches = p.n_streams * p.stream_size;
  p.n_kernels = p.n_batches * p.batch_size;

  p.n_per_stream = n / p.n_streams;
  p.n_per_batch = p.n_per_stream / p.stream_size;
  p.n_per_kernel = p.kernel_size;
  p.n_per_block = p.n_per_kernel / (double) p.gridSize;
  p.n_per_thread = p.n_per_block / (double) p.blockSize;

  p.kernels_per_stream = p.stream_size * p.batch_size;

  assert(n == p.n_per_kernel * p.n_kernels);
  assert(n == p.n_per_batch * p.n_batches);
  assert(p.n_batches > 0);
  assert(p.n_kernels > 0);
  assert(p.n_per_stream > 0);
  assert(p.n_per_batch > 0);
  assert(p.n_per_kernel > 0);
  assert(p.n_per_block > 0.0);
  assert(p.n_per_thread > 0.0);
  assert(p.kernels_per_stream > 0);

  if (p.n_per_block < 1)
    print("Warning, not all _blocks_ are used");
  if (p.n_per_thread < 1)
    print("Warning, not all _threads_ are used");

  check_hyper_params(p);
  return p; // copy on return is permissible
}

std::vector<STYPE> plane(size_t n, Plane p) {
  auto v = std::vector<STYPE>(n * DIMS);
  const size_t n_sqrt = round(sqrt(v.size() / DIMS));
  const double dS = p.width * SCALE / (double) n_sqrt; // actually dS^(1/DIMS)
  const double offset = 0.5 * p.width;
  const size_t n_random = 2 * n_sqrt;
  const double margin = 0.;
  const double random_range = dS - 0.5 * margin;
  curandGenerator_t generator;
  float *d_random, random[n_random];

  if (p.randomize) {
    init_random(&generator, 11235);
    cu( cudaMalloc( (void **) &d_random, n_random * sizeof(float) ) );
  }

  for (unsigned int i = 0; i < n_sqrt; ++i) {
    if (p.randomize)
      gen_random(random, d_random, n_random, generator);

    for (unsigned int j = 0; j < n_sqrt; ++j) {
      v[Ix(i,j,0)] = i * dS - offset;
      v[Ix(i,j,1)] = j * dS - offset;
      v[Ix(i,j,2)] = p.z_offset;

      if (p.randomize) {
        v[Ix(i,j,0)] += random_range * (random[j*2] - 0.5);
        v[Ix(i,j,1)] += random_range * (random[j*2+1] - 0.5);
      }
    }
  }

  if (p.randomize) {
    curandDestroyGenerator(generator);
    cu( cudaFree( d_random ) );
  }
  return v;
}

std::vector<STYPE> sparse_plane(size_t n, const double width) {
  // each plane x,u y,v z,w is a set of points in 3d space
  /* const double dS = width * SCALE / (double) N_sqrt; // actually dS^(1/DIMS) */
  /* const double offset = 0.5 * width; */
  auto u = std::vector<STYPE>(n * DIMS);
  if (n > 1) {
    const double
      du = width / (double) n, // TODO use SCALE?
      half_width = width / 2.0;

    for (unsigned int i = 0; i < u.size(); i+=DIMS)
      u[i] = i * du - half_width;

  }
  return u;
}

template<typename T>
std::vector<T*> pinned_malloc_vector(T **d_ptr, size_t dim1, size_t dim2) {
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto vec = std::vector<T*>(dim1);
  // for (auto&& row : matrix)
  for (size_t i = 0; i < dim1; ++i)
    vec[i] = *d_ptr + i * dim2;
  return vec;
}

template<typename T>
std::vector<DeviceVector<T>> pinned_malloc_matrix(T **d_ptr, size_t dim1, size_t dim2) {
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto matrix = std::vector<DeviceVector<T>>(dim1);
  // std::vector<T>(*d_ptr + a, *d_ptr + b); has weird side effects
  // note that *ptr+i == &ptr[i], but that ptr[i] cannot be read
  for (size_t i = 0; i < dim1; ++i)
    matrix[i] = DeviceVector<T>{.data = *d_ptr + i * dim2, .size = dim2};
  return matrix;
}


///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
