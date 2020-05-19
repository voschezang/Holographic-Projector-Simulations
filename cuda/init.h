#ifndef INIT
#define INIT

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
/* #include <gsl/gsl_linalg.h> */

#include "macros.h"
#include "hyper_params.h"
#include "util.h"

/**
 * Initialization of params and vectors.
 * These functions are not (necessarily) optimized for performance.
 */

void init_random(curandGenerator_t *gen, unsigned int seed) {
  // TODO do this on CPU to avoid data copying
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

  Params params(Variable var, size_t n_planes) {
  // TODO allow multiple x planes
  const double z_offset = 0.0;
  const bool randomize = true;
  auto projections = std::vector<Plane>{};
  // Note that the projection params slightly differ from the projector params
  for (auto& i : range(n_planes))
    projections.push_back({name: 'z', width: 5e-4, z_offset: z_offset, randomize: randomize});

  // Setup the independent variable for the experiment
  if (n_planes > 1) {
    if (var == Variable::Offset) {
      const double delta = 0.01;
      auto values = linspace(n_planes, 0.0, delta * n_planes);
      for (auto& i : range(n_planes))
        projections[i].z_offset = values[i];
    }
    else if (var == Variable::Width) {
      // TODO logspace/geomspace
      auto values = linspace(n_planes, 1e-6, 1e-3);
      for (auto& i : range(n_planes))
        projections[i].width = values[i];
    }
  }
  /* switch (var) { */
  /* case Variable::Offset: */
  /*   const double delta = 0.01; */
  /*   auto values = linspace(n_planes, 0.0, delta * n_planes); */
  /*   for (auto& i : range(n_planes)) */
  /*     projections[i].z_offset = values[i]; */
  /*   break; */

  /* case Variable::Width: */
  /*   auto values = linspace(n_planes, 0.001, 0.1); */
  /*   for (auto& i : range(n_planes)) */
  /*     projections[i].width = values[i]; */
  /*   break; */
  /* } */

  return
    {   input       : {name: 'x', width: 1e-4, z_offset : 0.0,
          randomize : randomize},
        projector   : {name: 'y', width: 5e-4, z_offset : -0.02,
          randomize : randomize},
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

/**
 * Distribute sampling points over a 2D plane in 3D space.
 */
std::vector<STYPE> plane(size_t n, Plane p, bool hd) {
  static unsigned int seed = 1234; // TODO manage externally from this function
  auto v = std::vector<STYPE>(n * DIMS);
  /* const size_t n_sqrt = round(sqrt(n)); */
  double ratio = 1.;
  size_t
    x = round(sqrt(n)),
    y = x;
  assert(x * y == n);

  if (hd) {
    // assume HD dimensions
    ratio = 1920. / 1080.;
    x = sqrt(n * ratio), y = n / x;
    printf("Screen dimensions: %i x %i\n", x, y);
    assert(x * y <= n);
    // neglect remaining pixels
  }

  /* const double dS = p.width * SCALE / (double) n_sqrt; // actually dS^(1/DIMS) */
  /* // the corresponding height is applied implicitly */

  const double
    dx = p.width * SCALE / (double) x,
    dy = p.width * SCALE / (double) y * ratio,
    x_offset = 0.5 * p.width, // TODO rename => x_range
    y_offset = 0.5 * p.width * ratio,
    rel_margin = 0.01,
    x_margin = rel_margin * x_offset, // TODO min space between projector pixels
    y_margin = rel_margin * y_offset,
    x_random_range = dx - 0.5 * x_margin,
    y_random_range = dy - 0.5 * y_margin;

  const size_t n_random = x + y;
  curandGenerator_t generator;
  float *d_random, random[n_random];

  if (p.randomize) {
    init_random(&generator, seed++);
    cu( cudaMalloc( (void **) &d_random, n_random * sizeof(float) ) );
  }

  for (unsigned int i = 0; i < x; ++i) {
    if (p.randomize)
      gen_random(random, d_random, n_random, generator);

    for (unsigned int j = 0; j < y; ++j) {
      v[Ix(i,j,0,y)] = i * dx - y_offset;
      v[Ix(i,j,1,y)] = j * dy - x_offset;
      v[Ix(i,j,2,y)] = p.z_offset;
      /* if (i == 0 && j == 0) { */
      /*   std::cout << "v[" << Ix(i,j,0) << "]: " << v[Ix(i,j,0)] << '\n'; */
      /*   assert(abs(-1e-3) > 1e-30); */
      /* /\*   assert(v[0] > 1e-30); *\/ */
      /*   assert(abs(v[Ix(i,j,0)]) > 1e-30); */
      /* } */
      /* assert(abs(v[Ix(i,j,0)]) > 1e-30); */

      if (p.randomize) {
        v[Ix(i,j,0,y)] += x_random_range * (random[j*2] - 0.5);
        v[Ix(i,j,1,y)] += y_random_range * (random[j*2+1] - 0.5);
      }
    }
  }
  /* } else { */
  /*   // assume HD dimensions */
  /*   const auto x = 1920, y = 1080; */
  /*   assert(x * y == n); */
  /*   for (unsigned int i = 0; i < x; ++i) { */
  /*     for (unsigned int j = 0; j < y; ++j) { */

  /*     } */
  /*   } */
  /* } */

  if (p.randomize) {
    curandDestroyGenerator(generator);
    cu( cudaFree( d_random ) );
  }
  return v;
}

 std::vector<STYPE> sparse_plane(size_t n, Shape shape, double width) {
  // each plane x,u y,v z,w is a set of points in 3d space
  /* const double dS = width * SCALE / (double) N_sqrt; // actually dS^(1/DIMS) */
  /* const double offset = 0.5 * width; */
  auto u = std::vector<STYPE>(n * DIMS);
  assert(n != 0);
  if (n == 1) return u;

  switch (shape) {
  case Shape::Line: {
    // Distribute datapoints over a line
    auto
      du = width / (double) n, // TODO use SCALE?
      half_width = width / 2.0;

    for (unsigned int i = 0; i < u.size(); i+=DIMS)
      u[i] = i * du - half_width;
    break;
  }
  case Shape::Circle: {
    // Distribute datapoints over a circle
    // (using polar coordinates)
    auto
      radius = width / 2.0,
      /* circumference = TWO_PI * pow(radius, 2), */
      d_phase = TWO_PI / (double) n;

    for (unsigned int i = 0; i < u.size(); i+=DIMS) {
      u[i  ] = sin(i * d_phase) * radius;
      u[i+1] = cos(i * d_phase) * radius;
    }
    break;
  }
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
