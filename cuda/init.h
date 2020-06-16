#ifndef INIT
#define INIT

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
/* #include <gsl/gsl_linalg.h> */

#include "macros.h"
#include "hyper_params.h"
#include "algebra.h"
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

void randomize(float *x, float *d_x, size_t len, curandGenerator_t gen) {
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

Params params(const Variable var, const size_t n_z_planes, const bool hd, double rel_object_width) {
  // TODO allow multiple x planes
  const double
    z_offset = 0.4,
    width = PROJECTOR_WIDTH,
    object_width = rel_object_width * width; // max width of virtual object that is projected
  /* const bool randomize = true; */
  const bool randomize = false;
  auto projections = std::vector<Plane>{};

  /* const double width = 300 * 7e-6; // = 1920 x 7e-6 */
  /* const double width = 5e-4; */

  // Note that the projection params slightly differ from the projector params
  for (auto& i : range(n_z_planes))
    projections.push_back({name: 'z', width: object_width * 2, z_offset: 0.0, randomize: randomize, hd: false});

  if (n_z_planes > 1) {
    if (var == Variable::Offset) {
      const double delta = -0.5 * z_offset / (n_z_planes - 1.0);
      auto values = linspace(n_z_planes, 0.0, delta * n_z_planes);
      for (auto& i : range(n_z_planes))
        projections[i].z_offset = values[i];
    }
    else if (var == Variable::Width) {
      if (n_z_planes <= 5) {
        auto values = linspace(n_z_planes, object_width * 1.5, width * pow(n_z_planes - 1, 0.5));
        for (auto& i : range(n_z_planes))
          projections[i].width = values[i];
      } else {
        auto values = logspace(n_z_planes, -1, -4.5);
        for (auto& i : range(n_z_planes))
          projections[i].width = values[i];
      }
    }
  }

  // TODO use width var for x width
  return
    {   input       : {name: 'x', width: object_width, z_offset: 0.0,
          randomize: randomize, hd: false},
        projector   : {name: 'y', width: width, z_offset : z_offset,
          randomize : randomize, hd: hd},
        projections : projections};
}

void derive_secondary_geometry(const size_t n, Geometry &p) {
  p.stream_size = n / (p.n_streams * p.batch_size * p.kernel_size);

  assert(p.blockSize   > 0); assert(p.gridSize   > 0);
  assert(p.kernel_size > 0); assert(p.batch_size > 0);
  assert(p.stream_size > 0); assert(p.n_streams  > 0);
  assert(n == p.kernel_size * p.batch_size * p.stream_size * p.n_streams);

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
}

Geometry simple_geometry(const size_t n) {
  Geometry p;
  p.blockSize = 1;
  p.gridSize = 1;
  p.kernel_size = 1;
  p.batch_size = n;
  p.n_streams = 1;
  derive_secondary_geometry(n, p);
  return p;
}

Geometry geometry(const size_t n) {
  Geometry p;
  p.blockSize = BLOCKDIM;
  p.gridSize = GRIDDIM;
  p.kernel_size = KERNEL_SIZE;
  p.batch_size = BATCH_SIZE;
  p.n_streams = N_STREAMS;

  derive_secondary_geometry(n, p);
  return p; // copy on return is permissible
}

/**
 * Distribute sampling points over a 2D plane in 3D space.
 */
std::vector<STYPE> plane(size_t n, Plane p, double x_offset = 0, double y_offset = 0) {
  // TODO return ptr to device memory, copy pos data to CPU during batches
  static unsigned int seed = 1234; // TODO manage externally from this function
  auto v = std::vector<STYPE>(n * DIMS);
  /* const size_t n_sqrt = round(sqrt(n)); */
  double ratio = 1.;
  size_t
    x = round(sqrt(n)),
    y = x;
  assert(x * y == n);
  assert(x > 0 && y > 0);

  if (p.hd) {
    // assume HD dimensions, keep remaining pixels for kernel geometry compatibility
    ratio = 1920. / 1080.; // i.e. x / y
    // solve for x: n * ratio = x * y * (x/y) = x^2
    x = sqrt(n * ratio);
    y = n / x;
    printf("Screen dimensions: %i x %i\t", x, y);
    printf("Remaining points: %i/%i\n", n - x*y, n);
    assert(x * y <= n);
  }

  const double
    dx = p.width * SCALE / (double) x,
    dy = p.width * SCALE / ((double) y * ratio),
    x_half = 0.5 * p.width,
    y_half = 0.5 * p.width / ratio,
    rel_margin = p.randomize ? 0.05 : 0.0,
    x_margin = rel_margin * dx, // TODO min space between projector pixels
    y_margin = rel_margin * dy,
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
      randomize(random, d_random, n_random, generator);

    for (unsigned int j = 0; j < y; ++j) {
      v[Ix(i,j,0,y)] = i * dx - x_half;
      v[Ix(i,j,1,y)] = j * dy - y_half;
      v[Ix(i,j,2,y)] = p.z_offset;

      if (p.randomize) {
        v[Ix(i,j,0,y)] += x_random_range * (random[j*2] - 0.5);
        v[Ix(i,j,1,y)] += y_random_range * (random[j*2+1] - 0.5);
      }
    }
  }
  if (p.randomize) {
    curandDestroyGenerator(generator);
    cu( cudaFree( d_random ) );
  }

  // fill rest of array to minimize incompatibility issues
  for (unsigned int i = x * y; i < n; ++i)
    for (unsigned int j = 0; j < DIMS; ++j)
      v[i * DIMS + j] = v[j];

  if (x_offset != 0 || y_offset != 0)
    for (unsigned int i = 0; i < n; ++i) {
      v[i * DIMS + 0] += x_offset;
      v[i * DIMS + 1] += y_offset;
    }

  return v;
}

std::vector<STYPE> sparse_plane(size_t n, Shape shape, double object_width,  double x_offset, double y_offset, double modulate = 0.) {
  // each plane x,u y,v z,w is a set of points in 3d space
  /* const double dS = width * SCALE / (double) N_sqrt; // actually dS^(1/DIMS) */
  /* const double offset = 0.5 * width; */
  auto u = std::vector<STYPE>(n * DIMS, 0.0);
  assert(n != 0);
  if (n == 1) {
    u[0] = x_offset;
    u[1] = y_offset;
    return u;
  }

  switch (shape) {
  case Shape::Line: {
    // Distribute datapoints over a line
    auto
      du = object_width / (double) (n-1), // TODO use SCALE?
      half_width = object_width / 2.0;

    for (unsigned int i = 0; i < n; ++i)
      u[i*DIMS] = i * du - half_width;

    break;
  }
  case Shape::Cross: {
    const size_t half_n = n / 2;
    const auto
      du = object_width * object_width / (double) (n-1), // TODO use SCALE?
      half_width = object_width / 2.0;

    // if n is even, skip the center
    for (unsigned int i = 0; i < n / 2; ++i) {
      u[i * DIMS] = i * du - half_width;
      u[(i + half_n/2) * DIMS] = i * du - half_width;
    }
    break;
  }
  case Shape::Circle: {
    // Distribute datapoints over a circle
    // (using polar coordinates, s.t. $phi_i \equiv i \Delta \phi$)
    auto
      radius = object_width / 2.0,
      /* circumference = TWO_PI * pow(radius, 2), */
      d_phase = TWO_PI / (double) n,
      arbitrary_offset = 0.1125 + modulate * TWO_PI;

    for (unsigned int i = 0; i < n; ++i) {
      u[i * DIMS] = sin(i * d_phase + arbitrary_offset) * radius;
      u[i * DIMS + 1] = cos(i * d_phase + arbitrary_offset) * radius;
    }
    break;
  }
  case Shape::DottedCircle: {
    // (using polar coordinates)
    auto
      radius = object_width / 2.0,
      /* circumference = TWO_PI * pow(radius, 2), */
      /* don't include end; divide by n */
      d_phase = TWO_PI / (double) (n-1), // subtract center point
      arbitrary_offset = 0.1125 + modulate * TWO_PI;

    // TODO randomize slightly?
    for (unsigned int i = 1; i < n; ++i) {
      u[i * DIMS] = sin(i * d_phase + arbitrary_offset) * radius;
      u[i * DIMS + 1] = cos(i * d_phase + arbitrary_offset) * radius;
    }

    break;
  }
  }

  if (x_offset != 0 || y_offset != 0) {
    for (unsigned int i = 0; i < n; ++i) {
      u[i * DIMS + 0] += x_offset;
      u[i * DIMS + 1] += y_offset;
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
