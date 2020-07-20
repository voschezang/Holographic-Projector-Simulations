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

void derive_secondary_geometry(const size_t n, Geometry& p) {
  p.stream_size = n / (p.n_streams * p.batch_size * p.kernel_size);

  assert(p.blockSize   > 0); assert(p.gridDim   > 0);
  assert(p.kernel_size > 0); assert(p.batch_size > 0);
  assert(p.stream_size > 0); assert(p.n_streams  > 0);
  assert(n == p.kernel_size * p.batch_size * p.stream_size * p.n_streams);

  p.n_batches = p.n_streams * p.stream_size;
  p.n_kernels = p.n_batches * p.batch_size;

  p.n_per_stream = n / p.n_streams;
  p.n_per_batch = p.n_per_stream / p.stream_size;
  p.n_per_kernel = p.kernel_size;
  p.n_per_block = p.n_per_kernel / (double) p.gridDim;
  p.n_per_thread = p.n_per_block / (double) p.blockSize;

  p.kernels_per_stream = p.stream_size * p.batch_size;

  assert(n == p.n_per_kernel * p.n_kernels);
  assert(n == p.n_per_batch * p.n_batches);
  assert(p.n_batches >= 1);
  assert(p.n_kernels >= 1);
  assert(p.n_per_stream >= 1);
  assert(p.n_per_batch >= 1);
  assert(p.n_per_kernel >= 1);
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
  p.gridDim = 1;
  p.kernel_size = 1;
  p.batch_size = n;
  p.n_streams = 1;
  derive_secondary_geometry(n, p);
  return p;
}

Geometry geometry(const size_t n) {
  Geometry p;
  p.blockSize = BLOCKDIM;
  assert(p.blockSize == 64);
  p.gridDim = GRIDDIM; // TODO rename to gridDim and use dim3 dtype
  p.kernel_size = KERNEL_SIZE;
  p.batch_size = BATCH_SIZE;
  p.n_streams = N_STREAMS;

  derive_secondary_geometry(n, p);
  return p; // copy on return is permissible
}

/**
 * Distribute sampling points over a 2D plane in 3D space.
 */
void plane(std::vector<SPACE> &v, const Plane p) {
  // TODO return ptr to device memory, copy pos data to CPU during batches
  static unsigned int seed = 1234; // TODO manage externally from this function
  const size_t n = v.size() / DIMS;
  printf("offset: %f\n", p.offset.z);
  for (unsigned int i = 0; i < n; ++i)
    v[i*DIMS + 2] = p.offset.z;
  /*
   * Assume HD dimensions, keep remaining pixels for kernel geometry compatibility
   * solve for x:  `n * (ratio) = x * y * (x/y) = x^2   ->   x = sqrt(n * ratio)`
   */
  auto
    x = (size_t) sqrt(n * p.aspect_ratio),
    y = n / x;
  if (p.aspect_ratio != 1.0) {
    printf("Screen dimensions: %i x %i\t", x, y);
    printf("Remaining points: %i/%i\n", n - x*y, n);
  }
  assert(x * y <= n);

  // distribute points over a lattice (each point in the center of a cell in a grid)
  const double
    height = p.width / p.aspect_ratio,
    dx = p.width / (double) x, // excl. boundary; don't divide by (x-1)
    dy = height / (double) y,
    x_half = (p.width - dx) / 2., // subtract dx/2 to position points in center
    y_half = (height - dy) / 2.;

  printf("init::plane, width: %e, n_x: %lu, dx: %e, x_half: %e\n", p.width, x, dx, x_half);
  // properties to generate random offsets
  const double
    rel_margin = p.randomize ? 0.01 : 0.0,
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
      v[Ix(i,j,0,y)] = i * dx - x_half + p.offset.x;
      v[Ix(i,j,1,y)] = j * dy - y_half + p.offset.y;
      v[Ix(i,j,2,y)] = p.offset.z;

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

  // fill rest of array with semi-realistic values to minimize incompatibility issues
  for (unsigned int i = x * y; i < n; ++i)
    for (unsigned int j = 0; j < DIMS; ++j)
      v[i * DIMS + j] = v[j];
}

std::vector<SPACE> sparse_plane(std::vector<SPACE> &u, Shape shape, double width,
                                const Cartesian<double> &offset, double modulate = 0.) {
  const size_t n = u.size() / DIMS;
  for (unsigned int i = 0; i < n; ++i)
    u[i*DIMS + 2] = offset.z;

  assert(n != 0);
  if (n == 1) {
    u[0] = offset.x;
    u[1] = offset.y;
    return u;
  }

  // set the x,y dimensions
  switch (shape) {
  case Shape::Line: {
    // Distribute datapoints over a line
    auto
      du = width / (double) (n-1), // TODO use SCALE?
      half_width = width / 2.0;

    for (unsigned int i = 0; i < n; ++i)
      u[i*DIMS] = i * du - half_width;

    break;
  }
  case Shape::Cross: {
    const size_t half_n = n / 2;
    const auto
      du = width * width / (double) (n-1), // TODO use SCALE?
      half_width = width / 2.0;

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
      radius = width / 2.0,
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
      radius = width / 2.0,
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

  if (offset.x != 0 || offset.y != 0) {
    for (unsigned int i = 0; i < n; ++i) {
      u[i * DIMS + 0] += offset.x;
      u[i * DIMS + 1] += offset.y;
    }
  }

  return u;
}

template<typename T>
std::vector<T*> malloc_vectors(T **d_ptr, size_t dim1, size_t dim2) {
   // Return a host vector of pointers
   assert(dim1 >= 1); assert(dim2 >= 1);
   cu( cudaMalloc( d_ptr, dim1 * dim2 * sizeof(T) ) );
   auto vec = std::vector<T*>(dim1);
   for (size_t i = 0; i < dim1; ++i)
     vec[i] = *d_ptr + i * dim2;
   return vec;
 }

 template<typename T>
 std::vector<DeviceVector<T>> malloc_matrix(T **d_ptr, size_t dim1, size_t dim2) {
   // Return a host vector of DeviceVector elements
   assert(dim1 >= 1); assert(dim2 >= 1);
   cu( cudaMalloc( d_ptr, dim1 * dim2 * sizeof(T) ) );
   auto matrix = std::vector<DeviceVector<T>>(dim1);
   // std::vector<T>(*d_ptr + a, *d_ptr + b); has weird side effects
   // note that *ptr+i == &ptr[i], but that ptr[i] cannot be read by host if it's located on device
   for (size_t i = 0; i < dim1; ++i)
     matrix[i] = DeviceVector<T>{data: *d_ptr + i * dim2,
                                 size: dim2};
   return matrix;
 }

template<typename T>
std::vector<T*> pinned_malloc_vectors(T **d_ptr, size_t dim1, size_t dim2) {
  // Return a host vector of pointers
  assert(dim1 >= 1); assert(dim2 >= 1);
  // TODO consider cudaHostAllocMapped(..., cudaHostAllocMapped)
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto vec = std::vector<T*>(dim1);
  // for (auto&& row : matrix)
  for (size_t i = 0; i < dim1; ++i)
    vec[i] = *d_ptr + i * dim2;
  return vec;
}

template<typename T>
std::vector<DeviceVector<T>> pinned_malloc_matrix(T **d_ptr, size_t dim1, size_t dim2) {
  // Return a host vector of DeviceVector elements
  assert(dim1 >= 1); assert(dim2 >= 1);
  cu( cudaMallocHost( d_ptr, dim1 * dim2 * sizeof(T) ) );
  auto matrix = std::vector<DeviceVector<T>>(dim1);
  // std::vector<T>(*d_ptr + a, *d_ptr + b); has weird side effects
  // note that *ptr+i == &ptr[i], but that ptr[i] cannot be read by host if it's located on device
  for (size_t i = 0; i < dim1; ++i)
    matrix[i] = DeviceVector<T>{data: *d_ptr + i * dim2,
                                size: dim2};
  return matrix;
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
#endif
