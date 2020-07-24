// #define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h> // unused in this file but causes error if omitted
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "macros.h"
#include "hyper_params.h"
#include "params.h"
#include "kernel.cu"
#include "util.h"
#include "init.h"
#include "input.h"
#include "functions.cu"

/**
 * Input x,u is splitted over GPU cores/threads
 * Output y,v is streamed (send in batches).
 *
 * It is assumed that x,u all fit in GPU memory, but not necessarily in cache
 * Batches containing parts of y,v are send back to CPU immediately
 *
 * Naming convention
 * i,j,k = indices in flattened arrays
 * n,m = counters
 * N,M = sizes
 *
 * e.g. n = [0,..,N-1]
 */

int main(int argc, char** argv) {
  /**
   * Note there is an effective minimum distance between projected points,
   * but that projecting point further apart requires a higher sampling density.
   *
   * Projecting more that ~20 points may result in a higher (and less random)
   * noise floor.
   */
  auto params = input::read_args(argc, argv);
  Setup<size_t>
    &n_planes = params.n_planes,
    &n_per_plane = params.n_per_plane;

  const auto transformation = PROJECT_PHASE ? Transformation::Full : Transformation::Amplitude;
  const bool add_reference = transformation == Transformation::Amplitude;
  // const bool add_reference = false;

  // TODO rename non-spatial xyz,uvw to setup.obj, setup.projector etc

  std::vector<SPACE>
    u (DIMS * n_per_plane.obj),
    v (DIMS * n_per_plane.projector),
    w (DIMS * n_per_plane.projection);

#ifdef READ_INPUT
  print("Reading input files");
  // overwrite x,u
  auto x = read_bytes<WAVE>(std::string{"../tmp/x_phasor.input"});
  auto u = read_bytes<SPACE>(std::string{"../tmp/x_pos.input"});
  params.n_per_plane.obj = x.size();
  assert(params.n_per_plane.obj == n_per_plane.obj);
  printf("Number of input datapoints/plane: %u\n", x.size());
  printf("Number of input datapoints/plane: %u\n", u.size() / DIMS);
  assert(x.size() == u.size() / DIMS);
  {
    // scale pos, assume pos was normalized
    for (size_t i = 0; i < x.size(); ++i) {
      u[i * DIMS + 0] *= object_width;
      u[i * DIMS + 1] *= object_width;
    }
  }
#else
  // TODO use cmd arg for x length
  const auto shape = Shape::DottedCircle;
  // const auto shape = Shape::Circle;
  auto x = std::vector<WAVE>(n_per_plane.obj, from_polar(1.0));
#endif

  const Geometry
    projector = init::geometry(n_per_plane.projector),
    projection = init::geometry(n_per_plane.projection);
  print_info(projector, n_planes, n_per_plane);

  struct timespec t0, t1, t2;
  auto dt = std::vector<double>(max(n_planes.projection, 1L));
  clock_gettime(CLOCK_MONOTONIC, &t0);

  const auto y_plane = Plane {width: PROJECTOR_WIDTH,
                              offset: {x: 0., y: 0., z: 0.},
                              aspect_ratio: params.aspect_ratio.projector,
                              randomize: params.randomize};
  init::plane(v, y_plane);

  summarize_double('v', v); // TODO edit in case aspect ratio != 1
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("Runtime init: \t%0.3f\n", diff(t0, t1));
  cudaProfilerStart();

  // change offset in first dim
  // note that x,y,z correspond to the spatial dims
  for (auto& i : range(n_planes.obj)) {
    printf("x plane #%i\n", i);

    // linear/geometric interpolation
    const double ratio = i == 0 ? i : i / ((double) n_planes.obj - 1.);
    Cartesian<double> obj_offset = {x: lerp(params.obj_offset.x, ratio),
                                    y: lerp(params.obj_offset.y, ratio),
                                    z: gerp(params.obj_offset.z, ratio)};

    const auto x_plane = Plane {width: lerp(params.obj_width, ratio),
                                offset: obj_offset,
                                aspect_ratio: 1.,
                                randomize: false};
    printf("x_plane: %i, width: %e\n", i, x_plane.width);
#ifndef READ_INPUT
    const double modulate = i / (double) n_planes.obj;
    init::sparse_plane(u, shape, x_plane.width, x_plane.offset, modulate);
#endif

    const auto x_suffix = std::to_string(i);
    write_arrays(x, u, "x" + x_suffix, "u" + x_suffix, x_plane, 0, 0);
    printf("--- --- ---   --- --- ---  --- --- --- \n");

    // The projector distribution is obtained by doing a single backwards transformation
    // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x

    // dt[0] will be overwritten
    auto y = time_transform<Direction::Backwards, false, add_reference>(x, u, v, projector, &t1, &t2, &dt[0], true);
    check_cvector(y);

    if (i == 0) summarize_c('y', y);
    write_arrays(y, v, "y" + x_suffix, "v" + x_suffix, y_plane, dt[0], flops(dt[0], x.size(), y.size()));

    // The projection distributions at various locations are obtained using forward transformations
    for (auto& j : range(n_planes.projection)) {
      // skip half of forward transformations when simulating for prototype
      // TODO allow to disable forward transformation
      if (transformation == Transformation::Amplitude && n_planes.obj >= 4 && j % 2 == 1) continue;
      printf(" z plane #%i\n", j);
      const auto ratio = j == 0 ? 0 : j / ((double) n_planes.projection - 1.);
      const double width = gerp(params.projection_width, ratio);
      obj_offset.z = gerp(params.projection_z_offset, ratio);
      const auto z_plane = Plane {width: width,
                                  offset: obj_offset,
                                  aspect_ratio: params.aspect_ratio.projection,
                                  randomize: params.randomize};

      printf("z_plane: %i, width: %e\n", j, z_plane.width);
      init::plane(w, z_plane);
      // init::plane(w, z_plane, {obj_offset.x, obj_offset.y, z_plane.z_offset});

      // TODO mv z outside loop to avoid unnecessary mallocs
      // auto z = std::vector<WAVE>(n.z);
      auto z = time_transform<Direction::Forwards>(y, v, w, projection, &t1, &t2, &dt[j]);
      check_cvector(z);
      if (i == 0 && j == 0) summarize_c('z', z);

      const auto z_suffix = x_suffix + "_" + std::to_string(j);
      // TODO write this async, next loop can start already
      write_arrays(z, w, "z" + z_suffix, "w" + z_suffix, z_plane, dt[j], flops(dt[j], y.size(), z.size()));
    }

    if (n_planes.projection)
      print_result(dt, y.size(), n_per_plane.projection);
  }
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("done\n");
  cudaProfilerStop();
	return 0;
}
