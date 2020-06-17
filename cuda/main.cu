// #define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h> // unused in this file but causes error if omitted

#include "macros.h"
#include "hyper_params.h"
#include "params.h"
#include "kernel.cu"
#include "util.h"
#include "init.h"
#include "input.h"
#include "functions.h"

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

  // auto input_params = read_args(argc, argv);
  auto params = input::read_args(argc, argv);
  // Params params = init::params(Variable::Width, n_z_planes, input_params.hd, rel_object_width);

  // TODO rm aliases
  N n = {params.datapoins_per_plane.obj,
         params.datapoins_per_plane.projector,
         params.datapoins_per_plane.projection};

  Setup &n_planes = params.n_planes;
  // Setup &datapoints_per_plane;

  const auto transformation = PROJECT_PHASE ? Transformation::Full : Transformation::Amplitude;
  const bool add_reference = transformation == Transformation::Amplitude;

  auto
    u = std::vector<STYPE>(n.x * DIMS),
    v = std::vector<STYPE>(n.y * DIMS),
    w = std::vector<STYPE>(n.z * DIMS);

#ifdef READ_INPUT
  print("Reading input files");
  auto x = read_bytes<WTYPE>(std::string{"../tmp/x_phasor.input"});
  auto u = read_bytes<STYPE>(std::string{"../tmp/x_pos.input"});
  n.x = params.datapoints_per_plane = x.size();
  printf("Number of input datapoints: %u\n", n.x);
  printf("Number of input datapoints: %u\n", u.size() / DIMS);
  assert(x.size() == u.size() / DIMS);
  {
    // scale pos, assume pos was normalized
    const double width = rel_object_width * PROJECTOR_WIDTH;
    for (size_t i = 0; i < x.size(); ++i) {
      u[i * DIMS + 0] *= width;
      u[i * DIMS + 1] *= width;
    }
  }
#else
  // TODO use cmd arg for x length
  const auto shape = Shape::DottedCircle;
  // const auto shape = Shape::Circle;
  auto x = std::vector<WTYPE>(n.x, from_polar(1.0, 0.0));
#endif

  const auto
    geometry_y = init::geometry(n.y),
    geometry_z = init::geometry(n.z);
  print_info(geometry_y, n.x, n.y, n.z);

  struct timespec t0, t1, t2;
  auto dt = std::vector<double>(max(n_planes.projection, 1L));
  clock_gettime(CLOCK_MONOTONIC, &t0);

  const auto y_plane = Plane {width: PROJECTOR_WIDTH, z_offset: 0., randomize: params.randomize, hd: params.hd};
  init::plane(v, y_plane);

  summarize_double('v', v); // TODO edit in case hd == true
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("Runtime init: \t%0.3f\n", diff(t0, t1));
  cudaProfilerStart();

  // change offset in first dim
  // note that x,y,z correspond to the spatial dims
  for (auto& i : range(n_planes.obj)) {
    printf("x plane #%i\n", i);

    // linear/geometric interpolation
    const double di = i / (double) n_planes.obj;
    const Cartesian<double> obj_offset = {x: lerp(params.obj_offset.x, di),
                                       y: lerp(params.obj_offset.y, di),
                                       z: gerp(params.obj_offset.z, di)};
    auto plane = Plane {width: lerp(params.rel_obj_width, di) * PROJECTOR_WIDTH,
                        z_offset: obj_offset.z,
                        randomize: false,
                        hd: false};
#ifndef READ_INPUT
    const double modulate = i / (double) n_planes.obj;
    init::sparse_plane(u, shape, plane.width, obj_offset, modulate);
#endif

    const auto x_suffix = std::to_string(i);
    write_arrays(x, u, "x" + x_suffix, "u" + x_suffix, plane);
    printf("--- --- ---   --- --- ---  --- --- --- \n");

    // The projector distribution is obtained by doing a single backwards transformation
    // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x

    // dt[0] will be overwritten
    auto y = time_transform<Direction::Backwards, false, add_reference>(x, u, v, geometry_y, &t1, &t2, &dt[0], true);
    check_cvector(y);

    if (i == 0)
      summarize_c('y', y);

    write_arrays(y, v, "y" + x_suffix, "v" + x_suffix, y_plane);
    // square amp and rm phase after saving
    // TODO rename function, apply before normalization?
    if (transformation == Transformation::Amplitude)
      rm_phase(y);

    // The projection distributions at various locations are obtained using forward transformations
    for (auto& j : range(n_planes.projection)) {
      // skip half of forward transformations when simulating for prototype
      // TODO allow to disable forward transformation
      if (transformation == Transformation::Amplitude && n_planes.obj >= 4 && j % 2 == 1) continue;
      printf(" z plane #%i\n", j);
      const auto ratio = j / (double) n_planes.projection;
      plane = Plane {width:     lerp(params.rel_projection_width, ratio) * PROJECTOR_WIDTH,
                     z_offset:  gerp(params.projection_z_offset, ratio),
                     randomize: params.randomize,
                     hd:        false};

      init::plane(w, plane, {obj_offset.x, obj_offset.y, plane.z_offset});
      // TODO mv z outside loop to avoid unnecessary mallocs
      // auto z = std::vector<WTYPE>(n.z);
      auto z = time_transform<Direction::Forwards>(y, v, w, geometry_z, &t1, &t2, &dt[j]);
      check_cvector(z);
      if (i == 0 && j == 0)
        summarize_c('z', z);

      const auto z_suffix = x_suffix + "_" + std::to_string(j);
      // TODO write this async, next loop can start already
      write_arrays(z, w, "z" + z_suffix, "w" + z_suffix, plane);
    }

    print_result(dt, y.size(), n.z);
  }
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("done\n");
  cudaProfilerStop();
	return 0;
}
