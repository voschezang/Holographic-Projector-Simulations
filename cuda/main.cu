// #define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h> // unused in this file but causes error if omitted

#include "macros.h"
#include "hyper_params.h"
#include "kernel.cu"
#include "init.h"
#include "util.h"
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


int main() {
  const struct {size_t x,y,z;} n = {x: 1,
                                    y: N_sqrt * N_sqrt,
                                    z: N_sqrt * N_sqrt};
  // TODO add cmd line args
  // TODO struct n_planes .x .y. z
  const size_t
    n_x_planes = 4,
    n_z_planes = 1;

  // const Transformation projector = Transformation::Full;
  const auto transformation = Transformation::Amplitude2;
  // const bool hd = false;
  const bool hd = true;
  const auto shape = Shape::DottedCircle;
  // const auto shape = Shape::Circle;

  Params params = init::params(Variable::Width, n_z_planes, hd);
  const Geometry p = init::geometry(n.y);
  const bool add_const_source = transformation == Transformation::Amplitude2;
  const double height = params.projector.width * (hd ? 1080. / 1920. : 1.);
  print_info(p, n.x, n.y, n.z);

  struct timespec t0, t1, t2;
  // auto dt = std::vector<double>(n_x_planes * n_z_planes); // TODO
  auto dt = std::vector<double>(n_z_planes);
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // TODO use cmd arg for x length

  // TODO scale input intensity, e.g. 1/n, and also for distance: sqrt(p)/r^2
  // s.t. sum of irradiance/power/amp is 1
  // i.e. n A/d = 1
  // TODO make this optimization optional, as it introduces some error
  auto x = std::vector<WTYPE>(n.x, from_polar(1.0, 0.0));
  auto v = init::plane(n.y, params.projector);

  summarize_double('v', v); // TODO edit in case hd == true
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("Runtime init: \t%0.3f\n", diff(t0, t1));
  cudaProfilerStart();

  // change offset in first dim
  // note that x,z now correspond to the spatial dims
  auto rel_x_offsets = linspace(n_x_planes, 0.2, 0.8);
  auto rel_y_offsets = linspace(n_x_planes, 0.8, 0.2);
  auto z_offsets = geomspace(n_x_planes, 0.4, 0.4);
  for (auto& i : range(n_x_planes)) {
    printf("x plane #%i\n", i);
    params.projector.z_offset = z_offsets[i];
    const double
      x_offset = rel_x_offsets[i] * params.projector.width,
      y_offset = rel_y_offsets[i] * height,
      modulate = i / (double) n_x_planes;
    auto u = init::sparse_plane(x.size(), shape, params.input.width, x_offset, y_offset, modulate);
    const auto x_suffix = std::to_string(i);
    write_arrays<FileType::TXT>(x, u, "x" + x_suffix, "u" + x_suffix, params.input);
    printf("--- --- ---   --- --- ---  --- --- --- \n");

    // The projector distribution is obtained by doing a single backwards transformation
    // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x

    v = init::plane(n.y, params.projector);

    // dt[0] will be overwritten
    auto y = time_transform<Direction::Backwards, add_const_source>(x, u, v, p, &t1, &t2, &dt[0], true);
    check_cvector(y);

    if (i == 0)
      summarize_c('y', y);

    write_arrays<FileType::TXT>(y, v, "y" + x_suffix, "v" + x_suffix, params.projector);
    // square amp and rm phase after saving
    if (transformation == Transformation::Amplitude2)
      square_amp(y);

    // The projection distributions at various locations are obtained using forward transformations
    auto p = init::geometry(n.z);
    for (auto& j : range(params.projections.size())) {
      // skip half of forward transformations when simulating for prototype
      if (transformation == Transformation::Amplitude2 && n_x_planes >= 4 && j % 2 == 1) continue;
      printf(" z plane #%i\n", j);
      auto w = init::plane(n.z, params.projections[j], x_offset, y_offset);
      // TODO mv z outside loop to avoid unnecessary mallocs
      // auto z = std::vector<WTYPE>(n.z);
      auto z = time_transform<Direction::Forwards>(y, v, w, p, &t1, &t2, &dt[j]);
      check_cvector(z);
      if (i == 0 && j == 0)
        summarize_c('z', z);

      const auto z_suffix = x_suffix + "_" + std::to_string(j);
      // TODO write this async, next loop can start already
      write_arrays<FileType::TXT>(z, w, "z" + z_suffix, "w" + z_suffix, params.projections[j]);
    }

    print_result(dt, y.size(), n.z);
  }
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("done\n");
  cudaProfilerStop();
	return 0;
}
