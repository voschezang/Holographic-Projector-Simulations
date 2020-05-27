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
  const struct {size_t x,y,z;} n = {x: 4,
                                    y: N_sqrt * N_sqrt,
                                    z: N_sqrt * N_sqrt};
  // TODO add cmd line args
  const size_t n_planes = N_PLANES;
  // const bool hd = false;
  const bool hd = true;
  const auto shape = Shape::DottedCircle;
  // const auto shape = Shape::Line;
  const Params params = init::params(Variable::Width, n_planes, hd);
  const Geometry p = init::geometry(n.y);
  // const double mean_amplitude = params.projector.z_offset / (double) n.x;
  const double mean_amplitude = 1.;
  print_info(p, n.x, n.y, n.z);
  struct timespec t0, t1, t2;
  auto dt = std::vector<double>(n_planes);
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // TODO use cmd arg for x length

  // TODO scale input intensity, e.g. 1/n, and also for distance: sqrt(p)/r^2
  // s.t. sum of irradiance/power/amp is 1
  // i.e. n A/d = 1
  // TODO make this optimization optional, as it introduces some error
  auto
    x = std::vector<WTYPE>(n.x, {mean_amplitude, 0.0});

  auto
    u = init::sparse_plane(x.size(), shape, params.input.width),
    v = init::plane(n.y, params.projector);

  summarize_double('u', u);
  summarize_double('v', v); // TODO edit in case hd == true
  write_arrays<FileType::TXT>(x, u, "x", "u", true, params.input);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("Runtime init: \t%0.3f\n", diff(t0, t1));
  cudaProfilerStart();
  printf("--- --- ---   --- --- ---  --- --- --- \n");

  // The projector distribution is obtained by doing a single backwards transformation
  // TODO if x does not fit on GPU then do y += transform(x') for each subset x' in x
  // dt[0] will be overwritten
  auto y = time_transform<Direction::Backward, true>(x, u, v, p, &t1, &t2, &dt[0], true);
  check_cvector(y);
  summarize_c('y', y);
  // TODO edit in case hd == true
  write_arrays<FileType::TXT>(y, v, "y", "v", false, params.projector);

  // The projection distributions at various locations are obtained using forward transformations
  for (size_t i = 0; i < params.projections.size(); ++i) {
    auto suffix = std::to_string(i);
    auto p = init::geometry(n.z);
    auto w = init::plane(n.z, params.projections[i]);
    auto z = time_transform<Direction::Forward, false>(y, v, w, p, &t1, &t2, &dt[i], false);
    check_cvector(z);
    if (i == 0)
      summarize_c('z', z);

    // TODO do this async
    write_arrays<FileType::TXT>(z, w, "z" + suffix, "w" + suffix, false, params.projections[i]);
  }

  print_result(dt, n_planes * y.size(), n_planes * n.z);
  printf("--- --- ---   --- --- ---  --- --- --- \n");
  printf("done\n");
  cudaProfilerStop();
	return 0;
}
