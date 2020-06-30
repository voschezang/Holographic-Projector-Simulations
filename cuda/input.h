#include <stddef.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>


void show_help(const char *p) {
  // TODO extend help
  printf("Usage %s [OPTION]..\n\n"
         "\t-n, -N set min, max"
         "\n for attr.max is set to min if it is not provided after the corresponding .min"
         , p);
}

namespace input {

Params read_args(int argc, char **argv) {
  /* const double z_offset = 1000 * LAMBDA; */
  const double z_offset = 0.35;
  const auto obj_z_offset = Range<double> {min: z_offset, max: z_offset};
  /* const auto obj_z_offset = Range<double> {min: 0.35, max: 0.35}; */
  // projector z_offset is always zero
  auto p = Params
    {n_planes:     {obj: 1,
                    projector: 1,
                    projection: 2},
     n_per_plane:  {obj: 1,
                    projector: N_sqrt * N_sqrt,
                    projection: N_sqrt * N_sqrt},
     aspect_ratio: {obj: 1.,
                    projector: 1., // HD
                    projection: 1.}, // 0.2

     /* obj_shape: Shape::DottedCircle, // TODO */
     obj_offset:  {x: {min: 0.0, max: 0.0}, // TODO make relative?
                   y: {min: 0.0, max: 0.0},
                   z: obj_z_offset},

     rel_obj_width: {min: 1. / PROJECTOR_WIDTH, max: 0.3}, // relative to PROJECTOR_WIDTH
     rel_projection_width: {min: N_sqrt * 8.019 * LAMBDA, max: N_sqrt * 16.3 * LAMBDA}, // 0.005
     projection_z_offset: obj_z_offset,

     randomize: false
     /* randomize: true */
    };

  // TODO count n non-constant ranges and static_assert result is <= 1

  // For all ranged params the max is set to min by default.
  int ch;
  /* while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r:")) != -1) */
  while ((ch = getopt(argc, argv, "X:Z:x:y:z:a:A:u:U:v:V:w:W:o:O:n:N:m:M:r:h:")) != -1)
    {
      switch(ch) {
      case 'X': p.n_planes.obj            = strtol(optarg, 0, 10); break;
      /* 'Y': p.n_planes.projector is constant */
      case 'Z': p.n_planes.projection     = strtol(optarg, 0, 10); break;
      case 'x': p.n_per_plane.obj         = strtol(optarg, 0, 10); break;
      case 'y': p.n_per_plane.projector   = strtol(optarg, 0, 10); break;
      case 'z': p.n_per_plane.projection  = strtol(optarg, 0, 10); break;

      case 'a': p.aspect_ratio.projector  = strtod(optarg, 0); break;
      case 'A': p.aspect_ratio.projection = strtod(optarg, 0); break;

      /* case 's': p.obj_shape.       = p.obj_shape        = str_to_shape(optarg, 0); break; */
      case 'u': p.obj_offset.x.min = p.obj_offset.x.min = strtod(optarg, 0); break;
      case 'U': p.obj_offset.x.max =                      strtod(optarg, 0); break;
      case 'v': p.obj_offset.y.min = p.obj_offset.y.max = strtod(optarg, 0); break;
      case 'V': p.obj_offset.y.max =                      strtod(optarg, 0); break;
      case 'w': p.obj_offset.z.min = p.obj_offset.z.max = strtod(optarg, 0); break;
      case 'W': p.obj_offset.z.max =                      strtod(optarg, 0); break;

      case 'o': p.rel_obj_width.min = p.rel_obj_width.max =               strtod(optarg, 0); break;
      case 'O': p.rel_obj_width.max =                                     strtod(optarg, 0); break;
      case 'n': p.rel_projection_width.min = p.rel_projection_width.min = strtod(optarg, 0); break;
      case 'N': p.rel_projection_width.max =                              strtod(optarg, 0); break;
      case 'm': p.projection_z_offset.min = p.projection_z_offset.max =   strtod(optarg, 0); break;
      case 'M': p.projection_z_offset.max =                               strtod(optarg, 0); break;

      case 'r': p.randomize = true; break;
      case 'h': default: show_help(argv[0]);
      }
    }

  assert(p.n_planes.obj >= 1);
  assert(p.n_planes.projector == 1);
  assert(p.n_planes.projection <= 10000);
  // TODO count n non-constant ranges and assert result is <= 1
  return p;
}

}
