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
  const auto obj_z_offset = Range<double> {min: 0.1, max: 0.1};
  // projector z_offset is always zero
  auto p = Params
    {n_planes: {obj: 1,
                projector: 1,
                projection: 3},

     n_per_plane: {obj: 1,
                   projector: N_sqrt * N_sqrt,
                   projection: N_sqrt * N_sqrt},
     hd: false,
     /* hd: true, // TODO default false */
     randomize: false,
     /* randomize: true, // TODO default false */
     obj_offset: {x: {min: 0.0, max: 0.0}, // TODO make relative
                  y: {min: 0.0, max: 0.0},
                  z: obj_z_offset},

     rel_obj_width: {min: 1., max: 0.3},
     rel_projection_width: {min: 0.005, max: 0.5},
     /* rel_projection_width: {min: 0.005, max: 0.05}, */
     /* rel_projection_height: {min: 0.005, max: 0.005}, */
     projection_z_offset: obj_z_offset
    };

  // TODO count n non-constant ranges and static_assert result is <= 1

  // For all ranged params the max is set to min by default.
  int ch;
  /* while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r:")) != -1) */
  while ((ch = getopt(argc, argv, "X:Z:x:y:z:d:r:u:U:v:V:w:W:o:O:n:N:m:M:h:")) != -1)
    {
      switch(ch) {
      case 'X': p.n_planes.obj                   = strtol(optarg, 0, 10); break;
      /* 'Y': p.n_planes.projector is constant */
      case 'Z': p.n_planes.projection            = strtol(optarg, 0, 10); break;
      case 'x': p.n_per_plane.obj        = strtol(optarg, 0, 10); break;
      case 'y': p.n_per_plane.projector  = strtol(optarg, 0, 10); break;
      case 'z': p.n_per_plane.projection = strtol(optarg, 0, 10); break;

      case 'd': p.hd = true; break;
      case 'r': p.randomize = true; break;

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
