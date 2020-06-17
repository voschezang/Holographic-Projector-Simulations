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
  const auto obj_z_offset = Range<double> {min: 0.4, max: 0.1};
  // projector z_offset is always zero
  auto p = Params
    {n_planes: {obj: 1,
                projector: 1,
                projection: 1},

     datapoins_per_plane: {obj: 100,
                           projector: N_sqrt * N_sqrt,
                           projection: N_sqrt * N_sqrt},
     hd: true, // TODO default false
     randomize: false,
     obj_offset: {x: {min: 0.0, max: 0.0},
                  y: {min: 0.0, max: 0.0},
                  z: obj_z_offset},

     rel_obj_width: {min: 0.3, max: 0.5},
     rel_projection_width: {min: 1, max: 1},
     projection_z_offset: obj_z_offset
    };

  // For all ranged params the max is set to min by default.
  int ch;
  /* while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r:")) != -1) */
  while ((ch = getopt(argc, argv, "X:Z:x:y:z:u:U:v:V:w:W:o:O:n:N:m:M:d:r:h:")) != -1)
    {
      switch(ch) {
      case 'X': p.n_planes.obj                   = strtol(optarg, 0, 10); break;
      /* 'Y': p.n_planes.projector is constant */
      case 'Z': p.n_planes.projection            = strtol(optarg, 0, 10); break;
      case 'x': p.datapoins_per_plane.obj        = strtol(optarg, 0, 10); break;
      case 'y': p.datapoins_per_plane.projector  = strtol(optarg, 0, 10); break;
      case 'z': p.datapoins_per_plane.projection = strtol(optarg, 0, 10); break;

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
  return p;
}

}
