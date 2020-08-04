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
  /* const double z_offset = 0.35; */
  /* const double z_offset = 0.01; */
  /* const auto obj_z_offset = Range<double> {min: z_offset, max: z_offset}; */
  const auto obj_z_offset = Range<double> {min: 0.1, max: 0.05};
  const double projection_width = N_sqrt * 7e-6;
  // projector z_offset is always zero
  // TODO use json file (similar to meta data)
  // TODO add rand seed
  auto p = Params
    {n_planes:     {obj: 1,
                    projector: 1, // unused
                    projection: 1}, // number of projection planes per obj plane
     n_per_plane:  {obj: 1,
                    projector: N_sqrt * N_sqrt,
                    projection: N_sqrt * N_sqrt},
     aspect_ratio: {obj: 1.,
                    projector: 1.,
                    /* projector: HD, */
                    projection: 1.}, // 0.2

     /* obj_shape: Shape::DottedCircle, // TODO */
     obj_offset:  {x: {min: 0.0, max: 0.0},
                   y: {min: 0.0, max: 0.0},
                   z: obj_z_offset},

     obj_width: {min: 0.1, max: 0.1},
     projection_width: {min: projection_width, max: projection_width},
     /* projection_z_offset: {min: 0.1, max: 0.1}, // added to obj offset */
     projection_z_offset: {min: 0., max: 0.}, // added to obj offset

     quadrant_projection: false,
     randomize: false
     /* randomize: true */
    };

  // TODO count n non-constant ranges and static_assert result is <= 1
  // TODO add PROJECT_PHASE to cmd args

  // For all ranged params the max is set to min by default.
  int ch;
  /* while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r:")) != -1) */
  while ((ch = getopt(argc, argv, "X:Z:x:y:z:a:A:u:U:v:V:w:W:o:O:n:N:m:M:q:r:h:")) != -1)
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
      case 'u': p.obj_offset.x.min = p.obj_offset.x.max = strtod(optarg, 0); break;
      case 'U': p.obj_offset.x.max =                      strtod(optarg, 0); break;
      case 'v': p.obj_offset.y.min = p.obj_offset.y.max = strtod(optarg, 0); break;
      case 'V': p.obj_offset.y.max =                      strtod(optarg, 0); break;
      case 'w': p.obj_offset.z.min = p.obj_offset.z.max = strtod(optarg, 0); break;
      case 'W': p.obj_offset.z.max =                      strtod(optarg, 0); break;

      case 'o': p.obj_width.min = p.obj_width.max =                     strtod(optarg, 0); break;
      case 'O': p.obj_width.max =                                       strtod(optarg, 0); break;
      case 'n': p.projection_width.min = p.projection_width.max =       strtod(optarg, 0); break;
      case 'N': p.projection_width.max =                                strtod(optarg, 0); break;
      case 'm': p.projection_z_offset.min = p.projection_z_offset.max = strtod(optarg, 0); break;
      case 'M': p.projection_z_offset.max =                             strtod(optarg, 0); break;

      case 'q': p.quadrant_projection = true; break;
      case 'r': p.randomize =           true; break;
      case 'h': default: show_help(argv[0]);
      }
    }

  assert(p.n_planes.obj >= 1);
  assert(p.n_planes.projector == 1);
  assert(p.n_planes.projection <= 10000);
  const double nonzero = 1e-6; // relatively small constant
  if (p.obj_offset.z.min > 0.)
    assert((p.obj_offset.z.min > nonzero && p.obj_offset.z.max > nonzero));
  else
    assert((-p.obj_offset.z.min > nonzero && -p.obj_offset.z.max > nonzero));

  assert(abs(p.obj_offset.z.min + p.projection_z_offset.min) > nonzero);
  assert(abs(p.obj_offset.z.min + p.projection_z_offset.max) > nonzero);
  assert(abs(p.obj_offset.z.max + p.projection_z_offset.max) > nonzero);
  assert(abs(p.obj_offset.z.max + p.projection_z_offset.min) > nonzero);
  return p;
}

}
