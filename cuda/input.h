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
  const auto obj_z_offset = Range<double> {min: 0.2, max: 0.01};
  /* const double projection_width = N_sqrt * 7e-6; */
  /* const double obj_width =  PROJECTOR_WIDTH; */
  /* const double obj_width =  N_sqrt * 7e-6; */
  const double obj_width =  0.00003;
  // projector z_offset is always zero
  // TODO use json file (similar to meta data)
  // TODO add rand seed
  auto p = Params
    {n_planes:     {obj: 1,
                    projector: 1, // unused
                    projection: 3}, // number of projection planes per obj plane
     n_per_plane:  {obj: 1,
                    projector: N_sqrt * N_sqrt,
                    projection: N_sqrt * N_sqrt},
     aspect_ratio: {obj: 1.,
                    projector: 1.,
                    /* projector: HD, */
                    projection: 1.},

     /* obj_shape: Shape::DottedCircle, // TODO */
     obj_offset:  {x: {min: 0.0, max: 0.0},
                   y: {min: 0.0, max: 0.0},
                   z: obj_z_offset},

     /* obj_width: {min: 0.003, max: 0.01}, */
     obj_width: {min: obj_width, max: obj_width},
     projector_width: {min: 1920 * 7e-6, max: 1920 * 7e-6},
     projection_width: {min: obj_width * 1.05, max: obj_width * 1.05},
     /* projection_width: {min: 0.000018, max: 0.01}, */
     projection_z_offset: {min: 0., max: 0.}, // added to obj offset

     algorithm: 2,
     quadrant_projection: false,
     randomize: false,
     /* randomize: true, */
     n_streams: 16,
     thread_size: {16, 4},
     blockDim: {BLOCKDIMX, BLOCKDIMY, 1},
     gridDim: {GRIDDIMX, GRIDDIMY, 1}
    };

  // TODO add PROJECT_PHASE to cmd args

  // For all ranged params the max is set to min by default.
  int ch;
  /* while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r:")) != -1) */
  while ((ch = getopt(argc, argv, "X:Z:x:y:z:a:A:u:U:v:V:w:W:o:O:l:L:n:N:m:M:p:q:rs:t:T:b:B:g:G:h")) != -1)
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
      case 'l': p.projector_width.min = p.projector_width.max =         strtod(optarg, 0); break;
      case 'L': p.projector_width.max =                                 strtod(optarg, 0); break;
      case 'n': p.projection_width.min = p.projection_width.max =       strtod(optarg, 0); break;
      case 'N': p.projection_width.max =                                strtod(optarg, 0); break;
      case 'm': p.projection_z_offset.min = p.projection_z_offset.max = strtod(optarg, 0); break;
      case 'M': p.projection_z_offset.max =                             strtod(optarg, 0); break;

      case 'p': p.algorithm = strtol(optarg, 0, 10); break;
      case 'q': p.quadrant_projection = true; break;
      case 'r': p.randomize =           true; break;
      case 's': p.n_streams =     strtol(optarg, 0, 10); break;
      case 't': p.thread_size.x = strtol(optarg, 0, 10); break;
      case 'T': p.thread_size.y = strtol(optarg, 0, 10); break;
      case 'b': p.blockDim.x =    strtol(optarg, 0, 10); break;
      case 'B': p.blockDim.y =    strtol(optarg, 0, 10); break;
      case 'g': p.gridDim.x =     strtol(optarg, 0, 10); break;
      case 'G': p.gridDim.y =     strtol(optarg, 0, 10); break;
      case 'h': default: show_help(argv[0]);
      }
    }

  assert(p.n_planes.obj >= 1);
  assert(p.n_planes.projector == 1);
  assert(p.n_planes.projection <= 10000);
  assert(p.n_per_plane.projector > 0);
  if (p.n_planes.projection)
    assert(p.n_per_plane.projection > 0);
  else
    p.n_per_plane.projection = 1;

  assert(is_square(p.n_per_plane.projector));
  assert(is_square(p.n_per_plane.projection));

  assert(p.blockDim.x * p.blockDim.y <= 1024);
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
