#ifndef PARAMS
#define PARAMS

#include <stdlib.h>
#include "macros.h"

enum class Shape {Line, Cross, Circle, DottedCircle};
enum class Variable {Offset, Width}; // TODO rm
enum class Transformation {Full, Amplitude}; // Full: keep phase+amp, Amplitude: rm phase

// x: object, y: projector, z: projection plane
struct N { size_t x,y,z; };

struct Setup { size_t obj, projector, projection; };

template<typename T>
struct Cartesian { T x,y,z; }; // 3D space

struct Polar { double amp, phi; }; // phasor TODO use to avoid ambiguity in cuda code

template<typename T = double>
struct Range {T min, max; };

struct Plane {
  // TODO simplify this struct

/* Plane() : width(1), z_offset(0), randomize(false) {}; */
  /* char name; */
  double width;
  /* double offset[DIMS]; // TODO */
  double z_offset;
  bool randomize;
  bool hd; // TODO
};

/* struct Params { */
/*   // Simulation parameters, used to init plane distributions */
/*   Plane input; // ground truth */
/*   /\* std::vector<Plane> inputs; // ground truth *\/ */
/*   Plane projector; */
/*   std::vector<Plane> projections; // approximation of input */
/* }; */


/* struct Params2 { */
/*   /\* N n; *\/ */
/*   std::vector<Plane> inputs; // ground truth */
/*   std::vector<Plane> projectors; */
/*   std::vector<Plane> projections; // approximation of input */
/* }; */


/**
 * Params for command line args, used to generate datapoints.
 *
 * It is assumed that a single parameter is changed for each plane.
 */
struct Params {
  Setup n_planes, datapoins_per_plane; // number of ..
  bool hd,
    randomize; // TODO add option to compute local average

  Cartesian<Range<double>> obj_offset;
  Range<double>
    rel_obj_width,        // relative to projector width, affects sampling density
    rel_projection_width, // relative to object width
    projection_z_offset;
  /* std::string input_filename; // input filename or empty string, positions will be scaled to `obj_width` */
};


/**
 * CUDA params
 *
 * Geometry Hierarchy (parameters)
 * thread < block < grid < kernel < batch < stream
 * (hyper parameters are defined using macro's, to avoid dynamic memory)
 *
 * note that 1 kernel computes 1 or more superpositions w.r.t all input datapoints.
 */
struct Geometry {
  size_t blockSize,   // prod(blockDim.x,y,z), i.e. threads per block
         gridSize,    // prod(gridDim.x,y,z), i.e. n blocks per kernel
         kernel_size, // n output datapoints per kernel
         batch_size,  // n kernels per batch
         stream_size, // n batches per stream
         n_streams;

  // secondary settings, derived from settings above
  // TODO use class with lazy getter methods
  // total number of ..
  size_t n_batches, // total n stream batches
         n_kernels; // total n kernel calls (excl. agg)
  size_t n_per_stream, n_per_batch, n_per_kernel; // n datapoints per stream/batch/kernel (grid)
  double n_per_block, n_per_thread; // expected values, e.g. `0.5` means half of the blocks/threads do nothing

  // TODO differentiate between input and output datapoints
  /* double n_per_block; // can be < 1, i.e. not all blocks are used */
  /* double n_per_thread; // can be < 1, i.e. not all threads are used */
  size_t kernels_per_stream;
};

#endif
