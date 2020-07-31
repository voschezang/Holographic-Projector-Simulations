#ifndef PARAMS
#define PARAMS

#include <stdlib.h>
#include "macros.h"

enum class Algorithm {Naive, Alt};

enum class Shape {Line, LogLine, Cross, Circle, DottedCircle};
enum class Transformation {Full, Amplitude}; // Full: keep phase+amp, Amplitude: rm phase

template<typename T>
struct Setup { T obj, projector, projection; };

template<typename T>
struct Cartesian { T x,y,z; }; // 3D space

template<typename T = double>
struct Polar { T amp, phi; }; // phasor TODO use to avoid ambiguity in cuda code

template<typename T = double>
struct Range {T min, max; };

struct Plane {
  // TODO simplify this struct, avoid duplicate data?
  double width;
  Cartesian<double> offset;
  double aspect_ratio; // image width / height
  bool randomize;
};

/**
 * Params for command line args, used to generate datapoints.
 *
 * It is assumed that a single parameter is changed for each plane.
 */
struct Params {
  // number of planes, number of datapoints per plane
  Setup<size_t> n_planes;
  Setup<size_t> n_per_plane;
  Setup<double> aspect_ratio; // image width / height

  Cartesian<Range<double>> obj_offset;

  Range<double>
    obj_width,        // relative to projector width, affects sampling density
    projection_width, // relative to object width
    projection_z_offset;

  bool
    quadrant_projection,
    randomize; // TODO add option to compute local average
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
         gridDim,     // prod(gridDim.x,y,z), i.e. n blocks per kernel
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
