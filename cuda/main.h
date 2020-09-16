#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include "macros.h"

enum class Algorithm {Naive, Alt};

enum class Shape {Line, LogLine, Cross, Circle, DottedCircle};
enum class Transformation {Full, Amplitude}; // Full: keep phase+amp, Amplitude: rm phase

template<typename T = double>
struct Range { T min, max; };

template<typename T>
struct Setup { T obj, projector, projection; };

struct Polar { double amp, phase; };

template<typename T>
struct Cartesian { T x,y,z; }; // 3D space, generic version of dim3

// similar to CUDA dim3 but fewer dims and larger size
struct dim2 { size_t x, y; };

struct SumRange {size_t sum, min, max; };

struct Plane {
  // TODO simplify this struct, avoid duplicate data?
  double width;
  Cartesian<double> offset;
  double aspect_ratio; // image width / height
  bool randomize; // TODO rename => randomize_projector_pixels
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
    projector_width,
    projection_width, // relative to object width
    projection_z_offset;

  int algorithm;
  bool
    quadrant_projection,
    randomize; // TODO add option to compute local average

  unsigned int n_streams;
  dim2 thread_size;
  dim3 blockDim, gridDim;
  /* std::string input_filename; // input filename or empty string, positions will be scaled to `obj_width` */
};


/**
 * CUDA params
 *
 * Geometry Hierarchy (parameters)
 * thread_size -> batch_size -> n
 * note that 1 kernel computes 1 or more superpositions of source datapoints (y) w.r.t all input datapoints (x).
 */
struct Geometry {
  int algorithm;
  dim3
    blockDim,
    gridDim,
    gridSize; // gridDim * blockDim
  dim2
    n, // number of source, target datapoints // TODO rm to avoid redundancy?
    batch_size, // number of datapoints per batch
    n_batches,
    thread_size; // number of datapoints per thread (per kernel)
  size_t n_streams;
};

/** CUDA GPU version of std::vector, used for device memory and pinned memory
 */
template<typename T>
struct CUDAVector {
  T *data;
  size_t size;
};

template<typename T>
struct ConstCUDAVector {
  const T *data;
  const size_t size;
};

#endif
