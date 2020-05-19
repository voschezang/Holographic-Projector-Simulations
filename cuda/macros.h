#ifndef MACROS
#define MACROS

#include "macros.h"

/* #define DIV(x,y) ((x + y - 1) / y) // ceil(int, int) */
#define CEIL(x,y) ((x + y - 1) / y) // ceil(int, int)


#define ZERO make_cuDoubleComplex(0,0)
/* #define VOL(type, x) *((type *) &x) */

#define HD (1920 * 1080)
#define LAMBDA (1 * 0.6328e-6)  // wavelength in vacuum: 632.8 nm (HeNe laser)
#define TWO_PI (2 * M_PI)
#define TWO_PI_OVER_LAMBDA (TWO_PI / LAMBDA)
/* #define SCALE (1 / LAMBDA) */
#define SCALE (LAMBDA / 0.6328e-6)
#define PROJECTOR_DISTANCE

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
/* #define WTYPE_cuda cuDoubleComplex // wave type for CUDA device */
/* #define WTYPE double complex // wave type */
/* #define ABS(x) (cabs(x)) */
#define WTYPE cuDoubleComplex // wave type for host
#define IO_PRECISION 8
/* #define ABS(x) (cuCabs(x)) */
#define STYPE double  // space (coordinate) type
#else
/* #define WTYPE_cuda cuFloatComplex // wave type for CUDA device */
#define WTYPE cuFloatComplex  // wave type
#define STYPE float // space (coordinate) type
#endif // DOUBLE_PRECISION


// #define Ix(i,j) i + j * N_sqrt
// #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * N_sqrt
/* #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * DIMS */
#define I_(i,j) (j + (i) * N_sqrt)
/* #define Ix(i,j,k) (k + (j) * DIMS + (i) * DIMS * N_sqrt) */
#define Ix(i,j,k,n) (k + (j) * DIMS + (i) * DIMS * (n))

/* #define Matrix(type, size1, size2) std::vector<std::vector<type>>(size1, std::vector<type>(size2)) */
/* #define Matrix(type, size1, size2) std::vector<vtype<dtype>>(size1, type(size2)) */
/* #define Matrix(type, size1) std::vector<vtype<dtype>>(size1, type(size2)) */
#define print(x) std::cout << x << '\n'


// TODO check # operations for abs/angle etc
// see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
// TODO use sincos instead of exp, and 2 muls (re, im)
#define WEIGHT_DIV 4
#define WEIGHT 8 // complex operations
#define FLOP_PER_POINT (                            \
             3     /* (u - v) with u,v \in R^3 */ + \
             3+(3+1)*WEIGHT_DIV /* |u| power, sum, power */ +         \
             2*WEIGHT    /* abs(x_i), angle(x_i) */ +                 \
             1     /* amp/distance */ +                               \
             3     /* phase - direction * distance * 2pi/lambda */ +  \
             1+WEIGHT    /* a exp(b) */                               \
             )

#endif
