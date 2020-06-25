#ifndef MACROS
#define MACROS

#include "macros.h"

/* #define DIV(x,y) ((x + y - 1) / y) // ceil(int, int) */
#define CEIL(x,y) ((x + y - 1) / y) // ceil(int, int)


/* #define DEBUG */
#define ZERO make_cuDoubleComplex(0,0)
/* #define VOL(type, x) *((type *) &x) */

#define HD (1920. / 1080.)
/* #define LAMBDA (0.6328e-6)  // wavelength in vacuum: 632.8 nm (HeNe laser) */
#define LAMBDA 0.65e-6 // 650 nm
#define TWO_PI (2 * M_PI)
#define TWO_PI_OVER_LAMBDA (TWO_PI / LAMBDA)
#define DISTANCE_REFERENCE_WAVE 0.24 // in meters
/* #define PROJECTOR_WIDTH 1.344e-2 // projector width = 1920 x 7e-6 */
/* #define PROJECTOR_WIDTH (1920 * 7e-6) */
#define PROJECTOR_WIDTH (N_sqrt * 7e-6)
/* #define PROJECTOR_HEIGHT 7.56e-3 // projector width = 1080 x 7e-6 */
/* #define PROJECTOR_HEIGHT(hd) ((hd ? 1080. / 1920. : 1.) * PROJECTOR_WIDTH) */
#define PROJECTOR_HEIGHT(aspect_ratio) ((PROJECTOR_WIDTH / aspect_ratio) // height = width / (width / height)
/* #define PROJECTOR_WIDTH(aspect_ratio) (aspect_ratio * PROJECTOR_HEIGHT) */
#define DIMS 3

#define MAX_INPUT_SIZE 0 // TODO, specific for GPU

#define ARBITRARY_PHASE 0.4912 // used in superposition::per_thread
/* #define ARBITRARY_PHASE 0. // used in superposition::per_thread */

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

#ifndef NORM_3D
#define NORM_3D norm3d // CUDA math api
#endif

// #define Ix(i,j) i + j * N_sqrt
// #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * N_sqrt
/* #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * DIMS */
/* #define I_(i,j) (j + (i) * N_sqrt) */
/* #define Ix(i,j,k) (k + (j) * DIMS + (i) * DIMS * N_sqrt) */
/* #define Ix(i,j,k,n) (k + (j) * DIMS + (i) * DIMS * (n)) */
#define Ix(i,j,k,n) (k + DIMS * ((j) + (i) * (n)))

// two variants: shape (DIMS, N) and (N, DIMS)
/* #define Ix2(i,j,_) (j + i * DIMS) */
/* #define Ix2(i,j,N) (i + j * N) */

/* #define Matrix(type, size1, size2) std::vector<std::vector<type>>(size1, std::vector<type>(size2))
/* #define Matrix(type, size1, size2) std::vector<vtype<dtype>>(size1, type(size2))
/* #define Matrix(type, size1) std::vector<vtype<dtype>>(size1, type(size2)) */
#define print(x) std::cout << x << '\n'


// TODO check # operations for abs/angle etc
// see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
#define WEIGHT_SQRT 4
#define WEIGHT 8 // complex operations
#define FLOP_PER_POINT (                                                \
                        3     /* (u - v) with u,v \in R^3 */ +          \
                        3+2+WEIGHT_SQRT /* power2, sum3, sqrt */ +      \
                        2*WEIGHT    /* abs(x_i), angle(x_i) */ +        \
                        1     /* amp / distance */ +                    \
                        3     /* phase - direction * distance * 2pi/lambda */ + \
                        WEIGHT    /* exp(I phase) == sincos */ +        \
                        2    /* a * {re, im} */                         \
                        )

#endif
