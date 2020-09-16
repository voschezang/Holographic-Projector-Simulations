#ifndef MACROS
#define MACROS

// integer functions
/* #define DIV(x,y) ((x + y - 1) / y) // ceil(int, int) */
#define CEIL(x,y) ((x + y - 1) / (y)) // ceil(int, int)
#define FLOOR(x,y) ((x) / (y)) // integer division
#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)

/* #define DEBUG */
/* #define TEST_CONST_PHASE */
#define STREAM_DATA

#define ZERO make_cuDoubleComplex(0,0)
/* #define VOL(type, x) *((type *) &x) */

#define HD (1920. / 1080.)
/* #define LAMBDA (0.6328e-6)  // wavelength in vacuum: 632.8 nm (HeNe laser) */
#define LAMBDA 0.65e-6 // 650 nm
#define TWO_PI (2 * M_PI)
#define TWO_PI_OVER_LAMBDA (TWO_PI / LAMBDA)
/* #define DISTANCE_REFERENCE_WAVE 0.24 // in meters */
#define DISTANCE_REFERENCE_WAVE 0.25 // in meters
/* #define PROJECTOR_WIDTH 1.344e-2 // projector width = 1920 x 7e-6 */
/* #define PROJECTOR_WIDTH (1920 * 7e-6) // fixed width to allow undersampling (fewer pixels) */
/* /\* #define PROJECTOR_WIDTH (N_sqrt * 7e-6) *\/ */
/* /\* #define PROJECTOR_HEIGHT 7.56e-3 // projector width = 1080 x 7e-6 *\/ */
/* /\* #define PROJECTOR_HEIGHT(hd) ((hd ? 1080. / 1920. : 1.) * PROJECTOR_WIDTH) *\/ */
/* /\* #define PROJECTOR_HEIGHT(aspect_ratio) ((PROJECTOR_WIDTH / aspect_ratio) // height = width / (width / height) *\/ */
/* /\* #define PROJECTOR_WIDTH(aspect_ratio) (aspect_ratio * PROJECTOR_HEIGHT) *\/ */
#define DIMS 3

#define WARP_SIZE 32

/* #define ARBITRARY_PHASE 0.4912 // used in superposition::per_thread */
#define ARBITRARY_PHASE 0. // used in superposition::per_thread

#define WAVE cuDoubleComplex
#define IO_PRECISION 8
#define SPACE double // space type (for each cartesian coordinate)

#ifndef NORM_3D
#define NORM_3D norm3d // CUDA math api
#endif

#define SPACE_MEMORY_LAYOUT 1
#if (SPACE_MEMORY_LAYOUT)
#define Ix(i, dim) ((i) * DIMS + (dim))
/* #define Ix2D(i,j,dim,n) ((k) + DIMS * ((j) + (i) * (n))) */
#else
#error "TODO this requires changes in superposition kernel"
#define Ix(i, dim) ((dim) * DIMS + (i))
/* #define Ix2D(i,j,dim,n) ((k) * DIMS + ((j) + (i) * (n))) */
#endif

// flattened indices of 2D matrix (i,j, width stride n), dimension (dim)
#define Ix2D(i,j,dim,n) Ix( (j) + (i) * (n), dim)


// phasor source (n, N) and target (m, M) data. Independent from spatial data
#define Yidx(n,m,N,M) ((m) + (n) * (M))
/* #define Yidx(n,m,N,M) ((n) + (m) * (N)) */


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
