#ifndef MACROS
#define MACROS

#ifndef PROJECT_PHASE
#define PROJECT_PHASE 1
#endif

#define N_sqrt 512
/* #define N_sqrt 1024 */
/* #define N_sqrt 1440 */

// integer functions
#define CEIL(x,y) ((x + y - 1) / (y)) // ceil(int, int)
#define FLOOR(x,y) ((x) / (y)) // integer division
#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)

/* #define DEBUG */
/* #define TEST_CONST_PHASE */

#define ZERO make_cuDoubleComplex(0,0)

#ifndef LAMBDA
#define LAMBDA 0.65e-6 // 650 nm
#endif
#ifndef DISTANCE_REFERENCE_WAVE
#define DISTANCE_REFERENCE_WAVE 0.25 // in meters
#endif

#define TWO_PI (2 * M_PI)
#define TWO_PI_OVER_LAMBDA (TWO_PI / LAMBDA)
#define DIMS 3
#define HD (1920. / 1080.)

#define WARP_SIZE 32

/* #define ARBITRARY_PHASE 0.4912 // used in superposition::per_thread */
#define ARBITRARY_PHASE 0. // used in superposition::per_thread

#define WAVE cuDoubleComplex
#define IO_PRECISION 8
#define SPACE double // space type (for each x,y,z in a cartesian coordinate)

#ifndef NORM_3D
#define NORM_3D norm3d // CUDA math api, can be replaced with a CPU-only function
#endif

#define SPACE_MEMORY_LAYOUT 1
#if (SPACE_MEMORY_LAYOUT)
#define Ix_(i, dim, dims) ((i) * dims + (dim))
#else
#error "TODO this requires changes in superposition kernel"
#define Ix_(i, dim, dims) ((dim) * dims + (i))
#endif

#define Ix(i, dim) Ix_(i, dim, DIMS)

// flattened indices of 2D matrix
#define Ix2D(i,j,dim,n) Ix( (j) + (i) * (n), dim)
#define Ix2D_(i,j,dim, n,dims) Ix_( (j) + (i) * (n), dim, dims)


// phasor source (n, N) and target (m, M) data. Independent from spatial data
#define Yidx(n,m,N,M) ((m) + (n) * (M))
// alt:
// #define Yidx(n,m,N,M) ((n) + (m) * (N))

#define print(x) std::cout << x << '\n'


#define WEIGHT_SQRT 8
#define WEIGHT 8 // complex operations
#define FLOP_PER_POINT (                                                \
                        3     /* (u - v) with u,v \in R^3 */ +          \
                        3     /* power2 x3 */ +                         \
                        2+WEIGHT_SQRT /* sum3, sqrt */ +                \
                        2     /* phase - direction * distance * 2pi/lambda */ + \
                        WEIGHT    /* exp(I phase) == sincos */ +        \
                        3    /* amp/distance * {re, im} */                \
                        )

#endif
