#ifndef MACROS
#define MACROS

/* #include <thrust/device_vector.h> */
/* #include <thrust/reduce.h> */
/* #include <thrust/complex.h> */

#define DEBUG
#define DIMS 3
// TODO use N,M
/* #define N_sqrt 128 */
#define N_sqrt 256
/* #define N_sqrt 512 */
/* #define N_sqrt 1024 */
/* #define N_sqrt 8 */
#define N (N_sqrt * N_sqrt)
#define N2 (N_sqrt * N_sqrt)
/* #define BATCH_SIZE (N / 65536 ) // number of y-datapoints per batch (kernel invocation), increase this to reduce sync overhead */
/* #define BATCH_SIZE (N / 32768 ) // number of y-datapoints per batch (kernel invocation), increase this to reduce sync overhead */
/* #define BATCH_SIZE (N / 8192) // number of y-datapoints per batch (kernel invocation), increase this to reduce sync overhead */
#define BATCH_SIZE 8 // number of y-datapoints per batch (kernel invocation), increase this to reduce sync overhead
// TODO compute optimal batch size as function of N
#define N_BATCHES (N + BATCH_SIZE - 1) / BATCH_SIZE

/* #define Y_BATCH_SIZE */
/* #define G_BATCH_SIZE */

// N^2 computations
// 1) N^2 cores
// 2) N cores with N datapoints per core
// 3) N cores with M batches = N/M datapoints per core
// if N > #cores, use grid-stride loop
//    for x or for y?

// TODO 1 thread per data element? or streaming? assume N >> total n threads

// THREADS_PER_BLOCK, BLOCKIDM are independent of N, but max. for the GPU
#define THREADS_PER_BLOCK 128
/* #define THREADS_PER_BLOCK 256 */
/* #define BLOCKDIM 256 */
#define BLOCKDIM (2 * THREADS_PER_BLOCK)
// #define BLOCKDIM (N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK

#define N_PER_THREAD N / BLOCKDIM / THREADS_PER_BLOCK // for input (x), thus independent of batches
// the value N_PER_THREAD is used implicitly in gridDim.x

#define LAMBDA 0.6328e-6  // wavelength in vacuum: 632.8 nm (HeNe laser)
#define TWO_PI (2 * M_PI)
#define TWO_PI_OVER_LAMBDA TWO_PI / LAMBDA
/* #define SCALE (1 / LAMBDA) */
#define SCALE (LAMBDA / 0.6328e-6)
#define PROJECTOR_DISTANCE

#define DOUBLE_PRECISION 1

#ifdef DOUBLE_PRECISION
#define WTYPE_cuda cuDoubleComplex // wave type for CUDA device
#define WTYPE double complex // wave type
#define STYPE double  // space (coordinate) type
#else
#define WTYPE_cuda cuFloatComplex // wave type for CUDA device
#define WTYPE float complex // wave type
#define STYPE float // space (coordinate) type
#endif // DOUBLE_PRECISION

#define ZERO make_cuDoubleComplex(0,0)


// #define Ix(i,j) i + j * N_sqrt
// #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * N_sqrt
/* #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * DIMS */
#define Ix(i,j,k) k + j * DIMS + i * DIMS * N_sqrt


// TODO check # operations for abs/angle etc
// see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
#define WEIGHT_DIV 4
#define W 8
#define FLOPS_PER_POINT (                           \
             3     /* (u - v) with u,v \in R^3 */ + \
             3+(3+1)*WEIGHT_DIV /* |u| power, sum, power */ +       \
             2*W   /* abs(x_i), angle(x_i) */ +                       \
             1     /* amp/distance */ +                               \
             3     /* phase - direction * distance * 2pi/lambda */ +  \
             1+W   /* a exp(b) */                                     \
             )

#endif
