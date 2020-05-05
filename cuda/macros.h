#ifndef MACROS
#define MACROS

/* #define DIV(x,y) ((x + y - 1) / y) // ceil(int, int) */
#define CEIL(x,y) ((x + y - 1) / y) // ceil(int, int)

/* #define DEBUG */
#define Y_TRANSFORM // compute y transform
/* #define Z_TRANSFORM // compute z transform */
#define RANDOM_Y_SPACE true // TODO consider better, non-correlated RNG
#define RANDOM_Z_SPACE true
#define CACHE_BATCH // this includes a threads sync and only improves speedup for certain params (BLOCKDIM must be larger than warp size, but many threads may increase sync time(?), and more blocks cause duplicate work)
/* #define PINNED_MEM // use cudaMallocHost over cudaMalloc // disable if in case of host memory issues // TODO speedup > 1 in absense of kernal invocation and otherwise < 1 */
#define REDUCE_SHARED_MEMORY 2 // reduce shared memory by this factor
#define PARALLEL_INTRA_WARP_AGG true

#define DIMS 3
// TODO use N,M
/* #define N_sqrt 8 */
/* #define N_sqrt 32 */
#define N_sqrt 64
/* #define N_sqrt 128 */
/* #define N_sqrt 512 */
/* #define N_sqrt 1024 */
#define N (N_sqrt * N_sqrt)
#define N2 (N_sqrt * N_sqrt)
/* #define STREAM_BATCH_SIZE 8 // n datapoints per stream // stream batch size // TODO rename to STREAM_BATCH_SIZE? */
/* #define KERNEL_BATCH_SIZE 8 // n datapoints per kernel, must be <= STREAM_BATCH_SIZE */

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 8 // n datapoints per kernel
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 1 // n kernels per (stream) batch
#endif

/* #define KERNELS_PER_BATCH (STREAM_BATCH_SIZE / KERNEL_BATCH_SIZE) // n kernel calls per stream batch */
// TODO compute optimal batch size as function of N

#define N_STREAMS 4 // TODO single stream results in incorrect output
/* #define STREAM_SIZE (N / N_STREAMS) // datapoints per stream */
/* #define BATCHES_PER_STREAM CEIL(STREAM_SIZE, STREAM_BATCH_SIZE) */
/* #define N_BATCHES (N_STREAMS * BATCHES_PER_STREAM) */

#define MAX_INPUT_SIZE 0 // TODO, specific for GPU

#define ARBITRARY_PHASE 0.4912 // used in superposition::per_thread

#define WARP_SIZE 32
// BLOCKDIM, BLOCKIDM are independent of N, but max. for the GPU
#if (N_sqrt <= 32)
#define BLOCKDIM 1
#elif (N_sqrt <= 64)
#define BLOCKDIM 16
#elif (N_sqrt <= 128)
#define BLOCKDIM 64
#elif (N_sqrt <= 512)
#define BLOCKDIM 128
#else
#define BLOCKDIM 256
#endif
/* #define BLOCKDIM 8 */
/* #define BLOCKDIM 128 */
/* #define BLOCKDIM 256 */
/* #define GRIDDIM 256 */
/* #define GRIDDIM (2 * BLOCKDIM) */
#define GRIDDIM (2 * BLOCKDIM)
/* #define GRIDDIM (N + BLOCKDIM-1) / BLOCKDIM */

#if (REDUCE_SHARED_MEMORY > 1 && KERNEL_SIZE >= REDUCE_SHARED_MEMORY)
#define SHARED_MEMORY_SIZE(blockSize) ((KERNEL_SIZE * blockSize) / REDUCE_SHARED_MEMORY)
#else
#define SHARED_MEMORY_SIZE(blockSize) (KERNEL_SIZE * blockSize)
#endif


/* #define N_PER_THREAD (N / GRIDDIM / BLOCKDIM) // for input (x), thus independent of batches */
// the value N_PER_THREAD is used implicitly in gridDim.x

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
/* #define ABS(x) (cuCabs(x)) */
#define STYPE double  // space (coordinate) type
#else
/* #define WTYPE_cuda cuFloatComplex // wave type for CUDA device */
#define WTYPE cuFloatComplex  // wave type
#define STYPE float // space (coordinate) type
#endif // DOUBLE_PRECISION



#define ZERO make_cuDoubleComplex(0,0)
/* #define VOL(type, x) *((type *) &x) */


// #define Ix(i,j) i + j * N_sqrt
// #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * N_sqrt
/* #define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * DIMS */
#define I_(i,j) (j + (i) * N_sqrt)
#define Ix(i,j,k) (k + (j) * DIMS + (i) * DIMS * N_sqrt)

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
