#ifndef HYPER_PARAMS
#define HYPER_PARAMS

/* #define DEBUG */
/* #define Z_TRANSFORM // compute z transform */
#define CACHE_BATCH 1 // this includes a threads sync and only improves speedup for certain params (BLOCKDIM must be larger than warp size, but many threads may increase sync time(?), and more blocks cause duplicate work)
#define REDUCE_SHARED_MEMORY 2 // reduce shared memory by this factor
#define PARALLEL_INTRA_WARP_AGG 1

#define DIMS 3
// TODO use N,M
/* #define N_sqrt 8 */
/* #define N_sqrt 32 */
/* #define N_sqrt 64 */
/* #define N_sqrt 128 */
#define N_sqrt 256
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





#endif
