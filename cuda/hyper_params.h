#ifndef HYPER_PARAMS
#define HYPER_PARAMS

#include "macros.h"

/**
 * Macro's that can be configured with compile flags (-D)
 */

/* #define READ_INPUT */
#define PROJECT_PHASE 0

/* #define N_sqrt 4 */
/* #define N_sqrt 8 */
/* #define N_sqrt 16 */
/* #define N_sqrt 32 */
/* #define N_sqrt 64 */
/* #define N_sqrt 128 */
/* #define N_sqrt 256 */
/* #define N_sqrt 512 */
/* #define N_sqrt 1024 */
#define N_sqrt 1440

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 2 // n datapoints per kernel
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 1 // n kernels per (stream) batch
#endif

/* #define KERNELS_PER_BATCH (STREAM_BATCH_SIZE / KERNEL_BATCH_SIZE) // n kernel calls per stream batch */
// TODO compute optimal batch size as function of N

#define N_STREAMS 8
/* #define STREAM_SIZE (N / N_STREAMS) // datapoints per stream */
/* #define BATCHES_PER_STREAM CEIL(STREAM_SIZE, STREAM_BATCH_SIZE) */
/* #define N_BATCHES (N_STREAMS * BATCHES_PER_STREAM) */

#define WARP_SIZE 32

// BLOCKDIM, BLOCKIDM are independent of N, but max. for the GPU
#if (N_sqrt <= 32)
#define BLOCKDIM 8
#elif (N_sqrt <= 64)
#define BLOCKDIM 16
#elif (N_sqrt <= 128)
#define BLOCKDIM 64 // TODO blocksize >32 causes matrix-bug (in combination with PARALLEL_INTRA_WARP_AGG?)
#elif (N_sqrt <= 256)
#define BLOCKDIM 64
#elif (N_sqrt <= 512)
#define BLOCKDIM 128
#else
#define BLOCKDIM 128
#endif
/* #define BLOCKDIM 1 */

#if (N_sqrt <= 64)
#define GRIDDIM 4
#elif (N_sqrt <= 256)
#define GRIDDIM (BLOCKDIM)
#else
#define GRIDDIM (BLOCKDIM / 2)
#endif
/* #define GRIDDIM (N + BLOCKDIM-1) / BLOCKDIM */

#define CACHE_BATCH 0 // this includes a threads sync and only improves speedup for certain params (BLOCKDIM must be larger than warp size, but many threads may increase sync time(?), and more blocks cause duplicate work)

/* #define CACHE_U 0 // TODO */
/* #define CACHE_V 0 // TODO */

/* #if (BLOCKDIM >= 16) */
/* #define REDUCE_SHARED_MEMORY 4 */
/* #elif (BLOCKDIM >= 32) */
/* #define REDUCE_SHARED_MEMORY 2 // reduce shared memory by this factor */
/* #else */
/* #define REDUCE_SHARED_MEMORY 1 */
/* #endif */

#define REDUCE_SHARED_MEMORY MIN(4, BLOCKDIM) // reduce shared memory by this factor

#define SHARED_MEMORY_LAYOUT 0

#if SHARED_MEMORY_LAYOUT
/* kid: kernel index, tid: thread index */
#define SIdx(kid, tid, size) ((kid) + (tid) * KERNEL_SIZE)
#else
#define SIdx(kid, tid, size) ((tid) + (kid) * (size))
#endif

#define PARALLEL_INTRA_WARP_AGG 0 // TODO reimplement

#if (REDUCE_SHARED_MEMORY > 1 && KERNEL_SIZE >= REDUCE_SHARED_MEMORY)
#define SHARED_MEMORY_SIZE(blockSize) ((KERNEL_SIZE * blockSize) / REDUCE_SHARED_MEMORY)
#else
#define SHARED_MEMORY_SIZE(blockSize) (KERNEL_SIZE * blockSize)
#endif

#endif
