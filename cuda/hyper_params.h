#ifndef HYPER_PARAMS
#define HYPER_PARAMS

#include "macros.h"

/**
 * Macro's that can be configured with compile flags (-D)
 */

/* #define READ_INPUT */
#define PROJECT_PHASE 1

/* #define N_sqrt 4 */
/* #define N_sqrt 8 */
/* #define N_sqrt 16 */
/* #define N_sqrt 32 */
/* #define N_sqrt 64 */
/* #define N_sqrt 128 */
/* #define N_sqrt 256 */
#define N_sqrt 512
/* #define N_sqrt 1024 */
/* #define N_sqrt 1440 */

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 16 // n datapoints per kernel
#endif

#ifndef N_STREAMS
#define N_STREAMS 16
#endif


// BLOCKDIM, BLOCKIDM are independent of N, but max. for the GPU
#if (N_sqrt <= 32)
#define BLOCKDIM 4
#elif (N_sqrt <= 64)
#define BLOCKDIM 8
#elif (N_sqrt <= 128)
#define BLOCKDIM 16
#elif (N_sqrt <= 256)
#define BLOCKDIM 32
/* #elif (N_sqrt <= 512) */
/* #define BLOCKDIM 64 */
#else
#define BLOCKDIM 64
#endif
/* #define BLOCKDIM 1 */

#if (N_sqrt <= 64)
#define GRIDDIM 4
#elif (N_sqrt <= 256)
#define GRIDDIM (BLOCKDIM)
#else
#define GRIDDIM (BLOCKDIM * 2)
#endif
/* #define GRIDDIM (N + BLOCKDIM-1) / BLOCKDIM */

#endif
