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

#if (N_sqrt <= 32)
#define BLOCKDIMX 4
#elif (N_sqrt <= 64)
#define BLOCKDIMX 8
#elif (N_sqrt <= 128)
#define BLOCKDIMX 16
#else
#define BLOCKDIMX 32
#endif

#if (BLOCKDIMX <= 32)
#define BLOCKDIMY BLOCKDIMX
#else
#define BLOCKDIMY CEIL(1024, BLOCKDIMX)
#endif

#if (N_sqrt <= 64)
#define GRIDDIMX 4
#else
#define GRIDDIMX 8
#endif

#define GRIDDIMY CEIL(GRIDDIMX, 2)

/* #define BLOCKDIMX 8 */
/* #define BLOCKDIMY 8 */
/* #define GRIDDIMX 4 */
/* #define GRIDDIMY 4 */

/* #define RANDOMIZE_SUPERPOSITION_INPUT // true MC, TODO rename & refactor */

#define SQUARE_TARGET_BATCHES

#endif
