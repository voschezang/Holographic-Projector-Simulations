#include <math.h>
#include <stdio.h>
#include <complex.h>
#include <cuComplex.h>

#define N_sqrt 1024
#define N (N_sqrt * N_sqrt)
#define DIMS 3
/* #define THREADS_PER_BLOCK 512 */
#define THREADS_PER_BLOCK 16
#define LAMBDA 0.6328e-6  // wavelength in vacuum: 632.8 nm (HeNe laser)
#define SCALE 1 / LAMBDA
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
#define Ix(i,j,k) i + j * N_sqrt + k * N_sqrt * DIMS

__device__ double angle(cuDoubleComplex  z) {
  return atan2(cuCreal(z), cuCimag(z));
}

__device__ cuDoubleComplex polar(double r, double i) {
  // return the complex number r * exp(i * I)
  cuDoubleComplex res;
  sincos(i, &res.x, &res.y);
  return cuCmul(make_cuDoubleComplex(r, 0), res);
}

__device__ WTYPE_cuda K(size_t i, WTYPE_cuda x_i, STYPE *u, STYPE *v) {
  // TODO unpack input to u1,u2,3 v1,v2,v3
  // TODO consider unguarded functions
  size_t j = i * DIMS; // TODO use struct?
  double amp, phase, distance;
  int direction = 1;
  distance = norm3d(v[j] - u[j], v[j+1] - u[j+1], v[j+2] - u[j+2]);
  amp = cuCabs(x_i);
  phase = angle(x_i);
  amp /= distance;
  phase -= direction * 2 * M_PI * distance / LAMBDA;
  return polar(amp, phase);
}

// __global__ void kernel2(WTYPE_cuda *x, STYPE *u, WTYPE_cuda  *y, STYPE *v)
// {
//   __shared__ int temp[THREADS_PER_BLOCK];
// 	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

//   //

//   double result = 0;
//   temp[threadIdx.x] = result;
// }
__global__ void kernel1(WTYPE_cuda *x, STYPE *u, WTYPE_cuda  *y, STYPE *v)
{
  // Single kernel, used in y_i = \sum_j K(y_i,x_j)
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  WTYPE_cuda sum = ZERO;

  for(int n = 0; n < N; n++)
    sum = cuCadd(K(n, x[n], u, v), sum);

  y[i] = sum;
}


int main() {
  printf("Init\n");
	size_t size = N * sizeof( WTYPE );

  // host
  WTYPE *x = (WTYPE *) malloc(size);
  WTYPE *y = (WTYPE *) malloc(size);

  STYPE *u = (STYPE *) malloc(DIMS  * N * sizeof(WTYPE));
  STYPE *v = (STYPE *) malloc(DIMS  * N * sizeof(STYPE));
  // STYPE *u[DIMS  * N * sizeof(WTYPE)]
  // STYPE *v[DIMS  * N * sizeof(WTYPE)]

  // device
	WTYPE_cuda *d_x, *d_y;
	STYPE *d_u, *d_v;
	cudaMalloc( (void **) &d_x, N * sizeof(WTYPE_cuda) );
	cudaMalloc( (void **) &d_y, N * sizeof(WTYPE_cuda) );

	cudaMalloc( (void **) &d_u, N * sizeof(STYPE) );

  cudaMalloc( (void **) &d_v, N * sizeof(STYPE) );

  const double width = 0.005;
  // const double dS = SCALE * 7 * 1e-6; // actually dS^1/DIMS
  const double dS = width * SCALE / N_sqrt; // actually dS^1/DIMS
  const double offset = 0.5 * N_sqrt * dS;
	for(int i = 0; i < N_sqrt; i++ ) {
    for(int j = 0; j < N_sqrt; j++ ) {
      size_t idx = i + j * N_sqrt;
      x[idx] = 1;
      if (i == N_sqrt / 2) {
        x[idx] = 1;
      }

      u[Ix(i,j,0)] = i * dS - offset;
      u[Ix(i,j,1)] = j * dS - offset;
      u[Ix(i,j,2)] = 0;

      v[Ix(i,j,0)] = i * dS - offset;
      v[Ix(i,j,1)] = j * dS - offset;
      v[Ix(i,j,2)] = -0.02;
    }
  }

  printf("loop\n");

  int m = (N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
  // cudaMemcpy()
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < THREADS_PER_BLOCK; j++) {
      // split x over threads
    }
  }

  // for (int i = 0; i < N; i++) {
  //   kernel2<<< m, THREADS_PER_BLOCK >>>( y[i], v[i], v[i+1], v[i+2] );
  // }

	cudaMemcpy( d_x, y, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_y, y, size, cudaMemcpyHostToDevice );

	cudaMemcpy( d_u, u, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_v, v, size, cudaMemcpyHostToDevice );

	kernel1<<< m, THREADS_PER_BLOCK >>>( d_x,d_u, d_y,d_v );
	cudaMemcpy( y, d_y, size, cudaMemcpyDeviceToHost );

  int k = N / 2;
	printf( "|x_i| = %0.2f, |y_i| = %0.2f\n", cabs(x[k]), cabs(y[k]) );
	/* printf( "c[%d] = %d\n",N-1, c[N-1] ); */

	free(x);
	free(y);
	cudaFree( d_x );
	cudaFree( d_y );

	return 0;
}
