#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

struct __device_builtin__ __builtin_align__(16) derp4 
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
} X;



__device__ uint4 getUint4(uint4* d4) {
	uint4 x;
	x.x = d4->x;
	x.y = d4->y;
	x.z = d4->z;
	x.w = d4->w;
	return x;
}

//__device__ derp4 getDerp4(derp4* d4) {
//	derp4 x;
//	x.x = d4->x;
//	x.y = d4->y;
//	x.z = d4->z;
//	x.w = d4->w;
//	return x;
//}


__global__ void kernel0(uint4* input, uint4* output) {
	int tid = threadIdx.x;
	output[tid] = getUint4(&input[tid]);
}

/* __global__ void kernel1(derp4* input, derp4* output) {
	int tid = threadIdx.x;
	output[tid] = getDerp4(&input[tid]);
} */

int main() {
	int N = 10;
	uint4 X;
	//derp4 X;
	X.x = 1u;
	X.y = 2u;
	X.z = 3u;
	X.w = 4u;
	//derp4* a_h;
	//derp4* a_d;            
	//size_t size = sizeof(derp4);
	uint4* a_h;
	uint4* a_d;
	size_t size = sizeof(derp4);
	cudaMalloc((void **) &a_d, size * N);
	//a_h = (derp4 *)malloc(size*N);
	a_h = (uint4 *)malloc(size*N);
	for (int i = 0; i < N; i++) a_h[i] = X;
	cudaMemcpy(a_d, a_h, size * N, cudaMemcpyHostToDevice);
	
	//derp4* r_h; 
	//derp4* r_d;
	uint4* r_h;
	uint4* r_d;
	cudaMalloc((void **) &r_d, size * N);
	//r_h = (derp4 *)malloc(size*N);
	r_h = (uint4 *)malloc(size * N);


	kernel0 <<<1,N>>> (a_d,r_d);
	cudaMemcpy(r_h, r_d, size * N, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) printf("(%d,%d,%d,%d)\n", r_h[i].x, r_h[i].y, r_h[i].z, r_h[i].w);
	free(r_h);
	free(a_h);
	cudaFree(a_d);
	cudaFree(r_d);
	getchar();
}