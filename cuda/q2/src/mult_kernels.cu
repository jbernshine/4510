#include "kernels.h"

__global__ void multiply(float *a, float *b)
{

	// determine this thread's global id
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	// determine the right b value to multiply (column)
	int b_id = global_id % 16;

	a[global_id] = a[global_id] * b[b_id];
}

__global__ void reduce(float *a, float *c, int n)
{
	// Find the starting a index for the block
	int block_start = blockIdx.x * n;

	int threads;
	int stride;
	int left, right;
	threads = n / 2;
	
	for (stride = 1; stride < n; stride *= 2, threads /= 2)
	{
		if ((threadIdx.x / n) < threads)
		{
			left = block_start + threadIdx.x * (stride * 2);;
			right = left + stride;
			a[left] = a[left] + a[right];
		}	

		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		c[blockIdx.x] = a[block_start];
	}
}
