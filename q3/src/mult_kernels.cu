#include "kernels.h"

__global__ void multiply(float *a, float *b, int n)
{
	// Determine starting a index for block
	int a_offset = blockIdx.x * 512;

	// Determine a index for thread
	int a_index = a_offset + threadIdx.x;
	
	// Determine B offset by determining the blocks's column
	int blocks_per_row = n / 512;
	int block_column = blockIdx.x % blocks_per_row;

	int b_offset = block_column * 512;

	// Determine b index for the thread;
	int b_index = b_offset + threadIdx.x;

	a[a_index] = a[a_index] * b[b_index];
}

__global__ void reduce(float *a, float *c, int n)
{

	int blocks_per_row = n / (blockDim.x * 2);
	int chunk_column = blockIdx.x % blocks_per_row;
	int chunk_row = blockIdx.x / blocks_per_row;

	// Find the starting a index for the block
	int a_block_start = blockIdx.x * 512;

	int threads;
	int stride;
	int left, right;
	threads = 512 / 2;
	
	for (stride = 1; stride < n; stride *= 2, threads /= 2)
	{
		if ((threadIdx.x / n) < threads)
		{
			left = a_block_start + threadIdx.x * (stride * 2);
			right = left + stride;
			a[left] = a[left] + a[right];
		}	

		__syncthreads();
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		int c_row = chunk_row * blocks_per_row;
		int c_index = c_row + chunk_column;
		c[c_index] = a[a_block_start];
	}
}
