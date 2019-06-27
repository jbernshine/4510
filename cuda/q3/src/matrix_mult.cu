#include <stdio.h>
#include <stdlib.h>
#include "vec_mult.h"
#include "utils.h"
#include "kernels.h"

void print_results(int n, int block_size, int blocks)
{
    printf("-------\n");
    printf("Results\n");
    printf("-------\n");
    
    // Print the resulting C vector and the timing stats
    printf("Data size: 2^%d = %d\n", (int) log2f(n), n);
    printf("blocks_size: %d\n", block_size);
    printf("blocks: %d\n", blocks);
    printf("\n");

}

void add_C(float *host_c_buffer, float *host_c_actual, int blocks, int blocks_per_row)
{
	int rows = blocks / blocks_per_row;
	int currBlock = 0;
	int i,j;

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < blocks_per_row; j++)
		{
			host_c_actual[i] = host_c_actual[i] + host_c_buffer[currBlock];
			currBlock++;
		}
	}
}

int main(int argc, char *argv[])
{

	if (argc != 2)
	{
		printf("Usage: ./vec_mult <n , where n is a multiple of 512>\n");
		exit(1);
	}

	int n = atoi(argv[1]);
	printf("%d\n", n);
	if (n % 512 != 0)
	{
		printf("Error: n must be a multiple of 512\n");
		exit(1);
	}

	// All CUDA API calls return a status, which we must check
    cudaError_t statusA; // records status of operations on A vec
    cudaError_t statusB; // for B vec
    cudaError_t statusC; // for C vec

	// Allocate host buffers
    
	size_t size_a = n * n * sizeof(float); //size in bytes
	size_t size_b = n * sizeof(float); //size in bytes
	int blocks_per_row = n / 512;
	size_t size_c = blocks_per_row * n * sizeof(float);

    float *host_a = (float *) malloc(size_a);
    float *host_b = (float *) malloc(size_b);
    float *host_c_buffer = (float *) malloc(size_c);
	float *host_c_actual	= (float *) malloc(size_b);

	// Fill host buffers with integers and print them
    init_vec(host_a, (n * n));
    init_vec(host_b, n);
    //print_vec("A vector:\n", host_a, (n * n), n);
    //print_vec("B vector:\n", host_b, n, n);

	// Allocate device buffers
    float *dev_a;
    float *dev_b;
    float *dev_c;
    statusA = cudaMalloc(&dev_a, size_a);
    check_error(statusA, "Error allocating dev buffer A.");
    statusB = cudaMalloc(&dev_b, size_b);
    check_error(statusB, "Error allocating dev buffer B.");

	// Blocks per row is needed for c buffer
	
    statusC = cudaMalloc(&dev_c, size_c);
    check_error(statusC, "Error allocating dev buffer C.");

	// Transfer the input vectors from host to device
	statusA = cudaMemcpy(dev_a, host_a, size_a, cudaMemcpyHostToDevice);
    statusB = cudaMemcpy(dev_b, host_b, size_b, cudaMemcpyHostToDevice);

	check_error(statusA, "Error on CPU->GPU cudaMemcpy for A.");
    check_error(statusB, "Error on CPU->GPU cudaMemcpy for B.");
	
	// Determine number of blocks to launch
    // We'll use the max possible number of threads per block
    int block_size = 512;
    // We'll allocate n threads (one to add each column in the vectors).
    // If n is not evenly divisible by block_size, we'll need to launch one more
    // block to handle the extras (note: not all its threads will be active)
    int blocks = n * n / block_size;
	
	multiply<<<blocks, block_size>>>(dev_a, dev_b, n);
	
	//statusA = cudaMemcpy(host_a, dev_a, size_a, cudaMemcpyDeviceToHost);
	//check_error(statusA, "Error on GPU->CPU cudaMemcpy for A.");
	//print_vec("A vector:\n", host_a, (n * n), n);

	int blocks_reduce = n * blocks_per_row;
	
	printf("%d %d\n", blocks_reduce, (block_size / 2));
	// n blocks with n / 2 threads
	reduce<<<blocks_reduce, (block_size / 2)>>>(dev_a, dev_c, n);

	// Transfer the resulting C vector back to the host
	statusC = cudaMemcpy(host_c_buffer, dev_c, size_c, cudaMemcpyDeviceToHost);
	check_error(statusC, "Error on GPU->CPU cudaMemcpy for C.");
	
	//print_vec("C vector:\n", host_c_buffer, (blocks_per_row * n), n);
	// Create a single column of C
	add_C(host_c_buffer, host_c_actual, blocks, blocks_per_row);

	// Display the result
	print_vec("C vector:\n", host_c_actual, n, n);
	//print_results(N, block_size, blocks);

	// Clean up memory on host
    free(host_a);
    free(host_b);
    free(host_c_buffer);
	free(host_c_actual);

	// Clean up memory on device
    statusA = cudaFree(dev_a);
    statusB = cudaFree(dev_b);
    statusC = cudaFree(dev_c);
    check_error(statusA, "Error calling cudaFree on dev_a buffer" );
    check_error(statusB, "Error calling cudaFree on dev_b buffer" );
    check_error(statusC, "Error calling cudaFree on dev_c buffer" );

	return EXIT_SUCCESS;
}
