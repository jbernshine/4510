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

int main(int argc, char *argv[])
{
	// All CUDA API calls return a status, which we must check
    cudaError_t statusA; // records status of operations on A vec
    cudaError_t statusB; // for B vec
    cudaError_t statusC; // for C vec

	// Allocate host buffers
    size_t size = N * sizeof(float); //size in bytes
	size_t size_a = N * N * sizeof(float); //size in bytes
    float *host_a = (float *) malloc(size_a);
    float *host_b = (float *) malloc(size);
    float *host_c = (float *) malloc(size);

	// Fill host buffers with integers and print them
    init_vec(host_a, (N * N));
    init_vec(host_b, N);
    print_vec("A vector:\n", host_a, (N * N));
    print_vec("B vector:\n", host_b, N);

	// Allocate device buffers
    float *dev_a;
    float *dev_b;
    float *dev_c;
    statusA = cudaMalloc(&dev_a, size_a);
    check_error(statusA, "Error allocating dev buffer A.");
    statusB = cudaMalloc(&dev_b, size);
    check_error(statusB, "Error allocating dev buffer B.");
    statusC = cudaMalloc(&dev_c, size);
    check_error(statusC, "Error allocating dev buffer C.");

	// Transfer the input vectors from host to device
	statusA = cudaMemcpy(dev_a, host_a, size_a, cudaMemcpyHostToDevice);
    statusB = cudaMemcpy(dev_b, host_b, size, cudaMemcpyHostToDevice);

	check_error(statusA, "Error on CPU->GPU cudaMemcpy for A.");
    check_error(statusB, "Error on CPU->GPU cudaMemcpy for B.");
	
	// Determine number of blocks to launch
    // We'll use the max possible number of threads per block
    //int block_size = get_max_block_threads();
    // We'll allocate n threads (one to add each column in the vectors).
    // If n is not evenly divisible by block_size, we'll need to launch one more
    // block to handle the extras (note: not all its threads will be active)
    //int blocks = N / block_size + (N % block_size > 0 ? 1 : 0);
	
	// 1 block with 256 threads
	multiply<<<1, 256>>>(dev_a, dev_b);

	// n blocks with n / 2 threads
	reduce<<<N, (N / 2)>>>(dev_a, dev_c, N);

	// Transfer the resulting C vector back to the host
	statusC = cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
	check_error(statusC, "Error on GPU->CPU cudaMemcpy for C.");

	// Display the result
	print_vec("C vector:\n", host_c, N);
	//print_results(N, block_size, blocks);

	// Clean up memory on host
    free(host_a);
    free(host_b);
    free(host_c);

	// Clean up memory on device
    statusA = cudaFree(dev_a);
    statusB = cudaFree(dev_b);
    statusC = cudaFree(dev_c);
    check_error(statusA, "Error calling cudaFree on dev_a buffer" );
    check_error(statusB, "Error calling cudaFree on dev_b buffer" );
    check_error(statusC, "Error calling cudaFree on dev_c buffer" );

	return EXIT_SUCCESS;
}
