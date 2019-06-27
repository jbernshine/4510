#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "vec_mult.h"

// Fills a vector with random floats in the range [0, 1]
void init_vec(float *vec, int len)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand(time(NULL));
        seeded = 1;
    }
    
    int i;
    for (i = 0; i < len; i++)
    {
        vec[i] = (float) rand() / RAND_MAX;
    }    
}

// Prints the given vector to stdout
void print_vec(const char *label, float *vec, int len)
{
#if PRINT_VECS
    printf("%s", label);
  	
 
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%f ", vec[i]);

		if ((len == (N * N) && i % 16 == 15) || len == N)
		{
			printf("\n");
		}
    }
    printf("\n\n");
#endif
}

// Checks if an error occurred using the given status.
// If so prints the given message and halts.
void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately (makes debugging easier)
    }
}

int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}
