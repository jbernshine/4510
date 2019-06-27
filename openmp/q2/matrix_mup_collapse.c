#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N 1024
#define MAX_RAND 100

//	Running times:
//	2 - 84.5 sec
//	4 - 77.0 sec
//	8 - 72.3 sec
//	16 - 71.4 sec
//
//  Performance degraded using collapse compared to parallelizing the outer for loop in the first program.
//  Using collapse(2) generates more forks and joins, dim^2. With a limited number of cores, that means a lot more
//  swapping out of threads to do processing, increasing overhead.

int A[N][N];
int B[N][N];
int C[N][N];

double start_time; // use these for timing
double stop_time;

void printResults()
{
	printf("Running time: %f\n", stop_time - start_time);

}


void init_Vec()
{
	int i,j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = rand() % MAX_RAND;
			B[i][j] = rand() % MAX_RAND;
		}

	}
}

int main(int argc, char *argv[])
{
	
	int i,j,k;

	// Fill A and B vec
	init_Vec();

	start_time = omp_get_wtime();
	
	#pragma omp parallel for collapse(2)
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i][j] = 0;
			for (k = 0; k < N; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	
	stop_time = omp_get_wtime();

	printResults();

	return EXIT_SUCCESS;
}
