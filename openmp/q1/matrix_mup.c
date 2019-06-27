#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N 1024
#define MAX_RAND 100

//	Running times:
//	2 - 48.8 sec
//	4 - 25.6 sec
//	8 - 19.3 sec
//	16 - 19.6 sec

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
	
	#pragma omp parallel for private(j,k)
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
