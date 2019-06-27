#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define N 2048
#define NUM_THREADS 16
#define CHUNK 32

/*
	Execution times:
	
	Chunk size		1		   2		     4	         8             16              32		
	static		0.002320	0.002230	 0.002222     0.002178      0.002261        0.002161
	dynamic		0.002662    0.002535     0.002423     0.002376      0.002239        0.002294
	guided		0.002535    0.002429     0.002341     0.002352      0.002303        0.002300
*/

double f (int i)
{
	double x;
	x = (double) i / (double) N;
	return 4.0 / (1.0 + x * x);
}

int main(int argc, char *argv[])
{
	
	double start_time, stop_time;
	double area;
	int i;
	area = f(0) - f(N);
	omp_set_num_threads(NUM_THREADS);

	start_time = omp_get_wtime();
	
	#pragma omp parallel for schedule(guided,CHUNK) reduction(+:area)
	for (i = 1; i <= N / 2; i++)
	{
		area += 4.0 * f(2 * i - 1) + 2 * f(2 * i);
	}

	area /= (3.0 * N);

	stop_time = omp_get_wtime();
	
	printf("Approximation of pi: %13.11f\n", area);
	printf("Execution time: %f\n", stop_time - start_time);
	return EXIT_SUCCESS;
}