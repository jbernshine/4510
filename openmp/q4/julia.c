#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "qdbmp.h"
#include "julia.h"
#include "mpi.h"

void toRGB(unsigned int val,
           unsigned char *r, unsigned char *g, unsigned char *b)
{
    *r = (val & 0xff);
    val >>= 8;
    *b = (val & 0xff);
    val >>= 8;
    *g = (val & 0xff);
}

unsigned int sum_array(unsigned int *array, int len)
{
    unsigned int total = 0;
    for (int i = 0; i < len; i++)
    {
        total += array[i];
    }

    return total;
}

void add_arrays(unsigned int *dest, unsigned int *src, int len)
{
    for (int i = 0; i < len; i++)
    {
        dest[i] += src[i];
    }
}

void hist_eq(unsigned int *data, unsigned int *hist)
{
    printf("Performing Histogram Equalization...\n");
    fflush(stdout);
    
    unsigned int total = sum_array(hist, MAX_ITER);
    unsigned int val;
    
    float cache[MAX_ITER];
    float hue = 0.0;
    for (unsigned int i = 0; i < MAX_ITER; i++)
    {
        cache[i] = hue;
        hue += (float) hist[i] / total;
    }
    
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            val = data[y * WIDTH + x];

            if (val < MAX_ITER)
            {
                hue = cache[val];
            }
            else
            {
                hue = cache[MAX_ITER - 1];
                for (unsigned int i = MAX_ITER; i < val; i++)
                {
                    hue += (float) hist[i] / total;
                }
            }
            
            data[y * WIDTH + x] = (unsigned int) (hue * MAX_COLOUR);
        }
    }

    printf("Done.\n");
    fflush(stdout);
}

void write_bmp(unsigned int *data, char *fname)
{
    printf("Writing BMP file...\n");
    fflush(stdout);
    
    BMP *bmp = BMP_Create((UINT) WIDTH, (UINT) HEIGHT, (USHORT) DEPTH);
    unsigned char r, g, b;
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            toRGB(data[y * WIDTH + x], &r, &g, &b);
            BMP_SetPixelRGB(bmp, (UINT) x, (UINT) y,
                            (UCHAR) r, (UCHAR) g, (UCHAR) b);
        }
    }

    BMP_WriteFile(bmp, FNAME);
    BMP_Free(bmp);

    printf("Done.\n");
    fflush(stdout);
}

unsigned int julia_iters(float complex z)
{
    unsigned int iter = 0;
    while (fabsf(cimag(z)) < 50 && iter < MAX_ITER)
    {
        z = C * csin(z);
        iter++;
    }

    return iter;
}

void compute_row(int row, int local_row, unsigned int *data, unsigned int *hist)
{
    float complex z;
    float series_row;
    float series_col;
    unsigned int iters;
 
    for (int col = 0; col < WIDTH; col++)
    {
        series_row = row - HEIGHT / 2;
        series_col = col - WIDTH / 2;
        z = series_col / RES_FACTOR + (I / RES_FACTOR) * series_row;
        z *= SCALE;
        iters = julia_iters(z);
        data[local_row * WIDTH + col] = iters;
        hist[iters]++;
    }
}

void worker(int my_rank, int num_workers, int rows_per_proc)
{
    unsigned int data[rows_per_proc * WIDTH];
    unsigned int hist[MAX_ITER] = {0};

    int start_row = (my_rank - 1) * rows_per_proc;
    int stop_row = start_row + rows_per_proc;

    for (int row = start_row; row < stop_row; row++)
    {
        compute_row(row, row - start_row, data, hist);
    }
    
    MPI_Send(
        data,
        rows_per_proc * WIDTH,
        MPI_UNSIGNED,
        0,
        0,
        MPI_COMM_WORLD
        );

    MPI_Send(
        hist,
        MAX_ITER,
        MPI_UNSIGNED,
        0,
        0,
        MPI_COMM_WORLD
        );

    printf("Process %d done.\n", my_rank);
    fflush(stdout);
}

void root(int num_workers, int rows_per_proc)
{
    printf("Beginning julia set computation...\n");
    
    unsigned int data[HEIGHT * WIDTH];
    unsigned int hist[MAX_ITER] = {0};
    unsigned int temp_hist[MAX_ITER] = {0};
    
    for (int rank = 1; rank <= num_workers; rank++)
    {
        //receive data
        MPI_Status status;
        int offset = (rank - 1) * rows_per_proc * WIDTH;
        MPI_Recv(
            data + offset,
            rows_per_proc * WIDTH,
            MPI_UNSIGNED,
            rank,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &status
            );

        //receive hist
        MPI_Recv(
            temp_hist,
            MAX_ITER,
            MPI_UNSIGNED,
            rank,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &status
            );
        add_arrays(hist, temp_hist, MAX_ITER);
    }
    
    hist_eq(data, hist);
    write_bmp(data, FNAME);
    
    printf("Done.\n");
}

int main(int argc, char *argv[])
{
    int my_rank;
    int num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //grab this process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //grab the total num of processes
    
    int num_workers = num_procs - 1;
    int rows_per_proc = HEIGHT / num_workers;
    
    if (!my_rank)
    {
        root(num_workers, rows_per_proc);
    }

    else
    {
        worker(my_rank, num_workers, rows_per_proc);
    }
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}
