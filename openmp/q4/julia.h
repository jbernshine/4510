#ifndef _JULIA_H
#define _JULIA_H

#define WIDTH 800
#define HEIGHT 600
#define SCALE 4.0
#define FNAME "out.bmp"
#define C (-0.839 + 0.4I)

#define DEPTH 24
#define MAX_COLOUR ((1L << DEPTH) - 1)
#define MAX_ITER ((1L << 16) - 1)
#define RES_FACTOR (sqrt(pow(WIDTH / 2, 2) + pow(HEIGHT / 2, 2)))
#define PI 3.14159265358979323846

void toRGB(unsigned int val,
           unsigned char *r, unsigned char *g, unsigned char *b);
unsigned int sum_array(unsigned int *array, int len);
void add_arrays(unsigned int *dest, unsigned int *src, int len);
void hist_eq(unsigned int *data, unsigned int *hist);
void write_bmp(unsigned int *data, char *fname);
unsigned int julia_iters(float complex z);
void compute_row(int row, int local_row, unsigned int *data, unsigned int *hist);
void root(int num_workers, int rows_per_proc);
void worker(int my_rank, int num_workers, int rows_per_proc);

#endif
