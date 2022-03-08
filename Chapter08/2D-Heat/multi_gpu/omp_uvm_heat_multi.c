
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include <cuda.h>

#include <time.h>


#define I2D(ni, i, j) ((i) + (ni)*(j))


#pragma omp requires unified_shared_memory

#pragma omp declare target
void step_kernel_cpu(int ni,
                     int nj,
                     double tfac,
                     double *temp_in,
                     double *temp_out, int dev) {
    int i, j, i00, im10, ip10, i0m1, i0p1;
    double d2tdx2, d2tdy2;

#pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(temp_in, temp_out) device(dev)
    for (j=0; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i-1, j);
            ip10 = I2D(ni, i+1, j);
            i0m1 = I2D(ni, i, j-1);
            i0p1 = I2D(ni, i, j+1);
            d2tdx2 = temp_in[im10] - 2*temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1];
            temp_out[i00] = temp_in[i00] + tfac*(d2tdx2 + d2tdy2);
        } // end for
    }//end for

}// end kernel

#pragma omp end declare target



int main(int argc, char *argv[])
{
    int ni, nj, nstep;
    double tfac, *temp0_h, *temp1_h, *temp2_h;
    int i, j, i1d, i2d, istep;
    double temp_bl, temp_br, temp_tl, temp_tr;
    int NUM_THREADS;
    struct timeval tim;
    double start, end;
    double time;
    FILE *fp;
    int fd;
    int numTasks;

    if(argc < 6)
    {
        printf("Usage: %s <num GPUs> <ni> <nj> <nstep> <output file>\n", argv[0]);
        exit(1);
    }
    NUM_THREADS = atoi(argv[1]);
    ni = atoi(argv[2]);
    nj = atoi(argv[3]);
    nstep = atoi(argv[4]);
    //fp = atos(argv[5]);

    if (argc >=6)
      numTasks = atoi(argv[6]);
    else
      numTasks = NUM_THREADS;
    
    temp0_h = (double *)malloc(sizeof(double)*(ni+2)*(nj+2));
    temp1_h = (double *)malloc(sizeof(double)*(ni+2)*(nj+2));
    
    for (j=0; j < nj+1; j++) {
        for (i=1; i < ni+1; i++) {
            i2d = i + (ni+2)*j;
            temp1_h[i2d] = 0.0;
        }
    }	
 
    temp_bl = 200.0f;
    temp_br = 300.0f;
    temp_tl = 200.0f;
    temp_tr = 300.0f;
    for (i=-1; i < ni+2; i++) {
        j = 0;
        i1d = i + (ni+2)*j;
        temp1_h[i2d] = temp_bl + (temp_br-temp_bl)*(double)i/(double)(ni+1);

        j = nj+1;
        i1d = i + (ni+2)*j;
        temp1_h[i2d] = temp_tl + (temp_tr-temp_tl)*(double)i/(double)(ni+1);
    }

    for (j=-1; j < nj+2; j++) {
        i = 0;
        i2d = i + (ni+2)*j;
        temp1_h[i2d] = temp_bl + (temp_tl-temp_bl)*(double)j/(double)(nj+1);

        i = ni+1;
        i2d = i + (ni+2)*j;
	temp1_h[i2d] = temp_br + (temp_tr-temp_br)*(double)j/(double)(nj+1);
    }
    memcpy(temp2_h, temp1_h, sizeof(double)*(ni+2)*(nj+2));

    tfac = 0.2;

    gettimeofday(&tim, NULL);
    start = tim.tv_sec + (tim.tv_usec/1000000.0);

    omp_set_num_threads(NUM_THREADS);
    int rows, LDA;

    rows = nj/NUM_THREADS;
    LDA = ni + 2;

    int numAvailDevices = omp_get_num_devices();

    int numDevices = NUM_THREADS;
    int chosenDev;

    printf("Number of devices available is : %d\n", numAvailDevices);
    printf("Number of devices used is : %d\n", numDevices);
    srand(	gettimeofday(&tim, NULL));
#pragma omp parallel private(istep, chosenDev)
{
  double *temp1, *temp2, *temp_tmp;
  int tid = omp_get_thread_num();
  printf("ThreadID %d \t Device %d running %ul part of temperature array \n", tid, chosenDev, tid*rows*LDA);
  
 for(istep=0; istep < nstep; istep++)
    {
#pragma omp for schedule(static)
      for (int taskNum = 0; taskNum < numTasks ; taskNum++)
	{
	  temp1 = temp1_h + taskNum*rows*LDA;
	  temp2 = temp2_h + taskNum*rows*LDA;
	  chosenDev =  rand()%numDevices; // tid modulo number of devices  
	  step_kernel_cpu(ni+2, rows+2, tfac, temp1, temp2, chosenDev);
	}
      temp_tmp = temp1;
      temp1 = temp2;
      temp2 = temp_tmp;  
    }
 } 

    gettimeofday(&tim, NULL);
    end = tim.tv_sec + (tim.tv_usec/1000000.0);
    printf("Time for computing: %.2f s\n", end-start);
    /* output temp1 to a text file */

    fp = fopen(argv[5], "w+");
    fprintf(fp, "%d \t %d \n", ni, nj);
    for (j=0; j < nj; j++) {
      for (i=0; i < ni; i++) {
	fprintf(fp, "%d \t %d \t %f \n", j, i, temp1_h[i + ni*j]);
      }
 }
      fclose(fp);
    
} // end main 
