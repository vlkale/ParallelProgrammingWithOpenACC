#ifdef HAVE_MPI
#include <mpi.h>
#endif 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include <time.h>

#define I2D(ni, i, j) ((i) + (ni)*(j))
#define BLOCKING_MPI 0


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
    for (j=1; j < nj-1; j++) {
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
    double tfac, *temp1_h, *temp2_h;
    int i, j, i2d, istep;
    double temp_bl, temp_br, temp_tl, temp_tr;
    int NUM_THREADS;
    struct timeval tim;
    double start, end;
    double time;
    FILE *fp;
    FILE *fileOfTimings;
    int fd;

    int procID;
    int numProcs;
    int tag1 = 1;
    int tag2 = 2;
    int rcProc;

#ifdef HAVE_MPI
    MPI_Status status;
    MPI_Status statii[4];
    MPI_Request sendLeft;
    MPI_Request sendRight;
    MPI_Request recvLeft;
    MPI_Request recvRight;
    MPI_Request requests[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procID);

    #endif

    int numTasks;
    printf("starting program\n");
    
    if(argc < 6)
    {
        printf("Usage: %s <num GPUs> <ni> <nj> <nstep> <output file> <(numChunks)> \n", argv[0]);
        exit(1);
    }
    NUM_THREADS = atoi(argv[1]);
    ni = atoi(argv[2]);
    nj = atoi(argv[3]);
    nstep = atoi(argv[4]);
    // fp = atos(argv[5]);
    if(argc >= 6)
      numTasks = atoi( argv[6]);
    else
      numTasks = NUM_THREADS;
    int numProcesses = numProcs; // placeholder for divider on nj

    temp1_h = (double *)malloc(sizeof(double)*(ni+2)*(nj/numProcesses + 2));
    temp2_h = (double *)malloc(sizeof(double)*(ni+2)*(nj/numProcesses + 2));
    
    for (j=1; j < nj/numProcesses +1; j++) {
        for (i=1; i < ni+1; i++) {
            i2d = i + (ni+2)*j;
            temp1_h[i2d] = 0.0001f;
        }
    }
    
    temp_bl = 200.0f;
    temp_br = 300.0f;
    temp_tl = 200.0f;
    temp_tr = 300.0f;
    for (i=0; i < ni+2; i++) {

        j = 0;
        i2d = i + (ni+2)*j;
        temp1_h[i2d] = temp_bl + (temp_br-temp_bl)*(double)i/(double)(ni+1);

        j = nj/numProcesses + 1;
        i2d = i + (ni+2)*j;
        temp1_h[i2d] = temp_tl + (temp_tr-temp_tl)*(double)i/(double)(ni+1);
    }

    for (j=0; j < nj/numProcesses +2; j++) {
        i = 0;
        i2d = i + (ni+2)*j;
        temp1_h[i2d] = temp_bl + (temp_tl-temp_bl)*(double)j/(double)(nj+1);

        i = ni+1;
        i2d = i + (ni+2)*j;
	temp1_h[i2d] = temp_br + (temp_tr-temp_br)*(double)j/(double)(nj+1);
    }
    
    memcpy(temp2_h, temp1_h, sizeof(double)*(ni+2)*(nj/numProcesses + 2));

    tfac = 0.2;


    printf(" Completed allocation and initialization %d \n", procID );

    gettimeofday(&tim, NULL);
    start = tim.tv_sec + (tim.tv_usec/1000000.0);

    
    omp_set_num_threads(NUM_THREADS);
    int rows, LDA;
    
    rows = (nj/numProcs)/NUM_THREADS;
    LDA = ni + 2;
    
    int numAvailDevices = omp_get_num_devices();
    
    int numDevices = NUM_THREADS;
    int chosenDev;
    
    printf("Number of MPI procs is : %d\n", numProcs);
    printf("Number of devices available is : %d\n", numAvailDevices);
    printf("Number of devices used is : %d\n", numDevices);
    
    srand(	gettimeofday(&tim, NULL));

    // numTasks = 128;

#pragma omp parallel private(istep, chosenDev) 
{
  double *temp1, *temp2, *temp_tmp;
  int tid = omp_get_thread_num();
  //	chosenDev =  rand()%numDevices;
 
  // TODO: Check how host can also participate in computation
  // acc_set_device_num(tid+1, acc_device_not_host);
  
  
  printf("Rank %d \t ThreadID %d \t Device %d running %ul part of temperature array \n", procID, tid, chosenDev, tid*rows*LDA);
    
  
  for(istep=0; istep < nstep; istep++)
    {
#pragma omp parallel for schedule(dynamic, 1)	
      for (int taskNum  =0 ; taskNum < numTasks; taskNum++)
	{
	  chosenDev =  tid%numDevices; // tid modulo number of devices per MPI process
	  //	  chosenDev = gpu_Scheduler();
	  
	  step_kernel_cpu(ni+2, rows+2, tfac, temp1, temp2, chosenDev);
	  
	}
      
      if (numProcs > 1)
	{
#pragma omp master
	  {  
	    if(!BLOCKING_MPI)
	      {

		printf(" ProcID %d \t send recv nonblocking\n", procID);
		int numRequests=0; 
	       if( procID > 0 )
		 {
		   
		   MPI_Irecv(&temp2[0], LDA, MPI_DOUBLE, procID - 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		   MPI_Isend(&temp2[LDA], LDA, MPI_DOUBLE, procID - 1, 0, MPI_COMM_WORLD, &requests[numRequests++]);
		 }
	       if(procID < numProcs - 1)
		 {
		   MPI_Irecv(&temp2[(rows+1)*LDA], LDA, MPI_DOUBLE, procID + 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		   MPI_Isend(&temp2[(rows)*LDA], LDA, MPI_DOUBLE, procID + 1, 0, MPI_COMM_WORLD, &requests[numRequests++]);
		 }
	      
	       MPI_Waitall(numRequests, requests, statii);
	       
	       //printf("finalizing");

	       //MPI_Finalize();
	       // exit(1);

	      }
	 
	    else
	      {
		if (procID < numProcs - 1)
		  {
		    MPI_Send ( &temp2[rows], LDA, MPI_DOUBLE, procID+1, tag1, MPI_COMM_WORLD ); // send from my row rows 
		    MPI_Recv ( &temp2[rows+1], LDA, MPI_DOUBLE, procID+1, tag2, MPI_COMM_WORLD, &status );	       // recv into my row rows+1
		  } 
		/*update the upper halo to the host except the first device*/
		if(procID > 0 )
		  {
		    MPI_Recv ( &temp2[0], LDA, MPI_DOUBLE, procID-1, tag1, MPI_COMM_WORLD, &status );// recv into my row 0 
		    MPI_Send ( &temp2[1], LDA, MPI_DOUBLE, procID-1, tag2, MPI_COMM_WORLD ); // send from my row 1
		  }
	      }
	  }
	    }

	  
	  // TODO: downsample print to test correctness	  
	  /*make sure another device has already updated the data into host*/
	//#pragma omp master
	//  MPI_Barrier(MPI_COMM_WORLD);
	
        #pragma omp barrier
	
        /* update the upper halo to the other device */ 
	// exchange
	// assume rows per process is a multiple of rows per device , rows per process ;	  

        temp_tmp = temp1;
	temp1 = temp2;
	temp2 = temp_tmp;
	}
 }
      
	/*update the final result to host*/
        // #pragma acc update host(temp1[LDA:rows*LDA])
       

    gettimeofday(&tim, NULL);
    end = tim.tv_sec + (tim.tv_usec/1000000.0);
    printf("Time for computing: %.2f s\n", end-start);

    /* output temp1 to a text file */

    
    /* output temp1 to a binary file */

    //    fd = create(argv[5], 00666);                                                                                                                                                                                                                                                                                 
    //fd = open(argv[5], O_WRONLY);                                                                                                                                                                                                                                                                              
    //write(fd, temp1_h, (ni+2)*(nj+2)*sizeof(double));                                                                                                                                                                                                                                                          
    // close(fd);                                                                                                                                                                                                                                           
    //      pFile = fopen ("myError.csv","w+");
    //    fprintf(pFile, "%s,%d,%d,%d,%d,%d\n", "error", 0.001, 0.001, 0.001, 0.001, 0.001 );
    //     if (pFile!=NULL) fclose (pFile);

    int parStrat = 2; // 2 for MPI + OpenMP
    int opt = 1; // 1 for static schedule 

    fileOfTimings = fopen ("heat2DTimings.out","w+");
    fprintf(fileOfTimings, "%s\t%d\t%d\t%d\t%d\t%f\n", "Heat2D", numProcs, NUM_THREADS, ni, nj, parStrat, opt, time);
    if (fileOfTimings!=NULL) fclose (fileOfTimings);

    fp = fopen(argv[5], "w");   
    fprintf(fp, "%d %d\n", ni, nj);
    for (j=0; j < nj; j++) {                                                                                                                                                                                                                                                   
      for (i=0; i < ni; i++) {                                                                                                                                                                                                                                                                
         fprintf(fp, "%.4f\n", j, i, temp1_h[i + ni*j]);                                                                                                                                       
      }                                                                                                                                                                                                            
    }

    fclose(fp);
#ifdef HAVE_MPI
    rcProc = MPI_Finalize();
    printf("MPI_Finalize return code is %d \n", rcProc);
#endif 
	
} // end main 
