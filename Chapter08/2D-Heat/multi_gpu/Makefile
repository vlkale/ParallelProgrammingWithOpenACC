#include make-crusher.def
#include make-perlmutter.de
include make-cori-gpu.def

all: omp_heat_multi mpi_omp_heat_multi ven_heat_multi 

mpi_omp_heat_multi: mpi_omp_heat_multi.c
	$(CC) $(CCFLAGS) $(OPENMPFLAGS) -lmpi -v mpi_omp_heat_multi.c -o mpi_omp_heat_multi 

omp_heat_multi: omp_heat_multi.c
	$(CC) $(CCFLAGS) $(OPENMPFLAGS) -v omp_heat_multi.c -o omp_heat_multi 

ven_heat_multi:
	$(VENDORCC) -O3 $(venHeatMulti) -o ven_heat_multi
	$(VENDORCC) -O3 --ptx $(heatKERNEL)


hip_heat_multi:
	$(VENDORCC) -O3 hip_heat_multi.c -o hip_heat_multi
	$(VENDORCC) -O3 --ptx $(heatKERNEL)


cuda_heat_multi:
	$(VENDORCC) -O3 cuda_heat_multi.c -o cuda_heat_multi
	$(VENDORCC) -O3 -arch=sm_80 -fmad=false --ptx heat_kernel.cu

acc_heat_multi: acc_heat_multi.c
	$(CC) -O3  $(OPENACCFLAGS) acc_heat_multi.c -o acc_heat_multi

clean:
	rm omp_heat_multi acc_heat_multi cuda_heat_multi mpi_omp_heat_multi


test:
	./omp_heat_multi 1 256 256 200 output-omp-multi.dat
	./omp_heat_multi 2 256 256 200 output-omp-multi.dat
	./omp_heat_multi 3 256 256 200 output-omp-multi.dat
	./omp_heat_multi 4 256 256 200 output-omp-multi.dat



run:
	./omp_heat_multi 1 1024 1024 20000 output-omp-multi.dat
	./omp_heat_multi 2 1024 1024 20000 output-omp-multi.dat
	./omp_heat_multi 4 1024 1024 20000 output-omp-multi.dat
	./mpi_omp_heat_multi 1 1024 1024 20000 output-mpi-omp-multi.dat
	./mpi_omp_heat_multi 2 1024 1024 20000 output-mpi-omp-multi.dat
	./mpi_omp_heat_multi 4 1024 1024 20000 output-mpi-omp-multi.dat
	./ven_heat_multi 1 1024 1024 20000 output-ven-multi.dat
	./ven_heat_multi 2 1024 1024 20000 output-ven-multi.dat
	./ven_heat_multi 4 1024 1024 20000 output-ven-multi.dat
	./acc_heat_multi 2 1024 1024 20000 output-acc-multi.dat
	./cuda_heat_multi 2 1024 1024 20000 output-cuda-multi.dat
