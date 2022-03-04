


CLANG= clang
CLANGP= clang++
CC= $(CLANG)
CXX = $(CLANGP)

all: omp_heat_multi cuda_heat_multi acc_heat_multi


omp_heat_multi: omp_heat_multi.c
	$(CC) -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_80 -fopenmp-version=51 omp_heat_multi.c -o omp_heat_multi


cuda_heat_multi: cuda_heat_multi.c
	nvcc -O3 cuda_heat_multi.c -o cuda_heat_multi -lcuda
	nvcc -O3 -arch=sm_80 -fmad=false --ptx heat_kernel.cu


acc_heat_multi: acc_heat_multi.c
	pgcc -O3 -mp -acc -ta=ampere:cc80,nofma -Minfo=accel acc_heat_multi.c -o acc_heat_multi

clean:
	rm omp_heat_multi acc_heat_multi cuda_heat_multi

run:
	./omp_heat_multi 4 1024 1024 20000 output-omp-multi.dat
	./acc_heat_multi 2 1024 1024 20000 output-acc-multi.dat
	./cuda_heat_multi 2 1024 1024 20000 output-cuda-multi.dat
