export PATH  := /usr/local/cuda-12.2/bin:$(PATH)

NVCC         = nvcc
NVCC_FLAGS   = -O3 --use_fast_math -arch=sm_87
INCLUDES     = -Iinclude -I/usr/local/include
LD_FLAGS     = -lcudart -L/usr/local/lib -lrtlsdr -lm

EXE          = fm_rx_gpu_naive
OBJ          = fm_demod_gpu_naive.o main.o

default: $(EXE)

fm_demod_gpu_naive.o: src/fm_demod_gpu_naive.cu include/fm_demod_gpu_naive.cuh
	$(NVCC) -c -o $@ src/fm_demod_gpu_naive.cu $(NVCC_FLAGS) $(INCLUDES)

main.o: src/main.c include/fm_demod_gpu_naive.cuh
	$(NVCC) -c -o $@ src/main.c $(NVCC_FLAGS) $(INCLUDES)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
