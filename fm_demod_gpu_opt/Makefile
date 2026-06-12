export PATH  := /usr/local/cuda-12.2/bin:$(PATH)

NVCC         = nvcc
NVCC_FLAGS   = -O3 --use_fast_math -arch=sm_87
INCLUDES     = -Iinclude -I/usr/local/include
LD_FLAGS     = -lcudart -L/usr/local/lib -lrtlsdr -lm

EXE          = fm_rx_gpu_opt
OBJ          = fm_demod_gpu.o main.o

default: $(EXE)

fm_demod_gpu.o: src/fm_demod_gpu.cu include/fm_demod_gpu.cuh
	$(NVCC) -c -o $@ src/fm_demod_gpu.cu $(NVCC_FLAGS) $(INCLUDES)

main.o: src/main.c include/fm_demod_gpu.cuh
	$(NVCC) -c -o $@ src/main.c $(NVCC_FLAGS) $(INCLUDES)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
