CC           = gcc
CC_FLAGS     = -O3 -ffast-math
INCLUDES     = -Iinclude -I/usr/local/include
LD_FLAGS     = -L/usr/local/lib -lrtlsdr -lm

EXE          = fm_rx_cpu
OBJ          = fm_demod_cpu.o main.o

default: $(EXE)

fm_demod_cpu.o: src/fm_demod_cpu.c include/fm_demod_cpu.h
	$(CC) -c -o $@ src/fm_demod_cpu.c $(CC_FLAGS) $(INCLUDES)

main.o: src/main.c include/fm_demod_cpu.h
	$(CC) -c -o $@ src/main.c $(CC_FLAGS) $(INCLUDES)

$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
