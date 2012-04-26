EXECS = part2 driver

all: $(EXECS)

part2: part2.c
	gcc -o part2 part2.c

driver: driver.c my_dgemm.h
	gcc -lcblas -lblas -o driver driver.c

clean:
	rm $(EXECS)

optimizations: driver.c my_dgemm.h
	gcc -O1 -lcblas -lblas -o driver_O1 driver.c
	gcc -O2 -lcblas -lblas -o driver_O2 driver.c
	gcc -O3 -lcblas -lblas -o driver_O3 driver.c
	gcc -O3 -funroll-loops -lcblas -lblas -o driver_O3_funroll driver.c

tiles: driver.c my_dgemm.h
	gcc -lcblas -lblas -o tiles driver.c

test_steps: test_steps.c my_dgemm.h
	touch tests/not_empty
	rm tests/*
	gcc -lcblas -lblas -o tests/test_steps test_steps.c

final: final.c my_dgemm.h
	gcc -lcblas -lblas -o O0/final_O0 final.c
	gcc -funroll-loops -lcblas -lblas -o O0_funroll/final_00_funroll final.c
	gcc -O1 -lcblas -lblas -o O1/final_O1 final.c
	gcc -O1 -funroll-loops -lcblas -lblas -o O1_funroll/final_O1_funroll final.c
	gcc -O2 -lcblas -lblas -o O2/final_O2 final.c
	gcc -O2 -funroll-loops -lcblas -lblas -o O2_funroll/final_O2_funroll final.c
	gcc -O3 -lcblas -lblas -o O3/final_O3 final.c
	gcc -O3 -funroll-loops -lcblas -lblas -o O3_funroll/final_O3_funroll final.c