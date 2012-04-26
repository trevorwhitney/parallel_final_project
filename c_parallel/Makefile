EXECS = my_pblas

all: $(EXECS)

my_pblas: my_pblas.c
	mpicc -o my_pblas my_pblas.c

clean:
	rm $(EXECS)
