/***********************************************************
 * mv_pblas.c -- parallel matrix - vector multiplication   *
 *                                                         *
 * Compute  y = A * x  by means of the                     *
 * Parallel Basic Linear Algebra Subroutines (PBLAS)       *
 * 3.11.2002 PA                                            *
 ***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define  max(a,b)  ((a) > (b) ? (a) : (b))
#define  abs(a)  max((a),-(a))

/***********************************************************
 *                      Globals                            *
 ***********************************************************/

int M=13, N=27, ZERO=0, ONE=1;

/***********************************************************
 *                      Main                               *
 ***********************************************************/

main(int argc, char* argv[]) {
    int myid;                              /* process rank */
    int p;                          /* number of processes */
    int ctxt, pr, pc, myrow, mycol;          /* BLACS stuff*/
    int descA[9], descx[9], descy[9], info;
    double *A, *x, *y, alpha, beta;
    int m, br, i, ir, jr, kr, lr, n, bc, j, ic, jc, kc, lc;
    int nbr, nbc, nbr_loc, nbc_loc;

    /* Start the MPI engine */
    MPI_Init(&argc, &argv);

    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Find out process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Get a BLACS context */
    Cblacs_get(0, 0, &ctxt);

    /* Determine pr and pc for the pr x pc process grid */
    for (pc=p/2; p%pc; pc--)
	;
    pr = p/pc;
    if (pr > pc){
	pc = pr; pr = p/pc;
    }

    /* Initialize the pr x pc process grid */
    Cblacs_gridinit(&ctxt, "Row-major", pr, pc);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    /* Set row and column block sizes */
    br = bc = 5;

    /* No of block rows/columns, the last one may be partially full only */
    nbr = (M - 1)/br + 1;
    nbc = (N - 1)/bc + 1;

    /* No of local row column blocks */
    nbr_loc = (nbr - 1)/pr;
    if (pr*nbr_loc + myrow < nbr) ++nbr_loc;
    nbc_loc = (nbc - 1)/pc;
    if (pc*nbc_loc + mycol < nbc) ++nbc_loc;

    /* Determine sizes of local matrices */
    m = nbr_loc*br; n = nbc_loc*bc;

    /* Allocate memory space (maybe a little too much on some processors */
    x = (double*) malloc(n*sizeof(double));
    y = (double*) malloc(m*sizeof(double));
    A = (double*) malloc(m*n*sizeof(double));

    /* Initialize BLACS descriptors */
    descinit_(descA,&M,  &N,  &br, &bc, &ZERO,&ZERO,&ctxt,&m, &info);
    descinit_(descx,&ONE,&N,  &ONE,&bc, &ZERO,&ZERO,&ctxt,&ONE,&info);
    descinit_(descy,&M,  &ONE,&br, &ONE,&ZERO,&ZERO,&ctxt,&m, &info);

    /* Initialize matrix A with A[i,j] = |i-j| */
    j = 0;
    for (lc = 0; lc < nbc_loc; lc++){
	jc = lc*pc + mycol;
	for (kc = 0; kc < bc; kc++){
	    ic = jc*bc + kc;
	    if (ic < N){
		i = 0;
		for (lr = 0; lr < nbr_loc; lr++){
		    jr = lr*pr + myrow;
		    for (kr = 0; kr < br; kr++){
			ir = jr*br + kr;
			if (ir < M) A[i+j*m] = abs(ir - ic);
			i++;
		    }
		}
	    }
	    j++;
	}
    }

    /* Initialize vector x = [0,1,..,N_1] */
    j = 0;
    for (lc = 0; lc < nbc_loc; lc++){
	jc = lc*pc + mycol;
	for (kc = 0; kc < bc; kc++){
	    ic = jc*bc + kc;
	    if (ic < N) x[j] = ic;
	    j++;
	}
    }

    /* Multiply  y = alpha*A*x + beta*y  (alpha=1, beta=0)  */
    alpha = 1.0; beta = 0.0;
    pdgemv_("No Transpose",&M,&N,&alpha,A,&ONE,&ONE,descA,
	    x,&ONE,&ONE,descx,&ONE,&beta,y,&ONE,&ONE,descy,&ONE);

    /* Print result */
    if (mycol == 0){
	i = 0;
	for (lr = 0; lr < nbr_loc; lr++){
	    jr = lr*pr + myrow;
	    for (kr = 0; kr < br; kr++){
		ir = jr*br + kr;
		if (ir < M) printf("y[%d] = %12.2f\n", ir, y[i]);
		i++;
	    }
	}
    }

    /* Free memory */
    free(A);
    free(x);
    free(y);

    /* Release process grid */
    Cblacs_gridexit(ctxt);

    /* Shut down MPI */
    MPI_Finalize();

} /* main */
