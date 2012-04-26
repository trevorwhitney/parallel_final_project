#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define  max(a,b)  ((a) > (b) ? (a) : (b))
#define  abs(a)  max((a),-(a))

int M=120, N=100, L = 150, ZERO=0, ONE=1;
int MBSIZE = 4, NBSIZE = 4, LBSIZE = 6;

main(int argc, char* argv[]) {
  int myid;
  int p;
  /* BLACS stuff */
  int ctxt, pr, pc, myrow, mycol;
  int i_loc, j_loc, i_glob, j_glob, nblks;
  int mbsize, nbsize, lbsize, mb, nb, lb;
  int nrow_A, ncol_A, nrow_B, ncol_B, nrow_C, ncol_C;
  int descA[9], descB[9], descC[9], info;
  double *A, *B, *C, alpha, beta, err;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  /* Get a BLACS context */
  Cblacs_get(0, 0, &ctxt);

  /* Initialize an 'optimal' 2-dimensional grid s.t. pr <= pc */
  for (pc=p/2; p%pc; pc--);
  pr = p/pc;
  if (pr > pc){
    pc = pr; pr = p/pc;
  }

  /* Initialize the pr x pc process grid */
  Cblacs_gridinit(&ctxt, "Row-major", pr, pc);
  Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

  /* Define block sizes */
  mbsize = MBSIZE; nbsize = NBSIZE; lbsize = LBSIZE;

  /* Determine sizes of local matrices */
  nblks = (M-1)/mbsize + 1;
  mb = ((nblks-1)/pr + 1)*mbsize;
  if (myrow < nblks%pr) {
    nrow_C = (nblks/pr + 1)*mbsize;
  if (M%mbsize && myrow == nblks%pr-1)
    nrow_C = nrow_C - mbsize + M%mbsize;
  }
  else
    nrow_C = (nblks/pr)*mbsize;
  
  nrow_A = nrow_C;

  nblks = (N-1)/nbsize + 1;
  nb = ((nblks-1)/pc + 1)*nbsize;
  if (mycol < nblks%pc) {
    ncol_C = (nblks/pc + 1)*nbsize;
  if (N%nbsize && mycol == nblks%pc-1)
    ncol_C = ncol_C - nbsize + N%nbsize;
  }
  else
    ncol_C = (nblks/pc)*nbsize;
  
  ncol_B = ncol_C;

  nblks = (L-1)/lbsize + 1;
  gif (mycol < nblks%pc) {
  ncol_A = (nblks/pc + 1)*lbsize;
  if (L%lbsize && mycol == nblks%pc-1)
    ncol_A = ncol_A - lbsize + L%lbsize;
  }
  else
    ncol_A = (nblks/pc)*lbsize;

  nblks = (L-1)/lbsize + 1;
  lb = ((nblks-1)/pr + 1)*lbsize;
  if (myrow < nblks%pr) {
    nrow_B = (nblks/pr + 1)*lbsize;
  if (L%lbsize)
    nrow_B = nrow_B - lbsize + L%lbsize;
  }
  else
    nrow_B = (nblks/pr)*lbsize;

  /* Allocate memory space */
  A = (double*) malloc(mb*lb*sizeof(double));
  B = (double*) malloc(lb*nb*sizeof(double));
  C = (double*) malloc(mb*nb*sizeof(double));

  /* Define array descriptors */
  descinit_(descA,&M,&L,&mbsize,&lbsize,&ZERO,&ZERO,&ctxt,&mb,&info);
  descinit_(descB,&L,&N,&lbsize,&nbsize,&ZERO,&ZERO,&ctxt,&lb,&info);
  descinit_(descC,&M,&N,&mbsize,&nbsize,&ZERO,&ZERO,&ctxt,&mb,&info);

   /* Initialize matrix  A   */
  for (j_loc = 0; j_loc < ncol_A; j_loc++){
    for (i_loc = 0; i_loc < nrow_A; i_loc++){
      A[i_loc+j_loc*mb] = 0.0;
      i_glob = ((i_loc/mbsize)*pr + myrow)*mbsize
        + i_loc%mbsize;
      j_glob = ((j_loc/lbsize)*pc + mycol)*lbsize
        + j_loc%lbsize;
      if (j_glob <= i_glob) A[i_loc+j_loc*mb] = 1.0;
    }
  }

  /* Initialize matrix  B   */
  for (j_loc = 0; j_loc < ncol_B; j_loc++){
    for (i_loc = 0; i_loc < nrow_B; i_loc++){
      i_glob = ((i_loc/lbsize)*pr + myrow)*lbsize
        + i_loc%lbsize;
      j_glob = ((j_loc/nbsize)*pc + mycol)*nbsize
        + j_loc%nbsize;
      B[i_loc+j_loc*lb] = j_glob;
    }
  }

  /* Multiply  C = alpha*A*B + beta*C  (alpha=1, beta=0)  */
  alpha = 1.0; beta = 0.0;
  pdgemm_("No Transpose", "No Transpose", &M, &N, &L, &alpha,
    A, &ONE, &ONE, descA, B, &ONE, &ONE, descB, &beta,
    C, &ONE, &ONE, descC);

  /* Check for correctness */
  err = 0.0;
  for (j_loc = 0; j_loc < ncol_C; j_loc++){
    j_glob = ((j_loc/nbsize)*pc + mycol)*nbsize
      + j_loc%nbsize;
    for (i_loc = 0; i_loc < nrow_C; i_loc++){
      i_glob = ((i_loc/mbsize)*pr + myrow)*mbsize
        + i_loc%mbsize;
      err += abs(C[i_loc+j_loc*mb] - (i_glob+1)*j_glob);
    }
  }

  /* Check result */
  printf("Local error on proc %d = %10.2f\n",myid,err);

  /* Free memory */
  free(A);
  free(B);
  free(C);

  /* Release process grid */
  Cblacs_gridexit(ctxt);

  /* Shut down MPI */
  MPI_Finalize();

} /* main */

