/* ---------------------------------------------------------------------
*
*  -- ScaLAPACK routine (version 1.0) --
*     University of Tennessee, Knoxville, Oak Ridge National Laboratory,
*     and University of California, Berkeley.
*     November 17, 1996
*
*  ---------------------------------------------------------------------
*/
/*
*  Include files
*/
#include "pblas.h"

void pdgemm_( transa, transb, m, n, k, alpha, A, ia, ja, desc_A,
              B, ib, jb, desc_B, beta, C, ic, jc, desc_C )
/*
*  .. Scalar Arguments ..
*/
   F_CHAR      transa, transb;
   int         * ia, * ib, * ic, * ja, * jb, * jc, * k, * m, * n;
   double      * alpha, * beta;
/* ..
*  .. Array Arguments ..
*/
   int         desc_A[], desc_B[], desc_C[];
   double      A[], B[], C[];
{
/*
*  Purpose
*  =======
*
*  PDGEMM  performs one of the matrix-matrix operations
*
*     sub( C ) := alpha*op( sub( A ) )*op( sub( B ) ) + beta*sub( C ),
*
*  where sub( C ) denotes C(IC:IC+M-1,JC:JC+N-1),
*
*        op( X )  is one of
*        op( X ) = X   or   op( X ) = X',
*
*  thus  op( sub( A ) ) denotes A(IA:IA+M-1,JA:JA+K-1)  if TRANSA = 'N',
*                               A(IA:IA+K-1,JA:JA+M-1)' if TRANSA = 'T',
*                               A(IA:IA+K-1,JA:JA+M-1)' if TRANSA = 'C',
*
*        op( sub( B ) ) denotes B(IB:IB+K-1,JB:JB+N-1)  if TRANSB = 'N',
*                               B(IB:IB+N-1,JB:JB+K-1)' if TRANSB = 'T',
*                               B(IB:IB+N-1,JB:JB+K-1)' if TRANSB = 'C',
*
*  alpha and beta are scalars, and sub( A ), sub( B ) and sub( C ) are
*  distributed matrices, with op( sub( A ) ) an M-by-K distributed
*  matrix, op( sub( B ) ) a K-by-N distributed matrix and  sub( C ) an
*  M-by-N distributed matrix.
*
*  Notes
*  =====
*
*  Each global data object is described by an associated description
*  vector.  This vector stores the information required to establish
*  the mapping between an object element and its corresponding process
*  and memory location.
*
*  Let A be a generic term for any 2D block cyclicly distributed array.
*  Such a global array has an associated description vector descA.
*  In the following comments, the character _ should be read as
*  "of the global array".
*
*  NOTATION        STORED IN      EXPLANATION
*  --------------- -------------- --------------------------------------
*  DT_A   (global) descA[ DT_ ]   The descriptor type.  In this case,
*                                 DT_A = 1.
*  CTXT_A (global) descA[ CTXT_ ] The BLACS context handle, indicating
*                                 the BLACS process grid A is distribu-
*                                 ted over. The context itself is glo-
*                                 bal, but the handle (the integer
*                                 value) may vary.
*  M_A    (global) descA[ M_ ]    The number of rows in the global
*                                 array A.
*  N_A    (global) descA[ N_ ]    The number of columns in the global
*                                 array A.
*  MB_A   (global) descA[ MB_ ]   The blocking factor used to distribu-
*                                 te the rows of the array.
*  NB_A   (global) descA[ NB_ ]   The blocking factor used to distribu-
*                                 te the columns of the array.
*  RSRC_A (global) descA[ RSRC_ ] The process row over which the first
*                                 row of the array A is distributed.
*  CSRC_A (global) descA[ CSRC_ ] The process column over which the
*                                 first column of the array A is
*                                 distributed.
*  LLD_A  (local)  descA[ LLD_ ]  The leading dimension of the local
*                                 array.  LLD_A >= MAX(1,LOCr(M_A)).
*
*  Let K be the number of rows or columns of a distributed matrix,
*  and assume that its process grid has dimension p x q.
*  LOCr( K ) denotes the number of elements of K that a process
*  would receive if K were distributed over the p processes of its
*  process column.
*  Similarly, LOCc( K ) denotes the number of elements of K that a
*  process would receive if K were distributed over the q processes of
*  its process row.
*  The values of LOCr() and LOCc() may be determined via a call to the
*  ScaLAPACK tool function, NUMROC:
*          LOCr( M ) = NUMROC( M, MB_A, MYROW, RSRC_A, NPROW ),
*          LOCc( N ) = NUMROC( N, NB_A, MYCOL, CSRC_A, NPCOL ).
*  An upper bound for these quantities may be computed by:
*          LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A
*          LOCc( N ) <= ceil( ceil(N/NB_A)/NPCOL )*NB_A
*
*  Depending on the values of the arguments M, N and K, different
*  options are chosen, namely
*  if(      ( (*k) <= (*n) ) && ( (*k) <= (*m) ) ) matpos = 'C';
*  else if( ( (*n) <= (*m) ) && ( (*n) <= (*k) ) ) matpos = 'A';
*  else if( ( (*m) <= (*n) ) && ( (*m) <= (*k) ) ) matpos = 'B';
*  The distributed matrix chosen as a reference (A if N <= M and N <= K)
*  must be block aligned, i.e IX-1 must be a multiple of MB_X, and JX-1
*  a multiple of NB_X, where X one of A, B or C depending on the values
*  of M, N and K. Moreover, MB_op( A ) must be equal to MB_C, MB_op( B )
*  must be equal to NB_op( A ), and NB_op(B) must be equal to NB_C.
*  If for some reason, the matpos previously determined will be modified
*  until one successful possible choice will be found. In the following
*  iXrow and iXcol denote the process row and column owning respectively
*  the IX row and JX column of the matrix X.
*
*  If TRANSA == 'N' and TRANSB == 'N', then
*
*  If matpos = 'C', then the IA-1 (resp. JB-1) must be multiple of MB_A
*  (resp. NB_B), because of the alignment requirements on C.  If
*  K+MOD(JA-1,NB_A) > NB_A, then sub( A ) is not just contained into a
*  column of process, and the column offset of A must be equal to the
*  row offset of B, i.e MOD(JA-1,NB_A) = MOD(IB-1,MB_B); It is also
*  required that iarow = icrow and ibcol = iccol.
*  If matpos = 'A', then IB-1 (resp. IC-1) must be a multiple of MB_B
*  (resp. MB_C), because of the alignment requirements on A. If
*  N+MOD(JC-1,NB_C) > NB_C, then sub( B ) is not just contained into a
*  column of process, and the column offsets of B and C must be equal
*  to each other, i.e MOD(JB-1,NB_B) = MOD(JC-1,NB_C). It is also
*  required that iarow = icrow.
*  If matpos = 'B', then JA-1 (resp. JC-1) must be a multiple of NB_A
*  (resp. NB_C), because of the alignment requirements on B. If
*  M+MOD(IC-1,MB_C) > MB_C, then sub( C ) is not just contained into a
*  row of process, and the row offsets of A and C must be equal to each
*  other, i.e MOD(IA-1,MB_A) = MOD(IC-1,MB_C). It is also  required that
*  ibcol = iccol.
*
*  else if TRANSA <> 'N' and TRANSB == 'N', then
*
*  If matpos = 'C', then JA-1 (resp. JB-1) must be a multiple of NB_A
*  (resp. NB_B), because of the alignment requirements on C. If
*  K+MOD(IB-1,MB_B) > MB_B, then sub( B ) is not just contained into a
*  a row of process, and the row offset of A must be equal to the row
*  offset of B, i.e MOD(IA-1,MB_A) = MOD(IB-1,MB_B); It is also required
*  that ibcol = iccol.
*  If matpos = 'A', then IB-1 (resp. IC-1) must be a multiple of MB_B
*  (resp. MB_C), because of the alignment requirements on A. If
*  N+MOD(JC-1,NB_C) > NB_C, then sub( B ) is not just contained into a
*  column of process, and the column offsets of B and C must be equal
*  to each other, i.e MOD(JB-1,NB_B) = MOD(JC-1, NB_C). It is also
*  required that iarow = ibrow.
*  If matpos = 'B', then IA-1 (resp. JC-1) must be multiple of MB_A
*  (resp. NB_C), because of the alignment requirements on B. If
*  M+MOD(IC-1,MB_C) > MB_C, then sub( C ) is not just contained into a
*  a row of process, and the column offset of A and the row offset of
*  C must be equal to each other, i.e MOD(JA-1,NB_A) = MOD(IC-1,MB_C).
*  It is also required that iarow = ibrow and ibcol = iccol.
*
*  else if TRANSA == 'N' and TRANSB <> 'N', then
*
*  If matpos = 'C', then IA-1 (resp. IB-1) must be multiple of MB_A
*  (resp. MB_B), because of the alignment requirements on C. If
*  K+MOD(JA-1,NB_A) > NB_A, then sub( A ) is not just contained into a
*  column of process, and the column offsets of A and B must be equal
*  to each other, i.e MOD(JA-1,NB_A) = MOD(JB-1,NB_B); It is also
*  required that iarow = icrow.
*  If matpos = 'A', then IC-1 (resp. JB-1) must be a multiple of MB_C
*  (resp. NB_B), because of the alignment requirements on A. If
*  N+MOD(JC-1,NB_C) > NB_C, then sub( C ) is not just contained into a
*  column of process, and the column offset of C and the row offset of
*  B must be equal to each other, i.e MOD(JC-1,NB_C) = MOD(IB-1,MB_B).
*  It is also required that iarow = icrow and ibcol = iacol.
*  If matpos = 'B', then JA-1 (resp. JC-1) must be a multiple of NB_A
*  (resp. NB_C), because of the alignment requirements on B. If
*  M+MOD(IC-1,MB_C) > MB_C, then sub( C ) is not just contained into a
*  a row of process, and the rows offsets of A and C must be equal to
*  each other, i.e MOD(IA-1,MB_A) = MOD(IC-1,MB_C). It is also
*  required that iacol = ibcol.
*
*  else if TRANSA <> 'N' and TRANSB <> 'N', then
*
*  If matpos = 'C', then JA-1 (resp. IB-1) must be a multiple of NB_A
*  (resp. MB_B), because of the alignment requirements on C. If
*  K+MOD(IA-1,MB_A) > MB_A, then sub( A ) is not just contained into a
*  row of process, and the row offset of A must be equal to the column
*  offset of B, i.e MOD(IA-1,MB_A) = MOD(JB-1,NB_B);
*  If matpos = 'A', then IC-1 (resp. JB-1) must be a multiple of MB_C
*  (resp. NB_B), because of the alignment requirements on A. If
*  N+MOD(JC-1,NB_C) > NB_C, then sub( C ) is not just contained into a
*  column of process, and the column offset of C and the row offset of
*  B must be equal to each other, i.e MOD(JC-1,NB_C) = MOD(IB-1,MB_B).
*  If matpos = 'B', then IA-1 (resp. JC-1) must be a multiple of MB_A
*  (resp. NB_C), because of the alignment requirements on B. If
*  M+MOD(IC-1,MB_C) > MB_C, then sub( C ) is not just contained into a
*  a row of process, and the column offset of A and the row offset of
*  C must be equal to each other, i.e MOD(JA-1,NB_A) = MOD(IC-1,MB_C).
*
*  Parameters
*  ==========
*
*  TRANSA  (global input) pointer to CHARACTER
*          The form of op( A ) to be used in the matrix multiplication
*          as follows:
*
*                TRANSA = 'N' or 'n',    op( A ) = A,
*
*                TRANSA = 'T' or 't',    op( A ) = A',
*                TRANSA = 'C' or 'c',    op( A ) = A'.
*
*  TRANSB  (global input) pointer to CHARACTER
*          The form of op( B ) to be used in the matrix multiplication
*          as follows:
*
*                TRANSB = 'N' or 'n',    op( B ) = B,
*                TRANSB = 'T' or 't',    op( B ) = B',
*                TRANSB = 'C' or 'c',    op( B ) = B'.
*
*  M       (global input) pointer to INTEGER
*          The number of rows of the distributed matrices op( sub( A ) )
*          and sub( C ).  M >= 0.
*
*  N       (global input) pointer to INTEGER.
*          The number of columns of the distributed matrices
*           op( sub( B ) ) and sub( C ). N >= 0.
*
*  K       (global input) pointer to INTEGER.
*          The number of columns of the distributed matrix
*          op( sub( A ) ) and the number of rows of the distributed
*          matrix op( B ). K >= 0.
*
*  ALPHA   (global input) pointer to DOUBLE PRECISION
*          On entry, ALPHA specifies the scalar alpha.
*
*  A       (local input) DOUBLE PRECISION pointer into the local memory
*          to an array of dimension (LLD_A, KLa), where KLa is
*          LOCc(JA+K-1) when  TRANSA = 'N' or 'n',  and is LOCc(JA+M-1)
*          otherwise.  Before entry, this array must contain the local
*          pieces of the distributed matrix sub( A ).
*
*  IA      (global input) pointer to INTEGER
*          The global row index of the submatrix of the distributed
*          matrix A to operate on.
*
*  JA      (global input) pointer to INTEGER
*          The global column index of the submatrix of the distributed
*          matrix A to operate on.
*
*  DESCA   (global and local input) INTEGER array of dimension 8.
*          The array descriptor of the distributed matrix A.
*
*  B       (local input) DOUBLE PRECISION pointer into the local memory
*          to an array of dimension (LLD_B, KLb), where KLb is
*          LOCc(JB+N-1) when  TRANSB = 'N' or 'n', and is LOCc(JB+K-1)
*          otherwise. Before entry this array must contain the local
*          pieces of the distributed matrix sub( B ).
*
*  IB      (global input) pointer to INTEGER
*          The global row index of the submatrix of the distributed
*          matrix B to operate on.
*
*  JB      (global input) pointer to INTEGER
*          The global column index of the submatrix of the distributed
*          matrix B to operate on.
*
*  DESCB   (global and local input) INTEGER array of dimension 8.
*          The array descriptor of the distributed matrix B.
*
*  BETA    (global input) DOUBLE PRECISION
*          On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*          supplied as zero then sub( C ) need not be set on input.
*
*  C       (local input/local output) DOUBLE PRECISION pointer into the
*          local memory to an array of dimension (LLD_C, LOCc(JC+N-1)).
*          Before entry, this array must contain the local pieces of the
*          distributed matrix sub( C ). On exit, the distributed matrix
*          sub( C ) is overwritten by the M-by-N distributed matrix
*          alpha*op( sub( A ) )*op( sub( B ) ) + beta*sub( C ).
*
*  IC      (global input) pointer to INTEGER
*          The global row index of the submatrix of the distributed
*          matrix C to operate on.
*
*  JC      (global input) pointer to INTEGER
*          The global column index of the submatrix of the distributed
*          matrix C to operate on.
*
*  DESCC   (global and local input) INTEGER array of dimension 8.
*          The array descriptor of the distributed matrix C.
*
*  =====================================================================
*
*  .. Local Scalars ..
*/
   char        * ctop, * rtop, matpos, TrA, TrB;
   int         ablkcol, ablkrow, bblkrow, cblkcol, cblkrow, i, iacol,
               iarow, ibcol, iblk, ibrow, iccol, icoffa, icoffb, icoffc,
               icrow, ictxt, iia, iib, iic, in, info, iroffa, iroffb,
               iroffc, j, jblk, jja, jjb, jjc, jn, kb, kp0, kq0, lcm,
               lcmp, lcmq, mp0, mycol, myrow, nca, ncb, ncc, nota, notb,
               nprow, npcol, nq0, nra, nrb, nrc, tmp0, tmp1, tmp2, tmp3,
               tmp4, wksz;
   double      tbeta;
/* ..
*  .. PBLAS Buffer ..
*/
   double          * buff;
/* ..
*  .. External Functions ..
*/
   void        blacs_gridinfo_();
   void        pberror_();
   void        pbchkmat();
   char        * getpbbuf();
   char        * ptop();
   F_VOID_FCT  pbdgemm_();
   F_INTG_FCT  ilcm_();
/* ..
*  .. Executable Statements ..
*
*  Get grid parameters
*/
   ictxt = desc_A[CTXT_];
   blacs_gridinfo_( &ictxt, &nprow, &npcol, &myrow, &mycol );

/*
*  Test the input parameters
*/
   info = 0;
   if( nprow == -1 )
      info = -(1000+CTXT_+1);
   else
   {
      TrA = Mupcase( F2C_CHAR( transa )[0] );
      nota = ( TrA == 'N' );
      TrB = Mupcase( F2C_CHAR( transb )[0] );
      notb = ( TrB == 'N' );
      iroffa = (*ia-1) % desc_A[MB_];
      icoffa = (*ja-1) % desc_A[NB_];
      iroffb = (*ib-1) % desc_B[MB_];
      icoffb = (*jb-1) % desc_B[NB_];
      iroffc = (*ic-1) % desc_C[MB_];
      icoffc = (*jc-1) % desc_C[NB_];
      if( nota )
      {
         kb = desc_A[NB_];
         ablkrow = ( (*m) + iroffa <= desc_A[MB_] );
         ablkcol = ( (*k) + icoffa <= desc_A[NB_] );
         pbchkmat( *m, 3, *k, 5, *ia, *ja, desc_A, 10, &iia, &jja,
                   &iarow, &iacol, nprow, npcol, myrow, mycol,
                   &nra, &nca, &info );
      }
      else
      {
         kb = desc_A[MB_];
         ablkrow = ( (*k) + iroffa <= desc_A[MB_] );
         ablkcol = ( (*m) + icoffa <= desc_A[NB_] );
         pbchkmat( *k, 5, *m, 3, *ia, *ja, desc_A, 10, &iia, &jja,
                   &iarow, &iacol, nprow, npcol, myrow, mycol,
                   &nra, &nca, &info );
      }
      if( notb )
      {
         bblkrow = ( (*k) + iroffb <= desc_B[MB_] );
         pbchkmat( *k, 5, *n, 4, *ib, *jb, desc_B, 14, &iib, &jjb,
                   &ibrow, &ibcol, nprow, npcol, myrow, mycol,
                   &nrb, &ncb, &info );
      }
      else
      {
         bblkrow = ( (*n) + iroffb <= desc_B[MB_] );
         pbchkmat( *n, 4, *k, 5, *ib, *jb, desc_B, 14, &iib, &jjb,
                   &ibrow, &ibcol, nprow, npcol, myrow, mycol,
                   &nrb, &ncb, &info );
      }
      cblkrow = ( (*m) + iroffc <= desc_C[MB_] );
      cblkcol = ( (*n) + icoffc <= desc_C[NB_] );
      pbchkmat( *m, 3, *n, 4, *ic, *jc, desc_C, 19, &iic, &jjc,
                &icrow, &iccol, nprow, npcol, myrow, mycol,
                &nrc, &ncc, &info );
      if( info == 0 )
      {
         if( ( (*k) <= (*n) ) && ( (*k) <= (*m) ) )
         {
            matpos = 'C';
            if( nota && notb )
            {
               tmp0 = ( iroffa || icoffb );
               if( iroffc || icoffc || tmp0 || ( iarow != icrow ) ||
                   ( ibcol != iccol ) ||
                   ( !tmp0 && !ablkcol && icoffa != iroffb ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( icoffb )
                     info = -13;
                  else if( iroffc || ( iarow != icrow ) )
                     info = -17;
                  else if( icoffc || ( ibcol != iccol ) )
                     info = -18;
                  else
                     info = -12;
                  matpos = 'A';
                  tmp0 = ( iroffb || iroffc );
                  if( iroffa || icoffa || tmp0 || ( iarow != icrow ) ||
                      ( !tmp0 && !cblkcol && icoffb != icoffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else
                        info = -18;
                     matpos = 'B';
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else if( !icoffa && !icoffc && !cblkrow &&
                              iroffa != iroffc )
                        info = -17;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && notb )
            {
               tmp0 = ( icoffa || icoffb );
               if( iroffc || icoffc || tmp0 || ( ibcol != iccol ) ||
                   ( !tmp0 && !bblkrow && iroffa != iroffb ) )
               {
                  if( icoffa )
                     info = -9;
                  else if( icoffb )
                     info = -13;
                  else if( iroffc )
                     info = -17;
                  else if( icoffc || ( ibcol != iccol ) )
                     info = -18;
                  else
                     info = -12;
                  matpos = 'A';
                  tmp0 = ( iroffb || iroffc );
                  if( iroffa || icoffa || tmp0 || ( iarow != ibrow ) ||
                      ( !tmp0 && !cblkcol && icoffb != icoffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( iroffa )
                        info = -9;
                     else if( iroffb || ( iarow != ibrow ) )
                        info = -12;
                     else if( iroffc )
                        info = -17;
                     else
                        info = -18;
                     matpos = 'B';
                     if( iroffa )
                        info = -8;
                     else if( iroffb || ( iarow != ibrow ) )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else if( !iroffa && !icoffc && !cblkrow &&
                              icoffa != iroffc )
                        info = -17;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( nota && !notb )
            {
               tmp0 = ( iroffa || iroffb );
               if( iroffc || icoffc || tmp0 || ( iarow != icrow ) ||
                   ( !tmp0 && !ablkcol && icoffa != icoffb ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( iroffb )
                     info = -12;
                  else if( iroffc || ( iarow != icrow ) )
                     info = -17;
                  else if( icoffc )
                     info = -18;
                  else
                     info = -13;
                  matpos = 'A';
                  tmp0 = ( icoffb || iroffc );
                  if( iroffa || icoffa || tmp0 || ( iarow != icrow ) ||
                      ( ibcol != iacol ) ||
                      ( !tmp0 && !cblkcol && iroffb != icoffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( icoffb || ( ibcol != iacol ) )
                        info = -13;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else
                        info = -18;
                     matpos = 'B';
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb || ( ibcol != iacol ) )
                        info = -13;
                     else if( icoffc )
                        info = -18;
                     else if( !icoffa && !icoffc && !cblkrow &&
                              iroffa != iroffc )
                        info = -17;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && !notb )
            {
               tmp0 = ( icoffa || iroffb );
               if( iroffc || icoffc || tmp0 ||
                   ( !tmp0 && !ablkrow && iroffa != icoffb ) )
               {
                  if( icoffa )
                     info = -9;
                  else if( iroffb )
                     info = -12;
                  else if( iroffc )
                     info = -17;
                  else if( icoffc )
                     info = -18;
                  else
                     info = -13;
                  matpos = 'A';
                  tmp0 = ( icoffb || iroffc );
                  if( iroffa || icoffa || tmp0 ||
                      ( !tmp0 && !cblkcol && iroffb != icoffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( icoffb )
                        info = -13;
                     else if( iroffc )
                        info = -17;
                     else
                        info = -18;
                     matpos = 'B';
                     if( iroffa )
                        info = -8;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc )
                        info = -18;
                     else if( !iroffa && !icoffc && !cblkrow &&
                              icoffa != iroffc )
                        info = -17;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
         }
         else if( ( (*n) <= (*m) ) && ( (*n) <= (*k) ) )
         {
            matpos = 'A';
            if( nota && notb )
            {
               tmp0 = ( iroffb || iroffc );
               if( iroffa || icoffa || tmp0 || ( iarow != icrow ) ||
                   ( !tmp0 && !cblkcol && icoffb != icoffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( icoffa )
                     info = -9;
                  else if( iroffb )
                     info = -12;
                  else if( iroffc || ( iarow != icrow ) )
                     info = -17;
                  else
                     info = -18;
                  matpos = 'B';
                  tmp0 = ( icoffa || icoffc );
                  if( iroffb || icoffb || tmp0 || ( ibcol != iccol ) ||
                      ( !tmp0 && !cblkrow && iroffa != iroffc ) )
                  {
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = -17;
                     matpos = 'C';
                     if( iroffa )
                        info = -8;
                     else if( icoffb )
                        info = -13;
                     else if( !iroffa && !icoffb && !ablkcol &&
                              icoffa != iroffb )
                        info = -12;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && notb )
            {
               tmp0 = ( iroffb || iroffc );
               if( iroffa || icoffa || tmp0 || ( iarow != ibrow ) ||
                   ( !tmp0 && !cblkcol && icoffb != icoffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( icoffa )
                     info = -9;
                  else if( iroffb || ( iarow != ibrow ) )
                     info = -12;
                  else if( iroffc )
                     info = -17;
                  else
                     info = -18;
                  matpos = 'B';
                  tmp0 = ( iroffa || icoffc );
                  if( iroffb || icoffb || tmp0 || ( iarow != ibrow ) ||
                      ( ibcol != iccol ) ||
                      ( !tmp0 && !cblkrow && icoffa != iroffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( iroffb || ( iarow != ibrow ) )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = -17;
                     matpos = 'C';
                     if( icoffa )
                        info = -9;
                     else if( icoffb )
                        info = -13;
                     else if( !icoffa && !icoffb && !bblkrow &&
                              iroffa != iroffb )
                        info = -12;
                     else if( iroffc )
                        info = -17;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( nota && !notb )
            {
               tmp0 = ( icoffb || iroffc );
               if( iroffa || icoffa || tmp0 || ( ibcol != iacol ) ||
                   ( iarow != icrow ) ||
                   ( !tmp0 && !cblkcol && iroffb != icoffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( icoffa )
                     info = -9;
                  else if( icoffb || ( ibcol != iacol ) )
                     info = -13;
                  else if( iroffc || ( iarow != icrow ) )
                     info = -17;
                  else
                     info = -18;
                  matpos = 'B';
                  tmp0 = ( icoffa || icoffc );
                  if( iroffb || icoffb || tmp0 || ( iacol != ibcol ) ||
                      ( !tmp0 && !cblkrow && iroffa != iroffc ) )
                  {
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb || ( ibcol != iacol ) )
                        info = -13;
                     else if( icoffc )
                        info = -18;
                     else
                        info = -17;
                     matpos = 'C';
                     if( iroffa )
                        info = -8;
                     else if( iroffb )
                        info = -12;
                     else if( !iroffa && !iroffb && !ablkcol &&
                              icoffa != icoffb )
                       info = -13;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && !notb )
            {
               tmp0 = ( icoffb || iroffc );
               if( iroffa || icoffa || tmp0 ||
                   ( !tmp0 && !cblkcol && iroffb != icoffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( icoffa )
                     info = -9;
                  else if( icoffb )
                     info = -12;
                  else if( iroffc )
                     info = -17;
                  else
                     info = -18;
                  matpos = 'B';
                  tmp0 = ( iroffa || icoffc );
                  if( iroffb || icoffb || tmp0 ||
                      ( !tmp0 && !cblkrow && icoffa != iroffc ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( iroffb )
                        info = -12;
                     else if( icoffb )
                        info = -13;
                     else if( icoffc )
                        info = -18;
                     else
                        info = -17;
                     matpos = 'C';
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( !icoffa && !iroffb && !ablkrow &&
                              iroffa != icoffb )
                        info = -13;
                     else if( iroffc )
                        info = -17;
                     else if( icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
         }
         else if( ( (*m) <= (*n) ) && ( (*m) <= (*k) ) )
         {
            matpos = 'B';
            if( nota && notb )
            {
               tmp0 = ( icoffa || icoffc );
               if( iroffb || icoffb || tmp0 || ( ibcol != iccol ) ||
                   ( !tmp0 && !cblkrow && iroffa != iroffc ) )
               {
                  if( icoffa )
                     info = -9;
                  else if( iroffb )
                     info = -12;
                  else if( icoffb )
                     info = -13;
                  else if( icoffc || ( ibcol != iccol ) )
                     info = -18;
                  else
                     info = -17;
                  matpos = 'C';
                  tmp0 = ( iroffa || icoffb );
                  if( iroffc || icoffc || tmp0 || ( iarow != icrow ) ||
                      ( ibcol != iccol ) ||
                      ( !tmp0 && !ablkcol && icoffa != iroffb ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( icoffb )
                        info = -13;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = -12;
                     matpos = 'A';
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( !iroffb && !iroffc && !cblkcol &&
                              icoffb != icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && notb )
            {
               tmp0 = ( iroffa || icoffc );
               if( iroffb || icoffb || tmp0 || ( iarow != ibrow ) ||
                   ( ibcol != iccol ) ||
                   ( !tmp0 && !cblkrow && icoffa != iroffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( iroffb || ( iarow != ibrow ) )
                     info = -12;
                  else if( icoffb )
                     info = -13;
                  else if( icoffc || ( ibcol != iccol ) )
                     info = -18;
                  else
                     info = -17;
                  matpos = 'C';
                  tmp0 = ( icoffa || icoffb );
                  if( iroffc || icoffc || tmp0 || ( ibcol != iccol ) ||
                      ( !tmp0 && !bblkrow && iroffa != iroffb ) )
                  {
                     if( icoffa )
                        info = -9;
                     else if( icoffb )
                        info = -13;
                     else if( iroffc )
                        info = -17;
                     else if( icoffc || ( ibcol != iccol ) )
                        info = -18;
                     else
                        info = -12;
                     matpos = 'A';
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( iroffb || ( iarow != ibrow ) )
                        info = -12;
                     else if( iroffc )
                        info = -17;
                     else if( !iroffb && !iroffc && !cblkcol &&
                              icoffb != icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( nota && !notb )
            {
               tmp0 = ( icoffa || icoffc );
               if( iroffb || icoffb || tmp0 || ( iacol != ibcol ) ||
                   ( !tmp0 && !cblkrow && iroffa != iroffc ) )
               {
                  if( icoffa )
                     info = -9;
                  else if( iroffb )
                     info = -12;
                  else if( icoffb || ( iacol != ibcol ) )
                     info = -13;
                  else if( icoffc )
                     info = -18;
                  else if( !icoffa && !icoffc && !cblkrow &&
                           iroffa != iroffc )
                     info = -17;
                  matpos = 'C';
                  tmp0 = ( iroffa || iroffb );
                  if( iroffc || icoffc || tmp0 || ( iarow != icrow ) ||
                      ( !tmp0 && !ablkcol && icoffa != icoffb ) )
                  {
                     if( iroffa )
                        info = -8;
                     else if( iroffb )
                        info = -12;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( icoffc )
                        info = -18;
                     else
                        info = -13;
                     matpos = 'A';
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( icoffb || ( iacol != ibcol ) )
                        info = -13;
                     else if( iroffc || ( iarow != icrow ) )
                        info = -17;
                     else if( !icoffb && !iroffc && !cblkcol &&
                              iroffb != icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
            else if( !nota && !notb )
            {
               tmp0 = ( iroffa || icoffc );
               if( iroffb || icoffb || tmp0 ||
                   ( !tmp0 && !cblkrow && icoffa != iroffc ) )
               {
                  if( iroffa )
                     info = -8;
                  else if( iroffb )
                     info = -12;
                  else if( icoffb )
                     info = -13;
                  else if( icoffc )
                     info = -18;
                  else
                     info = -17;
                  matpos = 'C';
                  tmp0 = ( icoffa || iroffb );
                  if( iroffc || icoffc || tmp0 ||
                      ( !tmp0 && !ablkrow && iroffa != icoffb ) )
                  {
                     if( icoffa )
                        info = -9;
                     else if( iroffb )
                        info = -12;
                     else if( iroffc )
                        info = -17;
                     else if( icoffc )
                        info = -18;
                     else
                        info = -13;
                     matpos = 'A';
                     if( iroffa )
                        info = -8;
                     else if( icoffa )
                        info = -9;
                     else if( icoffb )
                        info = -12;
                     else if( iroffc )
                        info = -17;
                     else if( !icoffb && !iroffc && !cblkcol &&
                              iroffb != icoffc )
                        info = -18;
                     else
                        info = 0;
                  }
                  else
                  {
                     info = 0;
                  }
               }
            }
         }

         if( info == 0 )
         {
            if( (TrA != 'N') && (TrA != 'T') && (TrA != 'C') )
               info = -1;
            else if( (TrB != 'N') && (TrB != 'T') && (TrB != 'C') )
               info = -2;
            if( nota )
            {
               if( desc_A[MB_] != desc_C[MB_] )
                  info = -(1900+MB_+1);
               if( notb )
               {
                  if( desc_A[NB_] != desc_B[MB_] )
                     info = -(1400+MB_+1);
                  else if( desc_B[NB_] != desc_C[NB_] )
                     info = -(1900+NB_+1);
               }
               else
               {
                  if( desc_A[NB_] != desc_B[NB_] )
                     info = -(1400+NB_+1);
                  else if( desc_B[MB_] != desc_C[NB_] )
                     info = -(1900+NB_+1);
               }
            }
            else
            {
               if( desc_A[NB_] != desc_C[MB_] )
                     info = -(1900+MB_+1);
               if( notb )
               {
                  if( desc_A[MB_] != desc_B[MB_] )
                     info = -(1400+MB_+1);
                  else if( desc_B[NB_] != desc_C[NB_] )
                     info = -(1400+NB_+1);
               }
               else
               {
                  if( desc_A[MB_] != desc_B[NB_] )
                     info = -(1400+NB_+1);
                  else if( desc_B[MB_] != desc_C[NB_] )
                     info = -(1900+NB_+1);
               }
            }
            if( ictxt != desc_B[CTXT_] )
               info = -(1400+CTXT_+1);
            else if( ictxt != desc_C[CTXT_] )
               info = -(1900+CTXT_+1);
         }
      }
   }
   if( info )
   {
     pberror_( &ictxt, "PDGEMM", &info );
     return;
   }
/*
*  Quick return if possible.
*/
   if( ( *m == 0 ) || ( *n == 0 ) ||
       ( ( *alpha == ZERO || *k == 0 ) && ( *beta == ONE ) ) )
      return;
/*
*  Figure out the arguments to be passed to pbdgemm and compute
*  adequate workspace size
*/
   lcm = ilcm_( &nprow, &npcol );
   if( nota )
   {
      if( notb )
      {
         if( matpos == 'C' )
         {
            tmp1 = (*m) / desc_C[MB_];
            tmp2 = (*n) / desc_C[NB_];
            wksz = ( MYROC0( tmp1, *m, desc_C[MB_], nprow ) +
                     MYROC0( tmp2, *n, desc_C[NB_], npcol ) ) * desc_A[NB_];
         }
         else if( matpos == 'A' )
         {
            lcmq = lcm / npcol;
            tmp1 = (*m) / desc_C[MB_];
            mp0 = MYROC0( tmp1, *m, desc_C[MB_], nprow );
            tmp2 = (*k) / kb;
            kq0 = MYROC0( tmp2, *k, kb, npcol );
            tmp2 = kq0 / kb;
            tmp1 = MYROC0( tmp2, kq0, kb, lcmq );
            wksz = desc_C[NB_] * ( kq0 + MAX( tmp1, mp0 ) );
         }
         else
         {
            lcmp = lcm / nprow;
            tmp1 = (*k) / kb;
            kp0 = MYROC0( tmp1, *k, kb, nprow );
            tmp2 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp2, *n, desc_C[NB_], npcol );
            tmp2 = kp0 / kb;
            tmp1 = MYROC0( tmp2, kp0, kb, lcmp );
            wksz = desc_C[MB_] * ( kp0 + MAX( tmp1, nq0 ) );
         }
      }
      else
      {
         if( matpos == 'C' )
         {
            tmp0 = (*m) / desc_C[MB_];
            lcmq = lcm / npcol;
            tmp1 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp1, *n, desc_C[NB_], npcol );
            tmp1 = nq0 / desc_C[NB_];
            wksz = ( MYROC0( tmp0, *m, desc_C[MB_], nprow ) +
                     nq0 + MYROC0( tmp1, nq0, desc_C[NB_], lcmq ) ) *
                     desc_A[NB_];
         }
         else if( matpos == 'A' )
         {
            tmp0 = (*m) / desc_C[MB_];
            tmp1 = (*k) / kb;
            wksz = ( MYROC0( tmp1, *k, kb, npcol ) +
                     MYROC0( tmp0, *m, desc_C[MB_], nprow ) ) * desc_C[NB_];
         }
         else
         {
            lcmq = lcm / npcol;
            tmp1 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp1, *n, desc_C[NB_], npcol );
            tmp2 = (*k) / kb;
            kq0 = MYROC0( tmp2, *k, kb, npcol );
            tmp3 = nq0 / desc_C[NB_];
            tmp2 = MYROC0( tmp3, nq0, desc_C[NB_], lcmq );
            wksz = ( MYROC0( tmp1, *n, desc_C[NB_], nprow ) +
                     MAX( tmp2, kq0 ) ) * desc_C[MB_];
         }
      }
   }
   else
   {
      if( notb )
      {
         if( matpos == 'C' )
         {
            lcmp = lcm / nprow;
            tmp1 = (*m) / desc_C[MB_];
            mp0 = MYROC0( tmp1, *m, desc_C[MB_], nprow );
            tmp2 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp2, *n, desc_C[NB_], npcol );
            tmp2 = mp0 / desc_C[MB_];
            tmp1 = MYROC0( tmp2, mp0, desc_C[MB_], lcmp );
            wksz = desc_B[MB_] * ( mp0 + MAX( tmp1, nq0 ) );
         }
         else if( matpos == 'A' )
         {
            lcmp = lcm / nprow;
            tmp1 = (*m) / desc_C[MB_];
            mp0 = MYROC0( tmp1, *m, desc_C[MB_], nprow );
            tmp2 = (*k) / kb;
            kp0 = MYROC0( tmp2, *k, kb, nprow );
            tmp3 = mp0 / desc_C[MB_];
            tmp2 = MYROC0( tmp3, mp0, desc_C[MB_], lcmp );
            wksz = ( MYROC0( tmp1, *m, desc_C[MB_], npcol ) +
                     MAX( tmp2, kp0 ) ) * desc_C[NB_];
         }
         else
         {
            tmp1 = (*k) / kb;
            tmp2 = (*n) / desc_C[NB_];
            wksz = ( MYROC0( tmp1, *k, kb, nprow ) +
                     MYROC0( tmp2, *n, desc_C[NB_], npcol ) ) * desc_C[MB_];
         }
      }
      else
      {
         if( matpos == 'C' )
         {
            lcmp = lcm / nprow;
            lcmq = lcm / npcol;
            tmp1 = (*m) / desc_C[MB_];
            mp0 = MYROC0( tmp1, *m, desc_C[MB_], nprow );
            tmp2 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp2, *n, desc_C[NB_], npcol );
            tmp3 = mp0 / desc_C[MB_];
            tmp1 = MYROC0( tmp3, mp0, desc_C[MB_], lcmp );
            tmp3 = nq0 / desc_C[NB_];
            tmp2 = MYROC0( tmp3, nq0, desc_C[NB_], lcmq );
            wksz = desc_A[MB_] * ( mp0 + nq0 + MAX( tmp1, tmp2 ) );
         }
         else if( matpos == 'A' )
         {
            lcmp = lcm / nprow;
            tmp1 = (*m) / desc_C[MB_];
            mp0 = MYROC0( tmp1, *m, desc_C[MB_], nprow );
            tmp2 = (*k) / kb;
            kp0 = MYROC0( tmp2, *k, kb, nprow );
            tmp3 = mp0 / desc_C[MB_];
            tmp2 = MYROC0( tmp3, mp0, desc_C[MB_], lcmp );
            tmp4 = kp0 / kb;
            tmp3 = kp0 + MYROC0( tmp4, kp0, kb, lcmp );
            wksz = ( MYROC0( tmp1, *m, desc_C[MB_], npcol ) +
                     MAX( tmp2, tmp3 ) ) * desc_C[NB_];
         }
         else
         {
            lcmq = lcm / npcol;
            tmp1 = (*n) / desc_C[NB_];
            nq0 = MYROC0( tmp1, *n, desc_C[NB_], npcol );
            tmp2 = (*k) / kb;
            kq0 = MYROC0( tmp2, *k, kb, npcol );
            tmp3 = nq0 / desc_C[NB_];
            tmp2 = MYROC0( tmp3, nq0, desc_C[NB_], lcmq );
            tmp4 = kq0 / kb;
            tmp3 = kq0 + MYROC0( tmp4, kq0, kb, lcmq );
            wksz = ( MYROC0( tmp1, *n, desc_C[NB_], nprow ) +
                     MAX( tmp2, tmp3 ) ) * desc_C[MB_];
         }
      }
   }
   buff = (double *)getpbbuf( "PDGEMM", wksz*sizeof(double) );
/*
*  Call PB-BLAS routine
*/
   if( matpos == 'A' && cblkcol )
   {
      if( nota && !notb )
      {
         ctop = ptop( BROADCAST, COLUMN, TOPGET );
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                   C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
      else
      {
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                   C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
   }
   else if( matpos == 'B' && cblkrow )
   {
      if( !nota && notb )
      {
         rtop = ptop( BROADCAST, ROW, TOPGET );
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                   C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
      else
      {
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                   C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
   }
   else if( ( ( ablkcol && nota  ) || ( ablkrow && !nota ) ) &&
            ( matpos == 'C' ) )
   {
      if( nota && notb )
      {
         rtop = ptop( BROADCAST, ROW, TOPGET );
         ctop = ptop( BROADCAST, COLUMN, TOPGET );
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                   C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
      else
      {
         pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m, n, k,
                   &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                   &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                   &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                   &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                   &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                   C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                   C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                   buff );
      }
   }
   else
   {
      if( matpos == 'A' )               /* loop over the columns of C */
      {
         j = CEIL( (*jc), desc_C[NB_] ) * desc_C[NB_];
         jn = (*jc)+(*n)-1;
         jn = MIN( j, jn );
         jblk = jn-(*jc)+1;

         if( notb )                     /* loop over the columns of B */
         {                           /* Handle first block separately */
            pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                      &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                      &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                      &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                      beta, &C[iic-1+(jjc-1)*desc_C[LLD_]],
                      &desc_C[LLD_], &iarow, &iacol, &ibrow, &ibcol,
                      &icrow, &iccol, C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                      C2F_CHAR( NO ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                      C2F_CHAR( NO ), buff );
            if( mycol == ibcol )
            {
               jjb += jblk;
               jjb = MIN( jjb, ncb );
            }
            if( mycol == iccol )
            {
               jjc += jblk;
               jjc = MIN( jjc, ncc );
            }
            ibcol = (ibcol+1) % npcol;
            iccol = (iccol+1) % npcol;
                              /* loop over remaining block of columns */
            tmp0 = (*jc)+(*n)-1;
            for( j=jn+1; j <= tmp0; j+=desc_C[NB_] )
            {
               jblk = (*n)-j+(*jc);
               jblk = MIN( jblk, desc_C[NB_] );
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                         &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                         beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( mycol == ibcol )
               {
                  jjb += jblk;
                  jjb = MIN( jjb, ncb );
               }
               if( mycol == iccol )
               {
                  jjc += jblk;
                  jjc = MIN( jjc, ncc );
               }
               ibcol = (ibcol+1) % npcol;
               iccol = (iccol+1) % npcol;
            }
         }
         else                              /* loop over the rows of B */
         {                           /* Handle first block separately */
            if( nota )
            {
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                         &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( myrow == ibrow )
               {
                  iib += jblk;
                  iib = MIN( iib, nrb );
               }
               if( mycol == iccol )
               {
                  jjc += jblk;
                  jjc = MIN( jjc, ncc );
               }
               ibrow = (ibrow+1) % nprow;
               iccol = (iccol+1) % npcol;
                                        /* loop over remaining blocks */
               tmp0 = (*jc)+(*n)-1;
               for( j=jn+1; j <= tmp0; j+=desc_C[NB_] )
               {
                  jblk = (*n)-j+(*jc);
                  jblk = MIN( jblk, desc_C[NB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            beta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
                  if( myrow == ibrow )
                  {
                     iib += jblk;
                     iib = MIN( iib, nrb );
                  }
                  if( mycol == iccol )
                  {
                     jjc += jblk;
                     jjc = MIN( jjc, ncc );
                  }
                  ibrow = (ibrow+1) % nprow;
                  iccol = (iccol+1) % npcol;
               }
            }
            else
            {
               ctop = ptop( BROADCAST, COLUMN, TOPGET );
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                         m, &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb,
                         alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                         beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( myrow == ibrow )
               {
                  iib += jblk;
                  iib = MIN( iib, nrb );
               }
               if( mycol == iccol )
               {
                  jjc += jblk;
                  jjc = MIN( jjc, ncc );
               }
               ibrow = (ibrow+1) % nprow;
               iccol = (iccol+1) % npcol;
                                        /* loop over remaining blocks */
               tmp0 = (*jc)+(*n)-1;
               for( j=jn+1; j <= tmp0; j+=desc_C[NB_] )
               {
                  jblk = (*n)-j+(*jc);
                  jblk = MIN( jblk, desc_C[NB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, &jblk, k, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            beta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( ctop ), C2F_CHAR( NO ),
                            C2F_CHAR( NO ), buff );
                  if( myrow == ibrow )
                  {
                     iib += jblk;
                     iib = MIN( iib, nrb );
                  }
                  if( mycol == iccol )
                  {
                     jjc += jblk;
                     jjc = MIN( jjc, ncc );
                  }
                  ibrow = (ibrow+1) % nprow;
                  iccol = (iccol+1) % npcol;
               }
            }
         }
      }
      else if( matpos == 'B' )             /* loop over the rows of C */
      {
         i = CEIL( (*ic), desc_C[MB_] ) * desc_C[MB_];
         in = (*ic)+(*m)-1;
         in = MIN( i, in );
         iblk = in-(*ic)+1;

         if( nota )                        /* loop over the rows of A */
         {                           /* Handle first block separately */
            pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                      &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                      &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                      &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                      &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                      &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                      C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                      C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                      buff );
            if( myrow == iarow )
            {
               iia += iblk;
               iia = MIN( iia, nra );
            }
            if( myrow == icrow )
            {
               iic += iblk;
               iic = MIN( iic, nrc );
            }
            iarow = (iarow+1) % nprow;
            icrow = (icrow+1) % nprow;
                                        /* loop over remaining blocks */
            tmp0 = (*ic)+(*m)-1;
            for( i=in+1; i <= tmp0; i+=desc_C[MB_] )
            {
               iblk = (*m)-i+(*ic);
               iblk = MIN( iblk, desc_C[MB_] );
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                         &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb,
                         alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                         beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( myrow == iarow )
               {
                  iia += iblk;
                  iia = MIN( iia, nra );
               }
               if( myrow == icrow )
               {
                  iic += iblk;
                 iic = MIN( iic, nrc );
               }
               iarow = (iarow+1) % nprow;
               icrow = (icrow+1) % nprow;
            }
         }
         else                           /* loop over the columns of A */
         {                           /* Handle first block separately */
            if( notb )
            {
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                         &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( mycol == iacol )
               {
                  jja += iblk;
                  jja = MIN( jja, nca );
               }
               if( myrow == icrow )
               {
                  iic += iblk;
                  iic = MIN( iic, nrc );
               }
               iacol = (iacol+1) % npcol;
               icrow = (icrow+1) % nprow;
                                        /* loop over remaining blocks */
               tmp0 = (*ic)+(*m)-1;
               for( i=in+1; i <= tmp0; i+=desc_C[MB_] )
               {
                  iblk = (*m)-i+(*ic);
                  iblk = MIN( iblk, desc_C[MB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            beta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
                  if( mycol == iacol )
                  {
                     jja += iblk;
                     jja = MIN( jja, nca );
                  }
                  if( myrow == icrow )
                  {
                     iic += iblk;
                     iic = MIN( iic, nrc );
                  }
                  iacol = (iacol+1) % npcol;
                  icrow = (icrow+1) % nprow;
               }
            }
            else
            {
               rtop = ptop( BROADCAST, ROW, TOPGET );
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                         &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb,
                         alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                         beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( mycol == iacol )
               {
                  jja += iblk;
                  jja = MIN( jja, nca );
               }
               if( myrow == icrow )
               {
                  iic += iblk;
                  iic = MIN( iic, nrc );
               }
               iacol = (iacol+1) % npcol;
               icrow = (icrow+1) % nprow;
                                        /* loop over remaining blocks */
               tmp0 = (*ic)+(*m)-1;
               for( i=in+1; i <= tmp0; i+=desc_C[MB_] )
               {
                  iblk = (*m)-i+(*ic);
                  iblk = MIN( iblk, desc_C[MB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            &iblk, n, k, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            beta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                            C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                            C2F_CHAR( NO ), buff );
                  if( mycol == iacol )
                  {
                     jja += iblk;
                     jja = MIN( jja, nca );
                  }
                  if( myrow == icrow )
                  {
                     iic += iblk;
                     iic = MIN( iic, nrc );
                  }
                  iacol = (iacol+1) % npcol;
                  icrow = (icrow+1) % nprow;
               }
            }
         }
      }
      else if( matpos == 'C' )
      {
         tbeta = ONE;
         if( nota )                     /* loop over the columns of A */
         {
            j = CEIL( (*ja), desc_A[NB_] ) * desc_A[NB_];
            jn = (*ja)+(*k)-1;
            jn = MIN( j, jn );
            jblk = jn-(*ja)+1;

            if( notb )                     /* loop over the rows of B */
            {                        /* Handle first block separately */
               rtop = ptop( BROADCAST, ROW, TOPGET );
               ctop = ptop( BROADCAST, COLUMN, TOPGET );
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                         m, n, &jblk, &desc_C[MB_], &desc_C[NB_], &kb,
                         alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                         beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                         C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( mycol == iacol )
               {
                  jja += jblk;
                  jja = MIN( jja, nca );
               }
               if( myrow == ibrow )
               {
                  iib += jblk;
                  iib = MIN( iib, nrb );
               }
               iacol = (iacol+1) % npcol;
               ibrow = (ibrow+1) % nprow;
                                        /* loop over remaining blocks */
               tmp0 = (*ja)+(*k)-1;
               for( j=jn+1; j <= tmp0; j+=desc_A[NB_] )
               {
                  jblk = (*k)-j+(*ja);
                  jblk = MIN( jblk, desc_A[NB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, n, &jblk, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            &tbeta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( rtop ), C2F_CHAR( NO ),
                            C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                            buff );
                  if( mycol == iacol )
                  {
                     jja += jblk;
                     jja = MIN( jja, nca );
                  }
                  if( myrow == ibrow )
                  {
                     iib += jblk;
                     iib = MIN( iib, nrb );
                  }
                  iacol = (iacol+1) % npcol;
                  ibrow = (ibrow+1) % nprow;
               }
            }
            else                        /* loop over the columns of B */
            {                        /* Handle first block separately */
              pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                         n, &jblk, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( mycol == iacol )
               {
                  jja += jblk;
                  jja = MIN( jja, nca );
               }
               if( mycol == ibcol )
               {
                  jjb += jblk;
                  jjb = MIN( jjb, ncb );
               }
               iacol = (iacol+1) % npcol;
               ibcol = (ibcol+1) % npcol;
                                        /* loop over remaining blocks */
               tmp0 = (*ja)+(*k)-1;
               for( j=jn+1; j <= tmp0; j+=desc_A[NB_] )
               {
                  jblk = (*k)-j+(*ja);
                  jblk = MIN( jblk, desc_A[NB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, n, &jblk, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            &tbeta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
                  if( mycol == iacol )
                  {
                     jja += jblk;
                     jja = MIN( jja, nca );
                  }
                  if( mycol == ibcol )
                  {
                     jjb += jblk;
                     jjb = MIN( jjb, ncb );
                  }
                  iacol = (iacol+1) % npcol;
                  ibcol = (ibcol+1) % npcol;
               }
            }
         }
         else                              /* loop over the rows of A */
         {
            i = CEIL( (*ia), desc_A[MB_] ) * desc_A[MB_];
            in = (*ia)+(*k)-1;
            in = MIN( i, in );
            iblk = in-(*ia)+1;

            if( notb )                     /* loop over the rows of B */
            {                        /* Handle first block separately */
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                         n, &iblk, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( myrow == iarow )
               {
                  iia += iblk;
                  iia = MIN( iia, nra );
               }
               if( myrow == ibrow )
               {
                  iib += iblk;
                  iib = MIN( iib, nrb );
               }
               iarow = (iarow+1) % nprow;
               ibrow = (ibrow+1) % nprow;
                                        /* loop over remaining blocks */
               tmp0 = (*ia)+(*k)-1;
               for( i=in+1; i <= tmp0; i+=desc_A[MB_] )
               {
                  iblk = (*k)-i+(*ia);
                  iblk = MIN( iblk, desc_A[MB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, n, &iblk, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            &tbeta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
                  if( myrow == iarow )
                  {
                     iia += iblk;
                     iia = MIN( iia, nra );
                  }
                  if( myrow == ibrow )
                  {
                     iib += iblk;
                     iib = MIN( iib, nrb );
                  }
                  iarow = (iarow+1) % nprow;
                  ibrow = (ibrow+1) % nprow;
               }
            }
            else                        /* loop over the columns of B */
            {                        /* Handle first block separately */
               pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb, m,
                         n, &iblk, &desc_C[MB_], &desc_C[NB_], &kb, alpha,
                         &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                         &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_], beta,
                         &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                         &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                         C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ), C2F_CHAR( NO ),
                         C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ),
                         buff );
               if( myrow == iarow )
               {
                  iia += iblk;
                  iia = MIN( iia, nra );
               }
               if( mycol == ibcol )
               {
                  jjb += iblk;
                  jjb = MIN( jjb, ncb );
               }
               iarow = (iarow+1) % nprow;
               ibcol = (ibcol+1) % npcol;
                                        /* loop over remaining blocks */
               tmp0 = (*ia)+(*k)-1;
               for( i=in+1; i <= tmp0; i+=desc_A[MB_] )
               {
                  iblk = (*k)-i+(*ia);
                  iblk = MIN( iblk, desc_A[MB_] );
                  pbdgemm_( &ictxt, C2F_CHAR( &matpos ), transa, transb,
                            m, n, &iblk, &desc_C[MB_], &desc_C[NB_], &kb,
                            alpha,
                            &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                            &B[iib-1+(jjb-1)*desc_B[LLD_]], &desc_B[LLD_],
                            &tbeta,
                            &C[iic-1+(jjc-1)*desc_C[LLD_]], &desc_C[LLD_],
                            &iarow, &iacol, &ibrow, &ibcol, &icrow, &iccol,
                            C2F_CHAR( "A" ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( TOPDEF ),
                            C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
                  if( myrow == iarow )
                  {
                     iia += iblk;
                     iia = MIN( iia, nra );
                  }
                  if( mycol == ibcol )
                  {
                     jjb += iblk;
                     jjb = MIN( jjb, ncb );
                  }
                  iarow = (iarow+1) % nprow;
                  ibcol = (ibcol+1) % npcol;
               }
            }
         }
      }
   }
}
