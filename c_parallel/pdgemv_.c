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

void pdgemv_( trans, m, n, alpha, A, ia, ja, desc_A, X, ix, jx, desc_X,
              incx, beta, Y, iy, jy, desc_Y, incy )
/*
*  .. Scalar Arguments ..
*/
   F_CHAR      trans;
   int         * ia, * incx, * incy, * ix, * iy, * ja, * jx, * jy, * m,
               * n;
   double      * alpha, * beta;
/* ..
*  .. Array Arguments ..
*/
   int        desc_A[], desc_X[], desc_Y[];
   double      A[], X[], Y[];
{
/*
*  Purpose
*  =======
*
*  PDGEMV performs one of the distributed matrix-vector operations
*
*     sub( Y ) := alpha*sub( A )  * sub( X )  + beta*sub( Y ),  or
*     sub( Y ) := alpha*sub( A )' * sub( X )  + beta*sub( Y ),
*
*  where sub( A ) denotes A(IA:IA+M-1,JA:JA+N-1),
*
*        sub( X ) denotes if TRANS = 'N',
*                       X(IX:IX,JX:JX+N-1), if INCX = M_X,
*                       X(IX:IX+N-1,JX:JX), if INCX = 1 and INCX <> M_X,
*                     else
*                       X(IX:IX,JX:JX+M-1), if INCX = M_X,
*                       X(IX:IX+M-1,JX:JX), if INCX = 1 and INCX <> M_X,
*                     end if
*
*        sub( Y ) denotes if trans = 'N',
*                       Y(IY:IY,JY:JY+M-1), if INCY = M_Y,
*                       Y(IY:IY+M-1,JY:JY), if INCY = 1 and INCY <> M_Y,
*                     else
*                       Y(IY:IY,JY:JY+N-1), if INCY = M_Y,
*                       Y(IY:IY+N-1,JY:JY), if INCY = 1 and INCY <> M_Y,
*                     end if
*
*  alpha and beta are scalars, and sub( X ) and sub( Y ) are distributed
*  vectors and sub( A ) is a M-by-N distributed submatrix.
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
*  Because vectors may be seen as particular matrices, a distributed
*  vector is considered to be a distributed matrix.
*
*  If TRANS = 'N', INCX = M_X and INCY = M_Y, the process column having
*  the first entries of sub( X ) must also contain the first block of
*  sub( A ).  Moreover, the row blocksize of  A  must be equal to the
*  column blocksize of Y, i.e MB_A = NB_Y, and the column blocksize of A
*  must be equal to the column block size of X, i.e NB_A = NB_X.
*  Finally, the column offset of sub( X ) must be equal to the column
*  offset of sub( A ), i.e MOD(JX-1,NB_X) = MOD(JA-1,NB_A), and the row
*  offset of sub( A ) must be equal to the column offset of sub( Y ),
*  i.e MOD(IA-1,MB_A) = MOD(JY-1,NB_Y).
*
*  If TRANS = 'N', INCX = M_X, INCY = 1 and INCY <> M_Y, the process row
*  having the first entries of sub( Y ) must also contain the first
*  block of sub( A ), the process column having the first entries of
*  sub( X ) must also contain the first block of sub( A ). Moreover, the
*  row blocksize of  A  must be equal to the row blocksize of Y, i.e
*  MB_A = MB_Y, and the column blocksize of A must be equal to the
*  column block size of X, i.e NB_A = NB_X. Finally, the column offset
*  of sub( X ) must be equal to the column offset of sub( A ), i.e
*  MOD(JX-1,NB_X) = MOD(JA-1,NB_A), and the row offset of sub( A ) must
*  be equal to the row offset of sub( Y ), i.e
*  MOD(IA-1,MB_A) = MOD(IY-1,MB_Y).
*
*  If TRANS = 'N', INCX = 1, INCX <> M_X and INCY = M_Y, the row
*  blocksize of A must be equal to the column blocksize of Y, i.e
*  MB_A = NB_Y, and the column block size of A must be equal to the row
*  blocksize of X, i.e NB_A = MB_X. Finally, the row offset of sub( X )
*  must be equal to the column offset of sub( A ), i.e
*  MOD(IX-1,MB_X) = MOD(JA-1,NB_A), and the row offset of sub( A ) must
*  be equal to the column offset of sub( Y ), i.e
*  MOD(IA-1,MB_A) = MOD(JY-1,NB_Y).
*
*  If TRANS = 'N', INCX = 1, INCX <> M_X, INCY = 1 and INCY <> M_Y, the
*  process row having the first entries of sub( Y ) must also contain
*  the first block of sub( A ). Moreover, the row blocksize of  A  must
*  be equal to the row blocksize of Y, i.e MB_A = MB_Y, and the column
*  block size of A must be equal to the row block size of X, i.e
*  NB_A = MB_X. Finally, the row offset of sub( X ) must be equal to the
*  column offset of sub( A ), i.e MOD(IX-1,MB_X) = MOD(JA-1,NB_A), and
*  the row offset of sub( A ) must be equal to the row offset of
*  sub( Y ), i.e MOD(IA-1,MB_A) = MOD(IY-1,MB_Y).
*
*  When trans <> 'N', use the previous explanations and replace X by Y,
*  and Y by X everywhere, and sub( A ) by sub( A )'.
*
*  Parameters
*  ==========
*
*  TRANS   (global input) pointer to CHARACTER
*          On entry, TRANS specifies the operation to be performed as
*          follows:
*
*          if TRANS = 'N' or 'n',
*          sub( Y ) := alpha*sub( A )  * sub( X ) + beta*sub( Y ),
*
*          else if TRANS = 'T' or 't',
*          sub( Y ) := alpha*sub( A )' * sub( X ) + beta*sub( Y ),
*
*          else if TRANS = 'C' or 'c',
*          sub( Y ) := alpha*sub( A )' * sub( X ) + beta*sub( Y ).
*
*  M       (global input) pointer to INTEGER
*          The number of rows to be operated on i.e the number of rows
*          of the distributed submatrix sub( A ). M >= 0.
*
*  N       (global input) pointer to INTEGER
*          The number of columns to be operated on i.e the number of
*          columns of the distributed submatrix sub( A ). N >= 0.
*
*  ALPHA   (global input) pointer to DOUBLE PRECISION
*          On entry, ALPHA specifies the scalar alpha.
*
*  A       (local input) DOUBLE PRECISION pointer into the local memory
*          to an array of dimension (LLD_A,LOCc(JA+N-1) containing the
*          local pieces of the distributed matrix sub( A ).
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
*  X       (local input) DOUBLE PRECISION array containing the local
*          pieces of a distributed matrix of dimension of at least
*          if TRANS = 'N' or TRANS = 'n',
*                  ( (JX-1)*M_X + IX + ( N - 1 )*abs( INCX ) )
*          else
*                  ( (JX-1)*M_X + IX + ( M - 1 )*abs( INCX ) )
*          This array contains the entries of the distributed vector
*          sub( X ).
*
*  IX      (global input) pointer to INTEGER
*          The global row index of the submatrix of the distributed
*          matrix X to operate on.
*
*  JX      (global input) pointer to INTEGER
*          The global column index of the submatrix of the distributed
*          matrix X to operate on.
*
*  DESCX   (global and local input) INTEGER array of dimension 8.
*          The array descriptor of the distributed matrix X.
*
*  INCX    (global input) pointer to INTEGER
*          The global increment for the elements of X. Only two values
*          of INCX are supported in this version, namely 1 and M_X.
*
*  BETA    (global input) pointer to DOUBLE PRECISION
*          On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*          supplied as zero then sub( Y ) need not be set on input.
*
*  Y       (local input/local output) DOUBLE PRECISION array
*          containing the local pieces of a distributed matrix of
*          dimension of at least
*          if TRANS = 'N' or TRANS = 'n',
*                ( (JY-1)*M_Y + IY + ( M - 1 )*abs( INCY ) )
*          else
*                ( (JY-1)*M_Y + IY + ( N - 1 )*abs( INCY ) )
*          This array contains the entries of the distributed vector
*          sub( Y ).  On exit, sub( Y ) is overwritten by the updated
*          distributed vector sub( Y ).
*
*  IY      (global input) pointer to INTEGER
*          The global row index of the submatrix of the distributed
*          matrix Y to operate on.
*
*  JY      (global input) pointer to INTEGER
*          The global column index of the submatrix of the distributed
*          matrix Y to operate on.
*
*  DESCY   (global and local input) INTEGER array of dimension 8.
*          The array descriptor of the distributed matrix Y.
*
*  INCY    (global input) pointer to INTEGER
*          The global increment for the elements of Y. Only two values
*          of INCY are supported in this version, namely 1 and M_Y.
*
*  =====================================================================
*
*  .. Local Scalars ..
*/
   char        * rtop, * ctop, TrA, xdist, ydist;
   int         iacol, iarow, icoffa, ictxt, iia, iix, iiy, info, iroffa,
               ixcol, ixrow, iycol, iyrow, jja, jjx, jjy, lcm, lcmp,
               lcmq, locincx, locincy, mm, mp0, mq0, mycol, myrow, nca,
               nn, nota, np0, nprow, npcol, nra, nq0, tmp1, tmp2, wksz;
/* ..
*  .. PBLAS Buffer ..
*/
   double      * buff;
/* ..
*  .. External Functions ..
*/
   void        blacs_gridinfo_();
   void        pbchkmat();
   void        pbchkvect();
   void        pberror_();
   char        * getpbbuf();
   char        * ptop();
   F_VOID_FCT  pbdgemv_();
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
      info = -(800+CTXT_+1);
   else
   {
      TrA = Mupcase( F2C_CHAR( trans )[0] );
      nota = ( TrA == 'N' );
      iroffa = (*ia-1) % desc_A[MB_];
      icoffa = (*ja-1) % desc_A[NB_];
      pbchkmat( *m, 2, *n, 3, *ia, *ja, desc_A, 8, &iia, &jja,
                &iarow, &iacol, nprow, npcol, myrow, mycol,
                &nra, &nca, &info );
      if( nota )
      {
         pbchkvect( *n, 3, *ix, *jx, desc_X, *incx, 12, &iix, &jjx,
                    &ixrow, &ixcol, nprow, npcol, myrow, mycol,
                    &info );
         pbchkvect( *m, 2, *iy, *jy, desc_Y, *incy, 18, &iiy, &jjy,
                    &iyrow, &iycol, nprow, npcol, myrow, mycol,
                    &info );
      }
      else
      {
         pbchkvect( *m, 2, *ix, *jx, desc_X, *incx, 12, &iix, &jjx,
                    &ixrow, &ixcol, nprow, npcol, myrow, mycol,
                    &info );
         pbchkvect( *n, 3, *iy, *jy, desc_Y, *incy, 18, &iiy, &jjy,
                    &iyrow, &iycol, nprow, npcol, myrow, mycol,
                    &info );
      }
      if( info == 0 )
      {
         if( (TrA != 'N') && (TrA != 'T') && (TrA != 'C') )
            info = -1;
         if( nota )
         {
            if( *incx == desc_X[M_] )
            {
               if( ( ((*jx-1) % desc_A[NB_]) != icoffa ) ||
                   ( ixcol != iacol )  )
                  info = -11;
               else if( desc_A[NB_] != desc_X[NB_] )
                  info = -(1200+NB_+1);
            }
            else if( ( *incx == 1 ) && ( *incx != desc_X[M_] ) )
            {
               if( ((*ix-1) % desc_A[NB_]) != icoffa )
                  info = -10;
               else if( desc_A[NB_] != desc_X[MB_] )
                  info = -(1200+MB_+1);
            }
            else
            {
               info = -13;
            }
            if( *incy == desc_Y[M_] )
            {
               if( ((*jy-1) % desc_A[MB_]) != iroffa )
                  info = -17;
               else if( desc_A[MB_] != desc_Y[NB_])
                  info = -(1800+NB_+1);
            }
            else if( ( *incy == 1 ) && ( *incy != desc_Y[M_] ) )
            {
               if( ( ((*iy-1) % desc_A[MB_]) != iroffa ) ||
                   ( iyrow != iarow ) )
                  info = -16;
               else if( desc_A[MB_] != desc_Y[MB_] )
                  info = -(1800+MB_+1);
            }
            else
            {
               info = -19;
            }
         }
         else
         {
            if( *incx == desc_X[M_] )
            {
               if( ((*jx-1) % desc_A[MB_]) != iroffa )
                  info = -11;
               else if( desc_A[MB_] != desc_X[NB_] )
                  info = -(1200+NB_+1);
            }
            else if( ( *incx == 1 ) && ( *incx != desc_X[M_] ) )
            {
               if( ( ((*ix-1) % desc_A[MB_]) != iroffa ) ||
                   ( ixrow != iarow ) )
                  info = -10;
               else if( desc_A[MB_] != desc_X[MB_] )
                  info = -(1200+MB_+1);
            }
            else
            {
               info = -13;
            }
            if( *incy == desc_Y[M_] )
            {
               if( ( ((*jy-1) % desc_A[NB_]) != icoffa ) ||
                   ( iycol != iacol ) )
                  info = -16;
               else if( desc_A[NB_] != desc_Y[NB_] )
                  info = -(1800+NB_+1);
            }
            else if( ( *incy == 1 ) && ( *incy != desc_Y[M_] ) )
            {
               if( ((*iy-1) % desc_A[NB_]) != icoffa )
                  info = -16;
               else if( desc_A[NB_] != desc_Y[MB_] )
                  info = -(1800+MB_+1);
            }
            else
            {
               info = -19;
            }
         }
         if( ictxt != desc_X[CTXT_] )
            info = -(1200+CTXT_+1);
         if( ictxt != desc_Y[CTXT_] )
            info = -(1800+CTXT_+1);
      }
   }
   if( info )
   {
      pberror_( &ictxt, "PDGEMV", &info );
      return;
   }
/*
*  Quick return if possible.
*/
   if( ( *m == 0 ) || ( *n == 0 ) ||
       ( ( *alpha == ZERO ) && ( *beta == ONE ) ) )
      return;
/*
*  Figure out the arguments to be passed to pbdgemv
*/
   mm = *m + iroffa;
   nn = *n + icoffa;

   lcm = ilcm_( &nprow, &npcol );
   if( nota )
   {
      tmp1 = mm / desc_A[MB_];
      mp0 = MYROC0( tmp1, mm, desc_A[MB_], nprow );
      tmp2 = nn / desc_A[NB_];
      nq0 = MYROC0( tmp2, nn, desc_A[NB_], npcol );
      if( *incx == desc_X[M_] )
      {
         xdist = 'R';
         locincx = desc_X[LLD_];
         if( *incy == desc_Y[M_] )
         {
            lcmq = lcm / npcol;
            ydist = 'R';
            locincy = desc_Y[LLD_];
            tmp1 = mm / desc_A[NB_];
            mq0 = MYROC0( tmp1, mm, desc_A[NB_], npcol );
            tmp1 = mq0 / desc_A[NB_];
            tmp1 = MYROC0( tmp1, mq0, desc_A[NB_], lcmq );
            wksz = mp0 + MAX( nq0, tmp1 );
         }
         else
         {
            ydist = 'C';
            locincy = 1;
            wksz = nq0 + mp0;
         }
      }
      else
      {
         lcmq = lcm / npcol;
         xdist = 'C';
         locincx = 1;
         tmp1 = nq0 / desc_A[NB_];
         tmp1 = MYROC0( tmp1, nq0, desc_A[NB_], lcmq );
         if( *incy == desc_Y[M_] )
         {
            lcmp = lcm / nprow;
            ydist = 'R';
            locincy = desc_Y[LLD_];
            tmp1 += nq0;
            tmp2 = mp0 / desc_A[MB_];
            tmp2 = MYROC0( tmp2, mp0, desc_A[MB_], lcmp );
            wksz = mp0 + MAX( tmp1, tmp2 );
         }
         else
         {
            ydist = 'C';
            locincy = 1;
            wksz = nq0 + MAX( mp0, tmp1 );
         }
      }
    }
    else
    {
      tmp1 = mm / desc_A[MB_];
      mp0 = MYROC0( tmp1, mm, desc_A[MB_], nprow );
      tmp2 = nn / desc_A[NB_];
      nq0 = MYROC0( tmp2, nn, desc_A[NB_], npcol );
      if( *incx == desc_X[M_] )
      {
         xdist = 'R';
         locincx = desc_X[LLD_];
         if( *incy == desc_Y[M_] )
         {
            lcmp = lcm / nprow;
            ydist = 'R';
            locincy = desc_Y[LLD_];
            tmp1 = mp0 / desc_A[MB_];
            tmp1 = MYROC0( tmp1, mp0, desc_A[MB_], lcmp );
            wksz = mp0 + MAX( tmp1, nq0 );
         }
         else
         {
            lcmp = lcm / nprow;
            lcmq = lcm / npcol;
            ydist = 'C';
            locincy = 1;
            tmp1 = mp0 / desc_A[MB_];
            tmp1 = mp0 + MYROC0( tmp1, mp0, desc_A[MB_], lcmp );
            tmp2 = nq0 / desc_A[NB_];
            tmp2 = MYROC0( tmp2, nq0, desc_A[NB_], lcmq );
            wksz = nq0 + MAX( tmp1, tmp2 );
         }
      }
      else
      {
         xdist = 'C';
         locincx = 1;
         if( *incy == desc_Y[M_] )
         {
            ydist = 'R';
            locincy = desc_Y[LLD_];
            wksz = mp0 + nq0;
         }
         else
         {
            lcmp = lcm / nprow;
            ydist = 'C';
            locincy = 1;
            tmp1 = nn / desc_A[MB_];
            np0 = MYROC0( tmp1, nn, desc_A[MB_], nprow );
            tmp1 = np0 / desc_A[MB_];
            tmp1 = MYROC0( tmp1, np0, desc_A[MB_], lcmp );
            wksz = nq0 + MAX( mp0, tmp1 );
         }
      }
   }
   buff = (double *)getpbbuf( "PDGEMV", wksz*sizeof(double) );
/*
*  Call PB-BLAS routine
*/
   if( nota && xdist == 'R' )
  {
      ctop = ptop( BROADCAST, COLUMN, TOPGET );
      pbdgemv_( &ictxt, trans, C2F_CHAR( &xdist ), C2F_CHAR( &ydist ),
                m, n, &desc_A[MB_], &desc_A[NB_], &iroffa, &icoffa, alpha,
                &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                &X[iix-1+(jjx-1)*desc_X[LLD_]], &locincx, beta,
                &Y[iiy-1+(jjy-1)*desc_Y[LLD_]], &locincy,
                &iarow, &iacol, &ixrow, &ixcol, &iyrow, &iycol,
                C2F_CHAR( ctop ), C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
   }
   else if( !nota && xdist == 'C' )
   {
      rtop = ptop( BROADCAST, ROW, TOPGET );
      pbdgemv_( &ictxt, trans, C2F_CHAR( &xdist ), C2F_CHAR( &ydist ),
                m, n, &desc_A[MB_], &desc_A[NB_], &iroffa, &icoffa, alpha,
                &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                &X[iix-1+(jjx-1)*desc_X[LLD_]], &locincx, beta,
                &Y[iiy-1+(jjy-1)*desc_Y[LLD_]], &locincy,
                &iarow, &iacol, &ixrow, &ixcol, &iyrow, &iycol,
                C2F_CHAR( rtop ), C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
   }
   else
   {
      pbdgemv_( &ictxt, trans, C2F_CHAR( &xdist ), C2F_CHAR( &ydist ),
                m, n, &desc_A[MB_], &desc_A[NB_], &iroffa, &icoffa, alpha,
                &A[iia-1+(jja-1)*desc_A[LLD_]], &desc_A[LLD_],
                &X[iix-1+(jjx-1)*desc_X[LLD_]], &locincx, beta,
                &Y[iiy-1+(jjy-1)*desc_Y[LLD_]], &locincy,
                &iarow, &iacol, &ixrow, &ixcol, &iyrow, &iycol,
                C2F_CHAR( TOPDEF ), C2F_CHAR( NO ), C2F_CHAR( NO ), buff );
   }
}
