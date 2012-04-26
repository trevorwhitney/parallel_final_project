#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>
#include <math.h>

#define MIN(a,b) ( (a) < (b) ? (a) : (b))

double calculate_error(double *my_c, double *atlas_c, int n);
double my_dgemm(int n, double *a, double *b, double *c);
double my_cblas_dgemm(int n, double *a, double *b, double *c);
double step01(int n, double *a, double *b, double *c, int t);
double step02(int n, double *a, double *b, double *c, int t);
double step03(int n, double *a, double *b, double *c, int t);
double step04(int n, double *a, double *b, double *c, int t);
double step05(int n, double *a, double *b, double *c, int t);

double calculate_error(double *my_c, double *atlas_c, int n) {
  int i, j;
  double numerator;
  double denomenator;
  double error;

  numerator = 0.0;
  denomenator = 0.0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      numerator += pow((atlas_c[i + n*j] - my_c[i + n*j]), 2);
      denomenator += pow((atlas_c[i + n*j]), 2);
    }
  }

  error = sqrt(numerator)/sqrt(denomenator);
  return error;
}

double my_dgemm (int n, double *a, double *b, double *c) {
  //iterators
  int i, j ,k;
  clock_t start, end;
  double elapsed_time;

  start = clock();
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        c[i+n*j] = c[i+n*j] + a[i+n*k] * b[k+n*j];
      }
    }
  }
  end = clock();
  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double my_cblas_dgemm(int n, double *a, double *b, double *c) {
  clock_t start, end;
  double elapsed_time;
  
  start = clock();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    n, n, n, 1, a, n, b, n, 0, c, n);
  end = clock();

  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double step01(int n, double *a, double *b, double *c, int t) {
  int i, j ,k, ii, kk;
  clock_t start, end;
  double elapsed_time;
  register double r;

  start = clock();
  for (kk = 0; kk < n; kk += t) {
    for (ii = 0; ii < n; ii += t) {
      for (j = 0; j < n; j++) {
        for (k = kk; k < MIN(kk+t, n); k++) {
          r = b[n*j+k];
          for (i = ii; i < MIN(ii+t, n); i++) {
            c[j*n+i] = c[n*j+i] + r * a[n*k+i];
          }
        }
      }
    }
  }
  end = clock();
  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double step02(int n, double *a, double *b, double *c, int t) {
  int i, j ,k, ii, jj, kk;
  clock_t start, end;
  double elapsed_time;
  register double r;

  start = clock();
  for (jj = 0; jj < n; jj += t) {
    for (kk = 0; kk < n; kk += t) {
      for (ii = 0; ii < n; ii += t) {
        for (j = jj; j < MIN(jj+t, n); j++) {
          for (k = kk; k < MIN(kk+t, n); k++) {
            r = b[n*j+k];
            for (i = ii; i < MIN(ii+t, n); i++) {
              c[n*j+i] = c[n*j+i] + r * a[n*k+i];
            }
          }
        }
      }
    }
  }
  end = clock();
  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double step03(int n, double *a, double *b, double *c, int t) {
  int i, j ,k, ii, jj, kk, nxj;
  clock_t start, end;
  double elapsed_time;
  register double r;

  start = clock();
  for (jj = 0; jj < n; jj += t) {
    for (kk = 0; kk < n; kk += t) {
      for (ii = 0; ii < n; ii += t) {
        for (j = jj; j < MIN(jj+t, n); j++) {
          nxj = n*j;
          for (i = ii; i < MIN(ii+t, n); i++) {
            r = c[nxj+i];
            for (k = kk; k < MIN(kk+t, n); k++) {
              r = r + b[nxj+k] * a[n*k+i];
            }
            c[nxj+i] = r;
          }
        }
      }
    }
  }
  end = clock();
  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double step04(int n, double *a, double *b, double *c, int t) {
  int i, j ,k, ii, jj, kk, dif, nxj;
  clock_t start, end;
  double elapsed_time;
  register double r;

  start = clock();
  for (jj = 0; jj < n; jj += t) {
    for (kk = 0; kk < n; kk += t) {
      for (ii = 0; ii < n; ii += t) {
        for (j = jj; j < MIN(jj+t, n); j++) {
          nxj = n*j;
          for (i = ii; i < MIN(ii+t, n); i++) {
            r = c[nxj+i];
            for (k = kk; k < MIN(kk+t, n)-8; k+=8) {
              r = r + b[nxj+k] * a[n*k+i];
              r = r + b[nxj+k+1] * a[n*(k+1)+i];
              r = r + b[nxj+k+2] * a[n*(k+2)+i];
              r = r + b[nxj+k+3] * a[n*(k+3)+i];
              r = r + b[nxj+k+4] * a[n*(k+4)+i];
              r = r + b[nxj+k+5] * a[n*(k+5)+i];
              r = r + b[nxj+k+6] * a[n*(k+6)+i];
              r = r + b[nxj+k+7] * a[n*(k+7)+i];
            }
            for (k; k < MIN(kk+t, n); k++) {
              r = r + b[nxj+k] * a[n*k+i];
            }
            c[nxj+i] = r;
          }
        }
      }
    }
  }
  end = clock();

  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}

double step05(int n, double *a, double *b, double *c, int t) {
  int i, j ,k, ii, jj, kk, nxj0, nxj1, nxj2;
  clock_t start, end;
  double elapsed_time;
  double rc0, rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8, rc9, rc10, rc11, rb0, rb1, 
    rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11, rb12, rb13, rb14, rb15, 
    rb16, rb17, rb18, rb19, rb20, rb21, rb22, rb23, rb24;

  start = clock();
  for (jj = 0; jj < n; jj += t) {
    for (kk = 0; kk < n; kk += t) {
      for (ii = 0; ii < n; ii += t) {
        for (j = jj; j < MIN(jj+t, n); j+=2) {
          nxj0 = n*j;
          nxj1 = n*(j+1);
          for (i = ii; i < MIN(ii+t, n)-4; i+=4) {
            rc0 = c[nxj0+i];
            rc1 = c[nxj1+i];
            
            rc2 = c[nxj0+i+1];
            rc3 = c[nxj1+i+1];

            rc4 = c[nxj0+i+2];
            rc5 = c[nxj1+i+2];

            rc6 = c[nxj0+i+3];
            rc7 = c[nxj1+i+3];
            for (k = kk; k < MIN(kk+t, n)-8; k+=8) {
              rc0 = rc0 + b[nxj0+k] * a[n*k+i];
              rc1 = rc1 + b[nxj1+k] * a[n*k+i];
              rc2 = rc2 + b[nxj0+k] * a[n*k+i+1];
              rc3 = rc3 + b[nxj1+k] * a[n*k+i+1];
              rc4 = rc4 + b[nxj0+k] * a[n*k+i+2];
              rc5 = rc5 + b[nxj1+k] * a[n*k+i+2];
              rc6 = rc6 + b[nxj0+k] * a[n*k+i+3];
              rc7 = rc7 + b[nxj1+k] * a[n*k+i+3];

              rc0 = rc0 + b[nxj0+k+1] * a[n*(k+1)+i];
              rc1 = rc1 + b[nxj1+k+1] * a[n*(k+1)+i];
              rc2 = rc2 + b[nxj0+k+1] * a[n*(k+1)+i+1];
              rc3 = rc3 + b[nxj1+k+1] * a[n*(k+1)+i+1];
              rc4 = rc4 + b[nxj0+k+1] * a[n*(k+1)+i+2];
              rc5 = rc5 + b[nxj1+k+1] * a[n*(k+1)+i+2];
              rc6 = rc6 + b[nxj0+k+1] * a[n*(k+1)+i+3];
              rc7 = rc7 + b[nxj1+k+1] * a[n*(k+1)+i+3];

              rc0 = rc0 + b[nxj0+k+2] * a[n*(k+2)+i];
              rc1 = rc1 + b[nxj1+k+2] * a[n*(k+2)+i];
              rc2 = rc2 + b[nxj0+k+2] * a[n*(k+2)+i+1];
              rc3 = rc3 + b[nxj1+k+2] * a[n*(k+2)+i+1];
              rc4 = rc4 + b[nxj0+k+2] * a[n*(k+2)+i+2];
              rc5 = rc5 + b[nxj1+k+2] * a[n*(k+2)+i+2];
              rc6 = rc6 + b[nxj0+k+2] * a[n*(k+2)+i+3];
              rc7 = rc7 + b[nxj1+k+2] * a[n*(k+2)+i+3];

              rc0 = rc0 + b[nxj0+k+3] * a[n*(k+3)+i];
              rc1 = rc1 + b[nxj1+k+3] * a[n*(k+3)+i];
              rc2 = rc2 + b[nxj0+k+3] * a[n*(k+3)+i+1];
              rc3 = rc3 + b[nxj1+k+3] * a[n*(k+3)+i+1];
              rc4 = rc4 + b[nxj0+k+3] * a[n*(k+3)+i+2];
              rc5 = rc5 + b[nxj1+k+3] * a[n*(k+3)+i+2];
              rc6 = rc6 + b[nxj0+k+3] * a[n*(k+3)+i+3];
              rc7 = rc7 + b[nxj1+k+3] * a[n*(k+3)+i+3];

              rc0 = rc0 + b[nxj0+k+4] * a[n*(k+4)+i];
              rc1 = rc1 + b[nxj1+k+4] * a[n*(k+4)+i];
              rc2 = rc2 + b[nxj0+k+4] * a[n*(k+4)+i+1];
              rc3 = rc3 + b[nxj1+k+4] * a[n*(k+4)+i+1];
              rc4 = rc4 + b[nxj0+k+4] * a[n*(k+4)+i+2];
              rc5 = rc5 + b[nxj1+k+4] * a[n*(k+4)+i+2];
              rc6 = rc6 + b[nxj0+k+4] * a[n*(k+4)+i+3];
              rc7 = rc7 + b[nxj1+k+4] * a[n*(k+4)+i+3];

              rc0 = rc0 + b[nxj0+k+5] * a[n*(k+5)+i];
              rc1 = rc1 + b[nxj1+k+5] * a[n*(k+5)+i];
              rc2 = rc2 + b[nxj0+k+5] * a[n*(k+5)+i+1];
              rc3 = rc3 + b[nxj1+k+5] * a[n*(k+5)+i+1];
              rc4 = rc4 + b[nxj0+k+5] * a[n*(k+5)+i+2];
              rc5 = rc5 + b[nxj1+k+5] * a[n*(k+5)+i+2];
              rc6 = rc6 + b[nxj0+k+5] * a[n*(k+5)+i+3];
              rc7 = rc7 + b[nxj1+k+5] * a[n*(k+5)+i+3];

              rc0 = rc0 + b[nxj0+k+6] * a[n*(k+6)+i];
              rc1 = rc1 + b[nxj1+k+6] * a[n*(k+6)+i];
              rc2 = rc2 + b[nxj0+k+6] * a[n*(k+6)+i+1];
              rc3 = rc3 + b[nxj1+k+6] * a[n*(k+6)+i+1];
              rc4 = rc4 + b[nxj0+k+6] * a[n*(k+6)+i+2];
              rc5 = rc5 + b[nxj1+k+6] * a[n*(k+6)+i+2];
              rc6 = rc6 + b[nxj0+k+6] * a[n*(k+6)+i+3];
              rc7 = rc7 + b[nxj1+k+6] * a[n*(k+6)+i+3];

              rc0 = rc0 + b[nxj0+k+7] * a[n*(k+7)+i];
              rc1 = rc1 + b[nxj1+k+7] * a[n*(k+7)+i];
              rc2 = rc2 + b[nxj0+k+7] * a[n*(k+7)+i+1];
              rc3 = rc3 + b[nxj1+k+7] * a[n*(k+7)+i+1];
              rc4 = rc4 + b[nxj0+k+7] * a[n*(k+7)+i+2];
              rc5 = rc5 + b[nxj1+k+7] * a[n*(k+7)+i+2];
              rc6 = rc6 + b[nxj0+k+7] * a[n*(k+7)+i+3];
              rc7 = rc7 + b[nxj1+k+7] * a[n*(k+7)+i+3];
            }
            //k cleanup
            for (k; k < MIN(kk+t, n); k++) {
              rc0 = rc0 + b[nxj0+k] * a[n*k+i];
              rc1 = rc1 + b[nxj1+k] * a[n*k+i];
              rc2 = rc2 + b[nxj0+k] * a[n*k+i+1];
              rc3 = rc3 + b[nxj1+k] * a[n*k+i+1];
              rc4 = rc4 + b[nxj0+k] * a[n*k+i+2];
              rc5 = rc5 + b[nxj1+k] * a[n*k+i+2];
              rc6 = rc6 + b[nxj0+k] * a[n*k+i+3];
              rc7 = rc7 + b[nxj1+k] * a[n*k+i+3];
            }
            c[nxj0+i] = rc0;
            c[nxj1+i] = rc1;

            c[nxj0+i+1] = rc2;
            c[nxj1+i+1] = rc3;

            c[nxj0+i+2] = rc4;
            c[nxj1+i+2] = rc5;

            c[nxj0+i+3] = rc6;
            c[nxj1+i+3] = rc7;
          }
          //i cleanup
          for (i; i < MIN(ii+t, n); i++) {
            rc0 = c[nxj0+i];
            rc1 = c[nxj1+i];
            for (k = kk; k < MIN(kk+t, n)-8; k+=8) {
              rc0 = rc0 + b[nxj0+k] * a[n*k+i];
              rc1 = rc1 + b[nxj1+k] * a[n*k+i];

              rc0 = rc0 + b[nxj0+k+1] * a[n*(k+1)+i];
              rc1 = rc1 + b[nxj1+k+1] * a[n*(k+1)+i];

              rc0 = rc0 + b[nxj0+k+2] * a[n*(k+2)+i];
              rc1 = rc1 + b[nxj1+k+2] * a[n*(k+2)+i];

              rc0 = rc0 + b[nxj0+k+3] * a[n*(k+3)+i];
              rc1 = rc1 + b[nxj1+k+3] * a[n*(k+3)+i];

              rc0 = rc0 + b[nxj0+k+4] * a[n*(k+4)+i];
              rc1 = rc1 + b[nxj1+k+4] * a[n*(k+4)+i];
              
              rc0 = rc0 + b[nxj0+k+5] * a[n*(k+5)+i];
              rc1 = rc1 + b[nxj1+k+5] * a[n*(k+5)+i];
              
              rc0 = rc0 + b[nxj0+k+6] * a[n*(k+6)+i];
              rc1 = rc1 + b[nxj1+k+6] * a[n*(k+6)+i];

              rc0 = rc0 + b[nxj0+k+7] * a[n*(k+7)+i];
              rc1 = rc1 + b[nxj1+k+7] * a[n*(k+7)+i];
            }
            //k cleanup
            for (k; k < MIN(kk+t, n); k++) {
              rc0 = rc0 + b[nxj0+k] * a[n*k+i];
              rc1 = rc1 + b[nxj1+k] * a[n*k+i];
            }
            c[nxj0+i] = rc0;
            c[nxj1+i] = rc1;
          }
        }
      }
    }
  }
  end = clock();

  elapsed_time = ((double) (end-start))/CLOCKS_PER_SEC;
  return elapsed_time;
}