//
// Created by haiko on 07.02.21.
//
#ifndef GA_LOGGAK_H
#define GA_LOGGAK_H

#include <stdlib.h>
#include <math.h>
/* Useful constants */
#define LOG0 -10000          /* log(0) */
#define LOGP(x, y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))

double logGAK(double *seq1 , double *seq2, int nX, int nY, int dimvect, double sigma, int triangular);


#endif //GA_LOGGAK_H

