//
// Created by haiko on 07.02.21.
//
#ifndef GA_MATRIXLOGGAK_H
#define GA_MATRIXLOGGAK_H

#include <stdlib.h>
#include "logGAK.h"
#include <omp.h>
#include <stdio.h>

extern void trainGramMatrixExp(double *seq, int nInstances, int nLength, int nDim, double *res, double sigma, int triangular);

extern void testGramMatrixExp(double *train, double *test, int nInstances_train, int nInstances_test, int nLength_train,
                       int nLength_test, int nDim, double *res, double sigma, int triangular, int64_t *sv_indices, int64_t sv_size);


#endif //GA_MATRIXLOGGAK_H


