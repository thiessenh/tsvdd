//
// Created by haiko on 07.02.21.
//

#ifndef GA_MATRIXLOGGAK_H
#define GA_MATRIXLOGGAK_H

#endif //GA_MATRIXLOGGAK_H


void trainGramMatrixExp(double *seq, int nInstances, int nLength, int nDim, double *res, double sigma, int triangular);

void testGramMatrixExp(double *train, double *test, int nInstances_train, int nInstances_test, int nLength_train,
                       int nLength_test, int nDim, double *res, double sigma, int triangular);
