#include "matrixLogGAK.h"

int g_nInstances;
int g_nLength_train;
int g_nLength_test;
int g_nDim;

int resOffset(int i, int j) {
    return i * g_nInstances + j;
}

int seqOffset_train(int i) {
    return i * g_nLength_train * g_nDim;
}

int seqOffset_test(int i) {
    return i * g_nLength_test * g_nDim;
}

/* Compute the kernel matrix of a seq with equal length time series data
*
* seq is a flat array that contains the data in form nInstances, nLength, nDim
* nInstances is the amount of instances
* nLength is the length of the the series
* nDim is the dimension
* res is a flat array that contains the kernel matrix in form nInstances x nInstances
* sigma is the bandwidth parameter for the gauss-ish kernel
* triangular is the parameter to trade off global alignment vs. optimal alignment, i.e., more diagonal alignment
*/
void trainGramMatrixExp(double *seq, int nInstances, int nLength, int nDim, double *res, double sigma, int triangular) {
    g_nInstances = nInstances;
    g_nLength_train = nLength;
    g_nDim = nDim;

    double *cache = (double *) malloc(nInstances * sizeof(double));
    int j = 0;
    int i = 0;

    if (cache == NULL) {
        printf("Error! memory not allocated.");
        exit(1);
    }
    // compute GAK with itself
#pragma omp parallel for private(i)
    for (i = 0; i < nInstances; i++) {
        int seq_i = seqOffset_train(i);
        cache[i] = logGAK((double *) &seq[seq_i], (double *) &seq[seq_i], nLength, nLength, nDim, sigma, triangular);
    }
#pragma omp parallel for private(i,j)
    for (i = 0; i < nInstances; i++) {
        int seq_i = seqOffset_train(i);
        for (j = i; j < nInstances; j++) {
            int seq_j = seqOffset_train(j);

            // compute the global alignment kernel value
            double ga12 = logGAK((double *) &seq[seq_i], (double *) &seq[seq_j], nLength, nLength, nDim, sigma,
                                 triangular);
            // compute the normalization factor
            //double ga11 = logGAK((double*) &seq[seq_i], (double*) &seq[seq_i], nLength, nLength, nDim, sigma, triangular);
            double ga11 = cache[i];
            //double ga22 = logGAK((double*) &seq[seq_j], (double*) &seq[seq_j], nLength, nLength, nDim, sigma, triangular);
            double ga22 = cache[j];
            double nf = 0.5 * (ga11 + ga22);
            double mlnk = nf - ga12;
            int res_ij = resOffset(i, j);
            int res_ji = resOffset(j, i);
            res[res_ij] = mlnk;
            res[res_ji] = mlnk;
        }
    }
    free(cache);
}

void testGramMatrixExp(double *train, double *test, int nInstances_train, int nInstances_test, int nLength_train,
                       int nLength_test, int nDim, double *res ,double sigma, int triangular, int64_t *sv_indices, int64_t sv_size) {
    g_nInstances = nInstances_train;
    g_nLength_test = nLength_test;
    g_nLength_train = nLength_train;
    g_nDim = nDim;

    int i = 0;
    int l = 0;
    double *cache_train = (double *) malloc(nInstances_train * sizeof(double));
    double *cache_test = (double *) malloc(nInstances_test * sizeof(double));

    if (cache_train == NULL || cache_test == NULL) {
        printf("Error! memory not allocated.");
        exit(1);
    }
    // compute GAK with itself
#pragma omp parallel for private(l)
    for (l = 0; l < sv_size; l++) {
        int i = sv_indices[l];
        int seq_i = seqOffset_train(i);
        cache_train[i] = logGAK((double *) &train[seq_i], (double *) &train[seq_i], nLength_train, nLength_train, nDim,
                                sigma, triangular);
    }
#pragma omp parallel for private(i)
    for (i = 0; i < nInstances_test; i++) {
        int seq_i = seqOffset_test(i);
        cache_test[i] = logGAK((double *) &test[seq_i], (double *) &test[seq_i], nLength_test, nLength_test, nDim,
                               sigma, triangular);
    }

#pragma omp parallel for private(i,l)
    for (i = 0; i < nInstances_test; i++) {
        int seq_test = seqOffset_test(i);
        for (l = 0; l < sv_size; l++) {
            int j = sv_indices[l];
            int seq_train = seqOffset_train(j);

            // compute the global alignment kernel value
            double ga12 = logGAK((double *) &train[seq_train], (double *) &test[seq_test], nLength_train, nLength_test,
                                 nDim, sigma, triangular);
            // compute the normalization factor
            //double ga11 = logGAK((double*) &train[seq_train], (double*) &train[seq_train], nLength_train, nLength_train, nDim, sigma, triangular);
            double ga11 = cache_train[j];
            //double ga22 = logGAK((double*) &test[seq_test], (double*) &test[seq_test], nLength_test, nLength_test, nDim, sigma, triangular);
            double ga22 = cache_test[i];
            double nf = 0.5 * (ga11 + ga22);
            double mlnk = nf - ga12;
            int res_k = resOffset(i, j);
            res[res_k] = mlnk;
        }
    }
    free(cache_test);
    free(cache_train);
}

//void diagonalGramMatrixExp(double *seq, int nInstances, int nLength, int nDim, double *res, double sigma, int triangular)
//{
//    g_nInstances = nInstances;
//    g_nLength_train = nLength;
//    g_nDim = nDim;
//
//    int i = 0;
//
//    // compute GAK with itself
//    #pragma omp parallel for private(i)
//    for(i = 0; i < nInstances; i++){
//        int seq_i = seqOffset_train(i);
//        // compute the global alignment kernel value
//        double ga11 = logGAK((double*) &seq[seq_i], (double*) &seq[seq_i], nLength, nLength, nDim, sigma, triangular);
//        double nf = 0.5*(ga11+ga11);
//        double mlnk = nf - ga11;
//          res[i] = mlnk;
//    }
//
//}

