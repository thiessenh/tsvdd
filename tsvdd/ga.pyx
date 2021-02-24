"""
Wrapper around the Global Alignment kernel code from M. Cuturi

Original code: http://www.iip.ist.i.kyoto-u.ac.jp/member/cuturi/GA.html

Written by Adrien Gaidon - INRIA - 2011
http://lear.inrialpes.fr/people/gaidon/

LICENSE: cf. logGAK.c
"""

import numpy as np
cimport numpy as np

cdef extern from "logGAK.h" nogil:
    double logGAK(double *seq1 , double *seq2, int nX, int nY, int dimvect, double sigma, int triangular)

cdef extern from "matrixLogGAK.h" nogil:
    void trainGramMatrixExp(double *seq, int nInstances, int nLength, int nDim, double *res, double sigma, int triangular)
    void testGramMatrixExp(double *train, double *test, int nInstances_train, int nInstances_test, int nLength_train, int nLength_test, int nDim, double *res, double sigma, int triangular, signed long *sv_indices, signed long sv_size)

def tga_dissimilarity(np.ndarray[np.double_t,ndim=2] seq1, np.ndarray[np.double_t,ndim=2] seq2, double sigma, int triangular):
    """ Compute the Triangular Global Alignment (TGA) dissimilarity score

    What is computed is minus the log of the normalized global alignment kernel
    evaluation between the two time series, in order to use in a RBF kernel as
    exp(-gamma*mlnk)

    PARAMETERS:
      seq1: T1 x d multivariate (dimension d) time series of duration T1
        two-dimensional C-contiguous (ie. read line by line) numpy array of doubles,

      seq2: T2 x d multivariate (dimension d) time series of duration T2

      sigma: double, bandwitdh of the inner distance kernel
        good practice: {0.1, 0.5, 1, 2, 5, 10} * median(dist(x, y)) * sqrt(median(length(x)))

      triangular: int, parameter to restrict the paths (closer to the diagonal) used by the kernel
        good practice: {0.25, 0.5} * median(length(x))
        Notes:
          * 0: use all paths
          * 1: measuring alignment of (same duration) series, ie
            kernel value is 0 for different durations
          * higher = more restricted thus faster
          * kernel value is also 0 for series with difference in duration > triangular-1

    RETURN:
      mlnk: double,
        minus the normalized log-kernel
        (logGAK(seq1,seq1)+logGAK(seq2,seq2))/2 - logGAK(seq1,seq2)

    """
    T1 = seq1.shape[0]
    T2 = seq2.shape[0]
    d  = seq1.shape[1]
    _d = seq2.shape[1]
    # check preconditions
    assert d == _d, "Invalid series: dimension mismatch (%d != %d)" % (d, _d)
    assert seq1.flags['C_CONTIGUOUS'] and seq2.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
    assert triangular >= 0, "Invalid triangular parameter (%f)" % triangular
    # compute the global alignment kernel value
    ga12 = logGAK(<double*> seq1.data, <double*> seq2.data, <int> T1, <int> T2, <int> d, sigma, triangular)
    # compute the normalization factor
    ga11 = logGAK(<double*> seq1.data, <double*> seq1.data, <int> T1, <int> T1, <int> d, sigma, triangular)
    ga22 = logGAK(<double*> seq2.data, <double*> seq2.data, <int> T2, <int> T2, <int> d, sigma, triangular)
    nf = 0.5*(ga11+ga22)
    # return minus the normalized logarithm of the Global Alignment Kernel
    mlnk = nf - ga12
    return mlnk


def tga_similarity(np.ndarray[np.double_t,ndim=2] seq1, np.ndarray[np.double_t,ndim=2] seq2, double sigma, int triangular):
    """ Compute the Triangular Global Alignment (TGA) similarity score

    What is computed is the global alignment kernel evaluation between the two time series,
    in order to use precomputed kernel. The greater the more similar.

    PARAMETERS:
      seq1: T1 x d multivariate (dimension d) time series of duration T1
        two-dimensional C-contiguous (ie. read line by line) numpy array of doubles,

      seq2: T2 x d multivariate (dimension d) time series of duration T2

      sigma: double, bandwitdh of the inner distance kernel
        good practice: {0.1, 0.5, 1, 2, 5, 10} * median(dist(x, y)) * sqrt(median(length(x)))

      triangular: int, parameter to restrict the paths (closer to the diagonal) used by the kernel
        good practice: {0.25, 0.5} * median(length(x))
        Notes:
          * 0: use all paths
          * 1: measuring alignment of (same duration) series, ie
            kernel value is 0 for different durations
          * higher = more restricted thus faster
          * kernel value is also 0 for series with difference in duration > triangular-1

    RETURN:
      mlnk: double,
        minus the normalized log-kernel
        (logGAK(seq1,seq1)+logGAK(seq2,seq2))/2 - logGAK(seq1,seq2)

    """
    T1 = seq1.shape[0]
    T2 = seq2.shape[0]
    d  = seq1.shape[1]
    _d = seq2.shape[1]
    # check preconditions
    assert d == _d, "Invalid series: dimension mismatch (%d != %d)" % (d, _d)
    assert seq1.flags['C_CONTIGUOUS'] and seq2.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
    assert triangular >= 0, "Invalid triangular parameter (%f)" % triangular
    # compute the global alignment kernel value
    ga12 = logGAK(<double*> seq1.data, <double*> seq2.data, <int> T1, <int> T2, <int> d, sigma, triangular)
    return ga12

def train_kernel_matrix(np.ndarray[np.double_t,ndim=3] seq, double sigma, int triangular, str normalization_method):
    """ Compute the Triangular Global Alignment (TGA) similarity score

    What is computed is the global alignment kernel evaluation between the two time series,
    in order to use precomputed kernel.

    PARAMETERS:
      seq: T1 x d multivariate (dimension d) time series of duration T1
        two-dimensional C-contiguous (ie. read line by line) numpy array of doubles,


      sigma: double, bandwitdh of the inner distance kernel
        good practice: {0.1, 0.5, 1, 2, 5, 10} * median(dist(x, y)) * sqrt(median(length(x)))

      triangular: int, parameter to restrict the paths (closer to the diagonal) used by the kernel
        good practice: {0.25, 0.5} * median(length(x))
        Notes:
          * 0: use all paths
          * 1: measuring alignment of (same duration) series, ie
            kernel value is 0 for different durations
          * higher = more restricted thus faster
          * kernel value is also 0 for series with difference in duration > triangular-1

    RETURN:
      mlnk: double,
        minus the normalized log-kernel
        (logGAK(seq1,seq1)+logGAK(seq2,seq2))/2 - logGAK(seq1,seq2)

    """
    # get data dimensions
    cdef int nInstances  = seq.shape[0]
    cdef int nLength = seq.shape[1]
    cdef int nDim  = seq.shape[2]

    # check preconditions
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
    assert triangular >= 0, "Invalid triangular parameter (%f)" % triangular
    assert seq.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert normalization_method in ['exp'], f'normalization_method=`{normalization_method}` is not a valid argument.'

    # define matrix for kernel values
    cdef np.ndarray[np.double_t,ndim=2] res = np.zeros([nInstances, nInstances], order='C')

    if normalization_method == 'exp':
        trainGramMatrixExp(<double*> seq.data, <int> nInstances, <int> nLength, <int> nDim, <double*> res.data, <double> sigma, <int> triangular)
    else:
        raise ValueError(f'normalization_method=`{normalization_method}` is not a valid argument.')

    return np.exp(-res)

def test_kernel_matrix(np.ndarray[np.double_t, ndim=3] train, np.ndarray[np.double_t, ndim=3] test, double sigma, int triangular, str normalization_method, np.ndarray[np.int64_t, ndim=1] sv_indices):
    """ Compute the Triangular Global Alignment (TGA) similarity score

    What is computed is the global alignment kernel evaluation between the two time series,
    in order to use precomputed kernel.

    PARAMETERS:
      seq1: T1 x d multivariate (dimension d) time series of duration T1
        two-dimensional C-contiguous (ie. read line by line) numpy array of doubles,

      seq2: T2 x d multivariate (dimension d) time series of duration T2

      sigma: double, bandwitdh of the inner distance kernel
        good practice: {0.1, 0.5, 1, 2, 5, 10} * median(dist(x, y)) * sqrt(median(length(x)))

      triangular: int, parameter to restrict the paths (closer to the diagonal) used by the kernel
        good practice: {0.25, 0.5} * median(length(x))
        Notes:
          * 0: use all paths
          * 1: measuring alignment of (same duration) series, ie
            kernel value is 0 for different durations
          * higher = more restricted thus faster
          * kernel value is also 0 for series with difference in duration > triangular-1

      sv_indices: Indices starting at 0.

    RETURN:
      mlnk: double,
        minus the normalized log-kernel
        (logGAK(seq1,seq1)+logGAK(seq2,seq2))/2 - logGAK(seq1,seq2)

    """
    # get data dimensions
    cdef int nInstances_train = train.shape[0]
    cdef int nLength_train = train.shape[1]
    cdef int nDim = train.shape[2]
    cdef signed long sv_size = sv_indices.shape[0]

    cdef int nInstances_test= test.shape[0]
    cdef int nLength_test = test.shape[1]

    assert nDim == test.shape[2]

    # check preconditions
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
    assert triangular >= 0, "Invalid triangular parameter (%f)" % triangular
    assert train.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert test.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert normalization_method in ['exp'], f'normalization_method=`{normalization_method}` is not a valid argument.'
    assert sv_size <= nInstances_train, "There can't be more support vectors than train instances."

    # define matrix for kernel values
    cdef np.ndarray[np.double_t, ndim=2] res = np.zeros([nInstances_test, nInstances_train], order='C')

    if normalization_method == 'exp':
        testGramMatrixExp(<double*> train.data, <double*> test.data, <int> nInstances_train, <int> nInstances_test, <int> nLength_train, <int> nLength_test, <int> nDim, <double*> res.data, <double> sigma,
                          <int> triangular, <signed long*> sv_indices.data, <signed long> sv_size)
    else:
        raise ValueError(f'normalization_method=`{normalization_method}` is not a valid argument.')
    return np.exp(-res)


# def diagonal_gram_matrix(np.ndarray[np.double_t,ndim=3] seq, double sigma, int triangular, str normalization_method):
#     cdef int nInstances = seq.shape[0]
#     cdef int nLength = seq.shape[1]
#     cdef int nDim = seq.shape[2]
#
#     # check preconditions
#     assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma
#     assert triangular >= 0, "Invalid triangular parameter (%f)" % triangular
#     assert seq.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
#     assert normalization_method in ['exp'], f'normalization_method=`{normalization_method}` is not a valid argument.'
#
#     # define matrix for diagonal kernel values
#     cdef np.ndarray[np.double_t, ndim=1] res = np.zeros([nInstances], order='C')
#
#     if normalization_method == 'exp':
#         diagonalGramMatrixExp(<double*> seq.data, <int> nInstances, <int> nLength, <int> nDim, <double*> res.data, <double> sigma,
#                               <int> triangular)
#     else:
#         raise ValueError(f'normalization_method=`{normalization_method}` is not a valid argument.')
#     return res