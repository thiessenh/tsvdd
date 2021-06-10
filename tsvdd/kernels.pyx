"""
Wrapper around the Global Alignment kernel code from M. Cuturi

Original code: http://www.iip.ist.i.kyoto-u.ac.jp/member/cuturi/GA.html

Written by Adrien Gaidon - INRIA - 2011
http://lear.inrialpes.fr/people/gaidon/

LICENSE: cf. logGAK.c
"""

from dtaidistance import dtw
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
        @param normalization_method: 'exp'

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


def train_gds_dtw(np.ndarray[np.double_t,ndim=2] seq, double sigma):
    """
    RBF Kernel with DTW as distance substitute.
    @param seq:
    @return:
    """
    # get data dimensions
    cdef int n_instances = seq.shape[0]
    cdef int nLength = seq.shape[1]

    # check preconditions
    assert seq.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma

    res = dtw.distance_matrix_fast(seq)

    # dtaidistance==2.1.2 returns upper triangular, rest is inf
    # set info to zero
    res[res == np.inf] = 0
    # mirror along diagonal
    res = res + res.T - np.diag(np.diag(res))

    return np.exp(-np.divide(np.power(res.ravel(), 2), np.power(sigma, 2))).reshape(
        (n_instances, n_instances))


def test_gds_dtw(np.ndarray[np.double_t,ndim=2] train, np.ndarray[np.double_t,ndim=2] test, double sigma):
    """
    RBF Kernel with DTW as distance substitute.
    @param train: data from fit
    @param test: prediction data
    @param sigma: sigma, determine with, e.g., Rule of Cuturi
    @return:
    """
    # get data dimensions
    cdef int n_instances_train = train.shape[0]
    cdef int n_length_train = train.shape[1]
    cdef int n_instances_test = test.shape[0]
    cdef int n_length_test = test.shape[1]

    # check preconditions
    assert train.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert test.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"
    assert sigma > 0, "Invalid bandwidth sigma (%f)" % sigma

    if n_length_train == n_length_test:
        big_boy_array = np.vstack((test, train))
        big_boy_array = big_boy_array.astype(dtype=np.float64, order='C')

        assert big_boy_array.flags['C_CONTIGUOUS'], "Invalid series: not C-contiguous"

        block = ((0, n_instances_test), (n_instances_test, big_boy_array.shape[0]))
        res = dtw.distance_matrix_fast(big_boy_array, block=block)
        res_block = res[0:n_instances_test, n_instances_test:big_boy_array.shape[0]]
        K_xx_s_ = np.ones(n_instances_test, dtype=np.float64, order='C')
        for i in range(n_instances_test):
            seq_1 = test[i]
            K_xx_s_[i] = dtw.distance_fast(seq_1, seq_1)
        K_xx_s = np.exp(-np.divide(np.power(K_xx_s_.ravel(), 2), np.power(sigma, 2))).reshape(
            n_instances_test)

        return np.exp(-np.divide(np.power(res_block.ravel(), 2), np.power(sigma, 2))).reshape(
            (n_instances_test, n_instances_train)), K_xx_s
    else:

        res = np.zeros(shape=(n_instances_test, n_instances_train), dtype=np.float64, order='C')
        for i in range(n_instances_test):
            seq_1 = test[i]
            for j in range(n_instances_train):
                seq_2 = train[j]
                res[i, j] = dtw.distance_fast(seq_1, seq_2)
        res = np.exp(-np.divide(np.power(res.ravel(), 2), np.power(sigma, 2))).reshape(
            (n_instances_test, n_instances_train))

        K_xx_s_ = np.ones(n_instances_test, dtype=np.float64, order='C')
        for i in range(n_instances_test):
            seq_1 = test[i]
            K_xx_s_[i] = dtw.distance_fast(seq_1, seq_1)
        K_xx_s_ = np.exp(-np.divide(np.power(K_xx_s_.ravel(), 2), np.power(sigma, 2))).reshape(
            n_instances_test)
        return res, K_xx_s_


