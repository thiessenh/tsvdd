import pytest
import numpy as np
from tsvdd.kernels import train_gds_dtw, test_gds_dtw
from dtaidistance import dtw


class TestKernels:

    @pytest.mark.parametrize("n_train", [10, 30, 50])
    @pytest.mark.parametrize("length_train", [5, 10, 15])
    def test_dtw_train(self, n_train, length_train):
        rs = np.random.RandomState(1234)
        c_train_matrix = rs.rand(n_train, length_train).astype(dtype=np.float64, order='c')
        res = np.zeros(shape=(n_train, n_train))
        sigma = 2
        for i in range(n_train):
            seq_1 = c_train_matrix[i]
            for j in range(n_train):
                seq_2 = c_train_matrix[j]
                res[i, j] = dtw.distance_fast(seq_1, seq_2)
        res = np.exp(-np.divide(np.power(res.ravel(), 2), np.power(sigma, 2))).reshape(
            (n_train, n_train))
        res_cython = train_gds_dtw(c_train_matrix, sigma)
        np.testing.assert_array_equal(res, res_cython)

    @pytest.mark.parametrize("n_train", [10, 30, 50])
    @pytest.mark.parametrize("length_train", [5, 10, 15])
    @pytest.mark.parametrize("n_test", [10, 30, 50])
    @pytest.mark.parametrize("length_test", [5, 10, 15])
    def test_dtw_predict(self, n_train, length_train, n_test, length_test):
        rs = np.random.RandomState(1234)

        c_train_matrix = rs.rand(n_train, length_train).astype(dtype=np.float64, order='c')
        c_test_matrix = rs.rand(n_test, length_test).astype(dtype=np.float64, order='c')
        res = np.zeros(shape=(n_test, n_train), dtype=np.float64)
        sigma = 2

        for i in range(n_test):
            seq_1 = c_test_matrix[i]
            for j in range(n_train):
                seq_2 = c_train_matrix[j]
                res[i, j] = dtw.distance_fast(seq_1, seq_2)
        res = np.exp(-np.divide(np.power(res.ravel(), 2), np.power(sigma, 2))).reshape(
            (n_test, n_train))

        K_xx_s_ = np.ones(n_test, dtype=np.float64, order='C')
        for i in range(n_test):
            seq_1 = c_test_matrix[i]
            K_xx_s_[i] = dtw.distance_fast(seq_1, seq_1)
        K_xx_s_ = np.exp(-np.divide(np.power(K_xx_s_.ravel(), 2), np.power(sigma, 2))).reshape(
            n_test)

        res_cython, K_xx_s_cython_ = test_gds_dtw(c_train_matrix, c_test_matrix, sigma)

        np.testing.assert_array_equal(res, res_cython)
        np.testing.assert_array_equal(K_xx_s_, K_xx_s_cython_)
