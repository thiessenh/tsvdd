import pytest
import numpy as np
from tsvdd.kernels import tga_dissimilarity, train_kernel_matrix, test_kernel_matrix


class TestKernel:

    @pytest.mark.parametrize("n_train", [10, 30, 50])
    @pytest.mark.parametrize("length_train", [5, 10, 15])
    @pytest.mark.parametrize("n_test", [10, 30, 50])
    @pytest.mark.parametrize("length_test", [5, 10, 15])
    @pytest.mark.parametrize("dim", [1, 2, 5])
    def test_train_matrix(self, n_train, length_train, n_test, length_test, dim):
        rs = np.random.RandomState(1234)
        c_train_matrix = rs.rand(n_train, length_train, dim).astype(dtype=np.float64, order='c')
        res = np.zeros(shape=(n_train, n_train))
        sigma = 2
        triangular = 0
        for i in range(n_train):
            seq_1 = c_train_matrix[i]
            for j in range(n_train):
                seq_2 = c_train_matrix[j]
                res[i, j] = tga_dissimilarity(seq_1, seq_2, sigma, triangular)
        res = np.exp(-res)
        res_cython = train_kernel_matrix(c_train_matrix, sigma, triangular, 'exp')
        np.testing.assert_array_equal(res, res_cython)

    @pytest.mark.parametrize("n_train", [10, 30, 50])
    @pytest.mark.parametrize("length_train", [5, 10, 15])
    @pytest.mark.parametrize("n_test", [10, 30, 50])
    @pytest.mark.parametrize("length_test", [5, 10, 15])
    @pytest.mark.parametrize("dim", [1, 2, 5])
    @pytest.mark.parametrize("sv_indices", ['all', 'random'])
    def test_predict_matrix(self, n_train, length_train, n_test, length_test, dim, sv_indices):
        rs = np.random.RandomState(1234)
        if sv_indices is 'all':
            indices = np.arange(n_train, dtype=np.int64)
        elif sv_indices is 'random':
            indices = rs.randint(low=0, high=n_train, size=int(n_train/2))

        c_train_matrix = rs.rand(n_train, length_train, dim).astype(dtype=np.float64, order='c')
        c_test_matrix = rs.rand(n_test, length_test, dim).astype(dtype=np.float64, order='c')
        res = np.zeros(shape=(n_test, n_train), dtype=np.float64)
        sigma = 2
        triangular = 0
        for i in range(n_test):
            seq_1 = c_test_matrix[i]
            for j in range(n_train):
                if j in indices:
                    seq_2 = c_train_matrix[j]
                    res[i, j] = tga_dissimilarity(seq_1, seq_2, sigma, triangular)
        res = np.exp(-res)
        indices = np.sort(indices).astype(dtype=np.int64, order='c')

        res_cython = test_kernel_matrix(c_train_matrix, c_test_matrix, sigma, triangular, 'exp', indices)
        np.testing.assert_array_equal(res, res_cython)
