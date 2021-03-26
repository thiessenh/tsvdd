from tsvdd.SVDD import SVDD
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from tsvdd.utils import svmlib_kernel_format
import pytest


class TestSVDD:
    @pytest.mark.parametrize("n_samples", [200, 500])
    def test_novelty_svdd(self, n_samples):
        X, y = make_circles(n_samples=n_samples, factor=.1, noise=0.00001)
        y[y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        inner = y_train == 1
        outer_test = y_test == -1
        X_train_clean = X_train[inner]
        X_test_noise = X_test[outer_test]

        gram_train = np.dot(X_train_clean, X_train_clean.T)
        gram_test = np.dot(X_test_noise, X_train_clean.T)
        gram_diagonal_test = np.diagonal(np.dot(X_test_noise, X_test_noise.T))


        svdd = SVDD('precomputed', C=1, tol=10e-16)
        svdd.fit(gram_train)
        y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
        y_pred_test = np.array(y_pred_test)

        count_test = 0

        for i in range(y_pred_test.shape[0]):
            if y_pred_test[i] != y_test[outer_test][i]:
                count_test += 1

        assert not count_test > 1

    @pytest.mark.parametrize("n_samples", [200, 500])
    def test_svdd(self, n_samples):
        X, y = make_circles(n_samples=n_samples, factor=.1, noise=0.0001)
        y[y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        outlier_ratio_train = y_train[y_train == -1].shape[0] / y_train.shape[0]

        outer_test = y_test == -1
        X_test_noise = X_test[outer_test]


        gram_train = rbf_kernel(X_train, X_train)
        gram_test = rbf_kernel(X_test_noise, X_train)
        gram_diagonal_test = np.diagonal(rbf_kernel(X_test_noise, X_test_noise))
        gram_diagonal_train = np.diagonal(gram_train)

        svdd = SVDD('precomputed', nu=outlier_ratio_train, tol=10e-16)
        svdd.fit(gram_train)
        y_pred_train = svdd.predict(gram_train, gram_diagonal_train)
        y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
        y_pred_test = np.array(y_pred_test)
        y_pred_train = np.array(y_pred_train)

        count_train = 0
        count_test = 0
        for i in range(y_pred_train.shape[0]):
            if y_pred_train[i] != y_train[i]:
                count_train += 1

        for i in range(y_pred_test.shape[0]):
            if y_pred_test[i] != y_test[outer_test][i]:
                count_test += 1

        assert not (count_test > 1 or count_train > 1)


    @pytest.mark.parametrize("n_train", [10, 30, 50])
    @pytest.mark.parametrize("length_train", [5, 10, 15])
    @pytest.mark.parametrize("n_test", [10, 30, 50])
    @pytest.mark.parametrize("length_test", [5, 10, 15])
    @pytest.mark.parametrize("dim", [1, 2, 5])
    def test_sv_svdd(self, n_train, length_train, n_test, length_test, dim):
        return
        rs = np.random.RandomState(1234)
        c_train_matrix = rs.rand(n_train, length_train, dim).astype(dtype=np.float64, order='c')
        c_test_matrix = c_train_matrix + 2

        index = rs.randint(low=0, high=n_train, size=int(n_train/2))

        train = np.concatenate([c_train_matrix, c_test_matrix])
        y_train = np.concatenate([np.ones(n_train), -1*np.ones(n_train)])
        test = np.concatenate([c_train_matrix[index], c_test_matrix[index]])
        y_test = np.concatenate([np.ones(int(n_train/2)), -1 * np.ones(int(n_train/2))])

        svdd = SVDD('tga', nu=0.5)
        svdd.fit(train)
        y_pred_train = svdd.predict(train)
        y_pred_test = svdd.predict(test)

        y_pred_test = np.array(y_pred_test)
        y_pred_train = np.array(y_pred_train)

        count_train =0
        count_test = 0
        for i in range(y_pred_train.shape[0]):
            if y_pred_train[i] != y_train[i]:
                count_train += 1

        for i in range(y_pred_test.shape[0]):
            if y_pred_test[i] != y_test[i]:
                count_test += 1

        assert not (count_test > 1 or count_train > 1)




