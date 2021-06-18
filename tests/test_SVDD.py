from tsvdd import SVDD
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
import pytest
import pandas as pd


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

        svdd = SVDD('precomputed', C=1, tol=1e-18)
        svdd.fit(gram_train)
        y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
        y_pred_test = np.array(y_pred_test)

        false_test =  np.sum(y_pred_test != y_test[outer_test])

        assert not false_test > 1

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
        gram_diagonal_test = np.ones(X_test_noise.shape[0])
        gram_diagonal_train = np.ones(gram_train.shape[0])

        svdd = SVDD('precomputed', nu=outlier_ratio_train, tol=1e-18)
        svdd.fit(gram_train)
        y_pred_train = svdd.predict(gram_train, gram_diagonal_train)
        y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
        y_pred_test = np.array(y_pred_test)
        y_pred_train = np.array(y_pred_train)

        false_train =  np.sum(y_pred_train != y_train)
        false_test =  np.sum(y_pred_test != y_test[outer_test])

        assert not (false_test > 1 or false_train > 1)

    def test_some_instantiations(self):
        rng = np.random.default_rng(42)
        X_2d = rng.random((20, 100))
        X_3d = rng.random((20, 100, 1))
        
        svdd = SVDD(kernel='tga', nu=0.05, sigma=1)
        svdd.fit(X_2d)
        svdd.fit(X_3d)

        svdd = SVDD(kernel='gds-dtw', nu=0.05, sigma=1)
        svdd.fit(X_2d)

        svdd = SVDD(kernel='rbf-gak', nu=0.05, sigma=1)
        svdd.fit(X_2d)

        X_2d = pd.DataFrame(X_2d)

        svdd = SVDD(kernel='tga', nu=0.05, sigma=1)
        svdd.fit(X_2d)

        svdd = SVDD(kernel='gds-dtw', nu=0.05, sigma=1)

        svdd = SVDD(kernel='rbf-gak', nu=0.05, sigma=1)
        svdd.fit(X_2d)
