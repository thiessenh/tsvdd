from tsvdd import libsvdd
import numpy as np
import pandas as pd
from .ga import test_kernel_matrix, train_kernel_matrix
from .utils import svmlib_kernel_format, sampled_gak_sigma


class SVDD:
    _kernels = ["precomputed", "tga"]

    def __init__(self, kernel='tga', nu=0.05, C=0.02, degree=3, gamma=1,
                 coef0=0.0, tol=1e-5, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200,
                 verbose=True):
        self.kernel = kernel
        self.C = C
        self.support_ = None
        self.support_vectors_ = None
        self._n_support = None
        self.dual_coef_ = None
        self.intercept_ = None
        self._probA = None
        self._probB = None
        self.degree = degree
        self.shrinking = shrinking
        self.fit_shape = None
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.cache_size = cache_size
        self.r_square = .0
        self.kernel = kernel
        self.sigma = sigma
        self.triangular = triangular
        self.normalization_method = normalization_method
        self.nu = nu
        self.verbose = verbose
        self.class_weight_ = np.empty(0, dtype=np.float64)
        self.probability = False
        self.epsilon = 0
        self.is_fit = False
        self.n_SV = None
        self.r_square = None
        self.rho = None

        if kernel not in self._kernels:
            raise ValueError()
        if self.tol < 10e-7:
            raise Warning(f'Small tolerance < {tolerance} might result in long training times.')
        if self.nu <= 0 or self.nu > 1:
            raise ValueError(f'Invalid parameter `nu={self.nu}`.')

    def __str__(self):
        return f'SVDD(kernel={self.kernel}, nu={self.nu}, C={self.C}, sigma={self.sigma},' \
               f'triangular={self.triangular}, normalization_method={self.normalization_method})'

    def fit(self, X, y=None, W=None):
        self.fit_shape = X.shape
        if X.ndim == 3:
            n_instances, n_length, n_dim = X.shape
        elif X.ndim == 2:
            n_instances, n_length = X.shape
        self.X_fit = self._check_X(X)
        if self.kernel == 'tga':
            if self.sigma == 'auto':
                sigmas = sampled_gak_sigma(X, 100)
                self.sigma = sigmas[3]
            if self.triangular == 'auto':
                self.triangular = .5 * n_length
        if y is not None:
            raise NotImplementedError('Y not yet implemented')
        y = np.ones(X.shape[0], dtype=np.float64)

        if self.kernel == 'tga':
            gram_matrix = train_kernel_matrix(self.X_fit, self.sigma, self.triangular, self.normalization_method)
        elif self.kernel == 'precomputed':
            if n_instances != n_length:
                raise ValueError("n_instances != n_length")
        if W is None:
            W = np.ones(n_instances, dtype=np.float64)
        elif W.ndim !=1:
            raise ValueError("W.ndim !=1")

        self.support_, self.support_vectors_, self._n_support, \
        self.dual_coef_, self.intercept_, self._probA, \
        self._probB, self.r_square, self.rho = libsvdd.fit(
            X, y,
            svm_type=5, sample_weight=W,
            class_weight=self.class_weight_, kernel='precomputed', C=self.C,
            nu=self.nu, probability=self.probability, degree=self.degree,
            shrinking=self.shrinking, tol=self.tol,
            cache_size=self.cache_size, coef0=self.coef0,
            gamma=self.gamma, epsilon=self.epsilon)
        self.is_fit = True


    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X, K_xx_s=None, dec_vals=False):
        if not self.is_fit:
            raise AttributeError('SVDD not fitted.')
        if X.ndim == 3:
            n_instances, n_length, n_dim = X.shape
        elif X.ndim == 2:
            n_instances, n_length = X.shape
        X = self._check_X(X)
        K_xx_s = self._check_X(K_xx_s)
        if K_xx_s.ndim != 1:
            raise ValueError('K_xx_s ndim unequal 1')
        if self.kernel == 'tga':
            gram_matrix = test_kernel_matrix(self.X_fit, X, self.sigma, self.triangular, self.normalization_method)
            if self.normalization_method == 'exp':
                gram_diagonal_test = np.ones(n_instances)
        elif self.kernel == 'precomputed':
            if K_xx_s is None:
                raise ValueError()
        if dec_vals:
            return libsvdd.decision_function(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                degree=self.degree, coef0=self.coef0, gamma=self.gamma,
                cache_size=self.cache_size)
        else:
            return libsvdd.predict(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                degree=self.degree, coef0=self.coef0, gamma=self.gamma,
                cache_size=self.cache_size)

    def decision_function(self, X, K_xx_s=None):
        p_val = self.predict(X, K_xx_s, dec_vals=True)
        return p_val

    @staticmethod
    def _check_X(X):
        if isinstance(X, pd.DataFrame) or (isinstance(X, np.ndarray) and not X.flags['C_CONTIGUOUS']):
            X = np.ascontiguousarray(X)
        return X
