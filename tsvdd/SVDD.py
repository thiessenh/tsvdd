from .libsvm import svmutil

import numpy as np
import pandas as pd
from .ga import test_kernel_matrix, train_kernel_matrix
from .utils import svmlib_kernel_format, sampled_gak_sigma


class SVDD:

    def __init__(self, kernel='tga', nu=0.05, sigma='auto', triangular='auto', tolerance=10e-5, normalization_method='exp'):
        self.model = None
        self.X_fit = None
        self.fit_shape = None
        self.r_square = .0
        self.kernel = kernel
        self.sigma = sigma
        self.triangular = triangular
        self.normalization_method = normalization_method
        self.nu = nu
        self.tolerance = tolerance
        if tolerance < 10e-5:
            raise Warning(f'Small tolerance < {tolerance} might result in long training times.')
        if self.nu <= 0 or self.nu > 1:
            raise ValueError(f'Invalid parameter `nu={self.nu}`.')

    def __str__(self):
        return f'SVDD(kernel={self.kernel}, nu={self.nu}, C={self.C}, sigma={self.sigma},' \
               f'triangular={self.triangular}, normalization_method={self.normalization_method})'

    def fit(self, X, W=None):
        self.fit_shape = X.shape
        if len(X.shape) == 3:
            n_instances, n_length, n_dim = X.shape
        elif len(X.shape) == 2:
            n_instances, n_length = X.shape
        self.X_fit = self._check_X(X)
        if self.kernel == 'tga':
            if self.sigma == 'auto':
                sigmas = sampled_gak_sigma(X, 100)
                self.sigma = sigmas[3]
            if self.triangular == 'auto':
                self.triangular = .5 * X.shape[1]

        if self.kernel == 'tga':
            gram_matrix = train_kernel_matrix(self.X_fit, self.sigma, self.triangular, self.normalization_method)
            gram_matrix_libsvm = svmlib_kernel_format(gram_matrix)
        elif self.kernel == 'precomputed':
            gram_matrix_libsvm = svmlib_kernel_format(X)
        prob = svmutil.svm_problem(np.ones(n_instances), gram_matrix_libsvm, isKernel=True, W=W)
        param = svmutil.svm_parameter(f'-s 5 -t 4 -n {self.nu} -e {self.tolerance}')
        self.model = svmutil.svm_train(prob, param)
        self.r_square = self.model.get_r2()


    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X_test, X_diag=None, dec_vals=False):
        if self.fit_shape is None:
            raise AttributeError('SVDD not fitted.')
        if len(X_test.shape) == 3:
            n_instances, n_length, n_dim = X_test.shape
        elif len(X_test.shape) == 2:
            n_instances, n_length = X_test.shape
        X_test = self._check_X(X_test)
        if self.kernel == 'tga':
            gram_matrix = test_kernel_matrix(self.X_fit, X_test, self.sigma, self.triangular, self.normalization_method)
            gram_matrix_libsvm = svmlib_kernel_format(gram_matrix)
            if self.normalization_method == 'exp':
                gram_diagonal_test = np.ones(n_instances)
        elif self.kernel == 'precomputed':
            gram_matrix_libsvm = svmlib_kernel_format(X_test)
            gram_diagonal_test = X_diag

        y_pred, p_val = svmutil.svm_predict(gram_matrix_libsvm, gram_diagonal_test, self.model)
        if dec_vals:
            return np.array(y_pred), np.array(p_val)
        else:
            return np.array(y_pred)

    def decision_function(self, X, X_diag=None):
        y_pred, p_val = self.predict(X, X_diag, dec_vals=True)
        return p_val

    @staticmethod
    def _check_X(X):
        if isinstance(X, pd.DataFrame) or (isinstance(X, np.ndarray) and not X.flags['C_CONTIGUOUS']):
            X = np.ascontiguousarray(X)
        return X
