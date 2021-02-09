from .libsvm import svmutil

import numpy as np
import pandas as pd
from .ga import test_kernel_matrix, train_kernel_matrix
from .utils import svmlib_kernel_format, sampled_gak_sigma


class SVDD:

    def __init__(self, kernel='tga', outlier_ratio=0.05, C=None, sigma='auto', triangular='auto', tolerance=10e-5, normalization_method='exp'):
        self.model = None
        self.X_fit = None
        self.fit_shape = None
        self.kernel = kernel
        self.sigma = sigma
        self.triangular = triangular
        self.normalization_method = normalization_method
        self.C = C
        self.outlier_ratio = outlier_ratio
        self.tolerance = tolerance
        if tolerance < 10e-5:
            raise Warning(f'Small tolerance < {tolerance} might result in long training times.')

    def __str__(self):
        return f'SVDD(kernel={self.kernel}, outlier_ratio={self.outlier_ratio}, C={self.C}, sigma={self.sigma},' \
               f'triangular={self.triangular}, normalization_method={self.normalization_method})'

    def fit(self, X):
        self.fit_shape = X.shape
        n_instances, n_length, n_dim = X.shape
        if self.C is None:
            self.C = 1 / (self.outlier_ratio * n_instances)
        self.X_fit = self._check_X(X)
        if self.sigma == 'auto':
            sigmas = sampled_gak_sigma(X, 100)
            self.sigma = sigmas[3]
        if self.triangular == 'auto':
            self.triangular = .5 * X.shape[1]
        gram_matrix = train_kernel_matrix(self.X_fit, self.sigma, self.triangular, self.normalization_method)
        gram_matrix_libsvm = svmlib_kernel_format(gram_matrix)
        prob = svmutil.svm_problem(np.ones(n_instances), gram_matrix_libsvm, isKernel=True)
        param = svmutil.svm_parameter(f'-s 5 -t 4 -c {self.C} -e {self.tolerance}')
        self.model = svmutil.svm_train(prob, param)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X_test):
        if self.fit_shape is None:
            raise AttributeError('SVDD not fitted.')
        n_instances, n_length, n_dim = X_test.shape
        X_test = self._check_X(X_test)
        gram_matrix = test_kernel_matrix(self.X_fit, X_test, self.sigma, self.triangular, self.normalization_method)
        gram_matrix_libsvm = svmlib_kernel_format(gram_matrix)
        if self.normalization_method == 'exp':
            gram_diagonal_test = np.ones(n_instances)
        y_pred, p_val = svmutil.svm_predict(gram_matrix_libsvm, gram_diagonal_test, self.model)
        return y_pred

    @staticmethod
    def _check_X(X):
        if isinstance(X, pd.DataFrame) or (isinstance(X, np.ndarray) and not X.flags['C_CONTIGUOUS']):
            X = np.ascontiguousarray(X)
        return X
