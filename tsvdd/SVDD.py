from tsvdd import libsvdd
import numpy as np
import pandas as pd
from .ga import test_kernel_matrix, train_kernel_matrix
from .utils import svmlib_kernel_format, sampled_gak_sigma
import time
from dtaidistance import dtw


class SVDD:
    _kernels = ["precomputed", "tga", "gds_dtw", "rbf"]

    def __init__(self, kernel='tga', nu=None, C=0.02, degree=3, gamma=1,
                 coef0=0.0, tol=1e-4, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200,
                 verbose=True):
        self.kernel = kernel
        self.C = C
        # from 1 ... model->l
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
        self.kernel_duration = None
        self.svdd_duration = None
        self.train_gram = None

        if kernel not in self._kernels:
            raise ValueError(f'Unknown kernel:{kernel}')
        if self.tol < 10e-7:
            raise Warning(f'Small tolerance < {tol} might result in long training times.')

    def __str__(self):
        return f'SVDD(kernel={self.kernel}, nu={self.nu}, C={self.C}, sigma={self.sigma},' \
               f'triangular={self.triangular}, normalization_method={self.normalization_method})'

    def fit(self, X, y=None, W=None):
        """
        X is either data or kernel matrix. y is not yet implemented.
        W are instance weights, primarily used for active learning.
        """

        # distinguish between precomputed and built-in kernels
        if self.kernel == 'precomputed':
            self.train_gram = self._check_kernel_matrix(X, is_fit=True)
        else:
            self.fit_shape = X.shape
            # check if input data are valid for kernel computation
            X = self._check_input_array(X)
            self.X_fit = X
            # check if valid arguments
            self._check_arguments()
            if self.X_fit.ndim == 3:
                n_instances, n_length, n_dim = self.X_fit.shape
            elif self.X_fit.ndim == 2:
                n_instances, n_length = self.X_fit.shape

            # start computing kernels
            if self.kernel == 'tga':
                start = time.time()
                _X = train_kernel_matrix(self.X_fit, self.sigma, self.triangular, self.normalization_method)
                self.train_gram = _X
                self.kernel_duration = time.time() - start
            elif self.kernel == 'gds_dtw':
                # GDS_{DTW}(x_i, x_j) = \exp (- \frac{DTW(x_i, x_j)^ 2}{\sigma^2})
                X_ = np.ones((n_instances, n_instances), dtype=np.float64, order='c')
                for i in range(n_instances):
                    seq_1 = self.X_fit[i]
                    for j in range(n_instances):
                        seq_2 = self.X_fit[j]
                        # DTW(x_i, x_j)
                        X_[i, j] = dtw.distance_fast(seq_1, seq_2)
                # \exp (- \frac{X_^ 2}{\sigma^2})
                self.train_gram = np.exp(-np.divide(np.power(X_.ravel(), 2), np.power(self.sigma, 2))).reshape((n_instances, n_instances))
            elif self.kernel == 'rbf':
                # GDS_{DTW}(x_i, x_j) = \exp (- \frac{||x_i, x_j||^ 2}{\sigma^2})
                X_ = np.ones((n_instances, n_instances), dtype=np.float64, order='c')
                for i in range(n_instances):
                    seq_1 = self.X_fit[i]
                    for j in range(n_instances):
                        seq_2 = self.X_fit[j]
                        # ||x_i, x_j||
                        X_[i, j] = np.linalg.norm(seq_1 - seq_2)
                # \exp (- \frac{X_^ 2}{\sigma^2})
                self.train_gram = np.exp(-np.divide(np.power(X_.ravel(), 2), np.power(self.sigma, 2))).reshape((n_instances, n_instances))

        # check y
        if y is not None:
            raise NotImplementedError('Y not yet implemented')
        y = np.ones(n_instances, dtype=np.float64)

        # check W
        if W is None:
            W = np.ones(n_instances, dtype=np.float64)
        elif W.ndim !=1:
            raise ValueError("W.ndim !=1")

        # start training
        start = time.time()
        self.support_, self.support_vectors_, self._n_support, \
        self.dual_coef_, self.intercept_, self._probA, \
        self._probB, self.r_square, self.rho = libsvdd.fit(
            self.train_gram, y,
            svm_type=5, sample_weight=W,
            class_weight=self.class_weight_, kernel='precomputed', C=self.C,
            nu=self.nu, probability=self.probability, degree=self.degree,
            shrinking=self.shrinking, tol=self.tol,
            cache_size=self.cache_size, coef0=self.coef0,
            gamma=self.gamma, epsilon=self.epsilon)

        self.svdd_duration = time.time() - start
        self.is_fit = True

    def fit_predict(self, X):
        """
        Fit then predict.
        """
        self.fit(X)
        return self.predict(X)

    def predict(self, X, K_xx_s=None, dec_vals=False):
        """
        Predict from data or from kernel matrix. K_xx_s is the `diagonal` and must be provided when kernel=precomputed.
        When dec_vals=True, `distance` to radius is returned instead of labels.
        """
        # fit before precit
        if not self.is_fit:
            raise AttributeError('SVDD not fitted.')

        # distinguish between precomputed and built-in kernels
        if self.kernel == 'precomputed':
            X, K_xx_s = self._check_kernel_matrix(X, is_predict=True, K_xx=K_xx_s)
        else:
            # check if input data are valid for kernel computation
            X = self._check_input_array(X)
            if X.ndim == 3:
                n_instances, n_length, n_dim = X.shape
            elif X.ndim == 2:
                n_instances, n_length = X.shape

            if self.kernel == 'tga':
                # sv_indices are passed to gram matrix computation
                sv_indices = np.sort(self.support_).astype(dtype=np.int64, order='C')
                # libsvm starts counting with 1
                sv_indices = sv_indices - 1
                # special case when fit and predict data are equal
                if np.array_equal(self.X_fit, X):
                    gram_matrix = self.train_gram
                else:
                    gram_matrix = test_kernel_matrix(self.X_fit, X, self.sigma, self.triangular, self.normalization_method, sv_indices)
                X = gram_matrix
                if self.normalization_method == 'exp':
                    K_xx_s = np.ones(n_instances)
                else:
                    gram_diagonal_test = train_kernel_matrix(self.X_fit, self.sigma, self.triangular, self.normalization_method)
                    gram_diagonal_test = np.diagonal(gram_diagonal_test)
                    K_xx_s = gram_diagonal_test
            elif self.kernel == 'gds_dtw':
                # GDS_{DTW}(x_i, x_j) = \exp (- \frac{DTW(x_i, x_j)^ 2}{\sigma^2})
                sv_indices = np.sort(self.support_).astype(dtype=np.int64, order='C')
                sv_indices = sv_indices - 1
                X_ = np.ones((n_instances, self.fit_shape[0]), dtype=np.float64, order='C')
                for i in range(n_instances):
                    seq_1 = X[i]
                    for j in sv_indices:
                        seq_2 = self.X_fit[j]
                        # DTW(x_i, x_j)
                        X_[i, j] = dtw.distance_fast(seq_1, seq_2)
                # \exp (- \frac{X_^ 2}{\sigma^2})
                X_ = np.exp(-np.divide(np.power(X_.ravel(), 2), np.power(self.sigma, 2)))
                # calculate gram diagonal
                K_xx_s_ = np.ones(n_instances, dtype=np.float64, order='C')
                for i in range(n_instances):
                    seq_1 = X[i]
                    K_xx_s_[i] = dtw.distance_fast(seq_1, seq_1)
                K_xx_s = np.exp(-np.divide(np.power(K_xx_s_.ravel(), 2), np.power(self.sigma, 2)))
                X = X_.reshape((n_instances, self.fit_shape[0]))
            elif self.kernel == 'rbf':
                # GDS_{DTW}(x_i, x_j) = \exp (- \frac{||x_i, x_j||^ 2}{\sigma^2})
                sv_indices = np.sort(self.support_).astype(dtype=np.int64, order='C')
                sv_indices = sv_indices - 1
                X_ = np.ones((n_instances, self.fit_shape[0]), dtype=np.float64, order='C')
                for i in range(n_instances):
                    seq_1 = X[i]
                    for j in sv_indices:
                        seq_2 = self.X_fit[j]
                        # ||x_i, x_j||
                        X_[i, j] = np.linalg.norm(seq_1 - seq_2)
                # \exp (- \frac{X_^ 2}{\sigma^2})
                X_ = np.exp(-np.divide(np.power(X_.ravel(), 2), np.power(self.sigma, 2)))
                # calculate gram diagonal
                K_xx_s_ = np.ones(n_instances, dtype=np.float64, order='C')
                for i in range(n_instances):
                    seq_1 = X[i]
                    K_xx_s_[i] = np.linalg.norm(seq_1 - seq_1)
                K_xx_s = np.exp(-np.divide(np.power(K_xx_s_.ravel(), 2), np.power(self.sigma, 2)))
                X = X_.reshape((n_instances, self.fit_shape[0]))

        if dec_vals:
            score = libsvdd.decision_function(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                degree=self.degree, coef0=self.coef0, gamma=self.gamma,
                cache_size=self.cache_size)
        else:
            score = libsvdd.predict(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                degree=self.degree, coef0=self.coef0, gamma=self.gamma,
                cache_size=self.cache_size)

        return score

    def decision_function(self, X, K_xx_s=None):
        p_val = self.predict(X, K_xx_s, dec_vals=True)
        return p_val

    def _info(self, line):
        if self.verbose:
            print(line)

    def _check_arguments(self):
        """
        Check if C is valid. If \nu is set, calculate C from \nu.
        If sigma=`auto` or traingular=`auto`, calculate accordingly.
        """
        n_instances = self.fit_shape[0]
        n_length = self.fit_shape[1]
        if self.C < (1 / n_instances):
            self.C = (1 / n_instances)
            raise Warning(f'C too small, set C to {self.C}')
        if self.nu:
            if self.nu <= 0 or self.nu > 1:
                raise ValueError(f'Invalid parameter `nu={self.nu}`.')
            else:
                self.C = 1 / (self.nu * n_instances)
        else:
            # necessary as Cython needs a valid float
            self.nu = 0.5

        if self.kernel == 'tga':
            if self.sigma == 'auto':
                sigmas = sampled_gak_sigma(self.X_fit, 100)
                self.sigma = sigmas[3]
            if self.triangular == 'auto':
                self.triangular = .5 * n_length

    def _check_kernel_matrix(self, gram_matrix, is_fit=True, is_predict=False, K_xx=None):
        """
        Check if user provided gram matrix has correct shape and is c-contiguous.
        """
        # check shapes for predict
        if is_predict:
            if K_xx is None:
                raise ValueError('K_xx can not be None')
            if gram_matrix[0] != K_xx.shape[0]:
                raise ValueError("Diagonal of gram matrix has wrong length.")
            if self.fit_shape[0] != gram_matrix[1]:
                raise ValueError("Prediction matrix does not fit train matrix.")
            is_fit = False
        # check shapes during fit
        if is_fit:
            if gram_matrix.shape[0] != gram_matrix.shape[0]:
                raise ValueError("Kernel matrix should be symmetric.")
        # check if C_CONTIGUOUS
        if not gram_matrix.flags['C_CONTIGUOUS']:
            gram_matrix = np.ascontiguousarray(gram_matrix, dtype=np.float64)
        if K_xx is not None and not K_xx.flags['C_CONTIGUOUS']:
            K_xx = np.ascontiguousarray(K_xx, dtype=np.float64)
        if K_xx.ndim != 1:
            raise ValueError('K_xx_s ndim unequal 1')
        return gram_matrix, K_xx

    def _check_input_array(self, X):
        """
        Check if X has correct shape for kernel and is c-contiguous.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.kernel == 'tga':
            if X.ndim != 3:
                raise ValueError("Input array X has wrong shape. Should be 3-tuple (n_instances, n_length, n_dim)")
        if self.kernel == 'RBF':
            if X.ndim != 2:
                raise ValueError("Input array X has wrong shape. Should be 2-tuple (n_instances, n_length)")
        if self.kernel == 'gds_dtw':
            if X.ndim != 2:
                raise ValueError("Input array X has wrong shape. Should be 2-tuple (n_instances, n_length)")
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X, dtype=np.float64)

        return X

