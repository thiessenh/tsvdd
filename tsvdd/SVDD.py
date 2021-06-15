from numpy.lib.arraysetops import isin
from tsvdd import libsvdd
import numpy as np
import pandas as pd
from .kernels import test_kernel_matrix, train_kernel_matrix, test_gds_dtw, train_gds_dtw
from .utils import sampled_gak_sigma, compute_rbf_kernel, rbf_kernel_fast, rbf_kernel_fast_test
import time
from dtaidistance import dtw
import warnings


class SVDD:
    """
    Support Vector Data Description
    """
    _kernels = ["precomputed", "tga", "gds-dtw", "rbf", "rbf-gak"]

    def __init__(self, kernel='tga', nu=0.05, C=None, tol=1e-4, alpha=.9,
                 sigma='auto', triangular='auto', normalization_method='exp',
                 shrinking=False, cache_size=200, max_iter=1000000, verbose=False):
        """Support Vector Data Description

        Parameters
        ----------
        kernel : str, optional
            Choose among kernels ["precomputed", "tga", "gds-dtw", "rbf", "rbf-gak], by default 'tga'.
        nu : float, optional
            Expected outlier ratio. Amount of observations considered as outliers, by default 0.05.
        C : [type], optional
            If \nu not provided, C will be used. C can be calculated as 1/(outlier_ratio * n_instances), by default None.
        tol : [type], optional
            Stopping criteria for SMO optimization, by default 1e-4.
        alpha : float, optional
            Ration of GA kernel used for RBF-GAK, by default .9.
        sigma : str, optional
            Sigma for Gaussian-like kernels, by default 'auto' which uses Cuturi's heuristic with a multiplier of 1.5.
        triangular : str, optional
            TGA's triangular parameter, by default 'auto', which corresponds to 0.5 * median time series length.
            For smaller values, TGA behaves like a Gaussian kernel, whereas for larger values TGA behaves like the GA kernel.
        normalization_method : str, optional
            Right now only 'exp' implemented, by default 'exp'.
        shrinking : bool, optional
            Whether to use the shrinking heuristic, by default False.
        cache_size : int, optional
            Specify the size of the kernel cache (in MB), by default 200.
        max_iter : int, optional
            [description], by default 1000000.
        verbose : bool, optional
            Enable verbose output, by default False.
        """

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
        self.shrinking = shrinking
        self.fit_shape = None
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size
        self.r_square = .0
        self.kernel = kernel
        self.sigma = sigma
        self.triangular = triangular
        self.normalization_method = normalization_method
        self.nu = nu
        self.verbose = verbose
        self.is_fit = False
        self.n_SV = None
        self.r_square = None
        self.rho = None
        self.kernel_duration = None
        self.svdd_duration = None
        self.train_gram = None
        self.alpha = alpha

    def __str__(self):
        attributes = list(
            map(lambda kv: f'"{kv[0]}"={kv[1]},', self.get_params().items()))
        return "SVDD(" + ''.join(attributes)[:-1] + ")"

    def fit(self, X, y=None, W=None):
        """Fit the model to the training data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Should have shape of (n_instances, time_series_length, n_dim=1).
        y : np.ndarray, optional
            Not yet implemented, by default None.
        W : np.ndarray, optional
            Instance weights with shape (n_instances), by default None.

        Returns
        -------
        self
            Fitted model.
        """
        if self.kernel not in self._kernels:
            raise ValueError(f'Unknown kernel:{self.kernel}')
        if self.tol > 10e-3 and self.verbose:
            warnings.warn(
                f'Large tolerance = {self.tol} might lead to poor results.', Warning)
        if self.tol < 10e-7 and self.verbose:
            warnings.warn(
                f'Small tolerance = {self.tol} might result in long training times.', Warning)

        # distinguish between precomputed and built-in kernels
        if self.kernel == 'precomputed':
            self.train_gram, _ = self._check_kernel_matrix(X, is_fit=True)
            n_instances = self.train_gram.shape[0]
            self.fit_shape = self.train_gram.shape
            # check if valid arguments
            self._check_arguments()
        elif self.is_fit and np.array_equal(self.X_fit, X):
            # SVDD already fitted; if X equals X_fit, gram_train can be reused
            n_instances = X.shape[0]
            self.fit_shape = X.shape
            # check if valid arguments
            self._check_arguments()
        else:
            # check if input data are valid for kernel computation
            X = self._check_input_array(X)
            self.fit_shape = X.shape
            self.X_fit = X
            # check if valid arguments
            self._check_arguments()
            n_instances = self.X_fit.shape[0]

            # start computing kernels
            if self.kernel == 'tga':
                start = time.time()
                _X = train_kernel_matrix(
                    self.X_fit, self._sigma, self._triangular, self.normalization_method)
                self.train_gram = _X
                self.kernel_duration = time.time() - start
            elif self.kernel == 'gds-dtw':
                X_ = train_gds_dtw(self.X_fit, self._sigma)
                self.train_gram = X_
            elif self.kernel == 'rbf':
                X_ = rbf_kernel_fast(self.X_fit), self._sigma
                self.train_gram = X_
            elif self.kernel == "rbf-gak":
                start = time.time()
                X_c = np.reshape(
                    X.values, (X.shape[0], X.shape[1], 1), order='C')
                _X = train_kernel_matrix(np.ascontiguousarray(
                    X_c, dtype=np.float64), self._sigma, self._triangular, self.normalization_method)
                self.kernel_duration = time.time() - start

                K_fe = compute_rbf_kernel(self.X_fit)

                self.train_gram = self.alpha * _X + ((1 - self.alpha) * K_fe)

        y = np.ones(n_instances, dtype=np.float64)

        # check W
        if W is None:
            W = np.ones(n_instances, dtype=np.float64)
        elif W.ndim != 1:
            raise ValueError("W.ndim !=1")

        # start training
        start = time.time()
        self.support_, self.support_vectors_, self._n_support, \
            self.dual_coef_, self.intercept_, self._probA, \
            self._probB, self.r_square, self.rho = libsvdd.fit(
                self.train_gram, y,
                svm_type=5, sample_weight=W, kernel='precomputed', C=self._C,
                shrinking=self.shrinking, tol=self.tol,
                cache_size=self.cache_size, max_iter=self.max_iter)

        self.svdd_duration = time.time() - start
        self.n_SV = len(self.support_)
        self.is_fit = True

        return self

    def fit_predict(self, X):
        """
        Fit SVDD to training data, then predict.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Should have shape of (n_instances, time_series_length).

        Returns
        -------
        np.array
            Array of length n_instances where -1 corresponds to an outlier and +1 to an inlier.
        """
        self.fit(X)
        return self.predict(X)

    def predict(self, X, K_xx_s=None, dec_vals=False):
        """Predict from data or from kernel matrix. K_xx_s is the `diagonal` and must be provided when kernel=`precomputed`.
        When dec_vals=`True`, the `distance` to the decision boundary is returned instead of labels.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Should have shape of (n_instances, time_series_length).
        K_xx_s : np.ndarray, optional
            Corresponds to the diagonal of the Gram matrix, i.e, K(x_1, x_1), ..., by default None.
            Needed when kernel=`precomputed`.
        dec_vals : bool, optional
            True if distance values should be returned, by default False.

        Returns
        -------
        np.array
            Array of length n_instances where -1 corresponds to an outlier and +1 to an inlier.
            When dec_vals=`True`, the `distance` to the decision boundary is returned.

        Raises
        ------
        AttributeError
            When fit() is not called before predict().
        """

        # fit before precit
        if not self.is_fit:
            raise AttributeError('SVDD not fitted.')

        # distinguish between precomputed and built-in kernels
        if self.kernel == 'precomputed':
            X, K_xx_s = self._check_kernel_matrix(
                X, is_predict=True, K_xx=K_xx_s)
        else:
            # check if input data are valid for kernel computation
            X = self._check_input_array(X)
            if X.ndim == 3:
                n_instances, n_length, n_dim = X.shape
            elif X.ndim == 2:
                n_instances, n_length = X.shape

            if self.kernel == 'tga':
                # sv_indices are passed to gram matrix computation
                sv_indices = np.sort(self.support_).astype(
                    dtype=np.int64, order='C')
                # libsvm starts counting with 1
                sv_indices = sv_indices - 1
                # special case when fit and predict data are equal
                if np.array_equal(self.X_fit, X):
                    gram_matrix = self.train_gram
                else:
                    gram_matrix = test_kernel_matrix(
                        self.X_fit, X, self._sigma, self._triangular, self.normalization_method, sv_indices)
                X = gram_matrix
                if self.normalization_method == 'exp':
                    K_xx_s = np.ones(n_instances)
                else:
                    gram_diagonal_test = train_kernel_matrix(
                        self.X_fit, self._sigma, self._triangular, self.normalization_method)
                    # This is incorrect. gram_diagonal_test's diagonal does not contain K(x_i,x_i)
                    gram_diagonal_test = np.diagonal(gram_diagonal_test)
                    K_xx_s = gram_diagonal_test
            elif self.kernel == 'gds-dtw':
                # GDS_{DTW}(x_i, x_j) = \exp (- \frac{DTW(x_i, x_j)^ 2}{\sigma^2})
                # sv_indices are passed to gram matrix computation
                sv_indices = np.sort(self.support_).astype(
                    dtype=np.int64, order='C')
                # libsvm starts counting with 1
                sv_indices = sv_indices - 1

                X, K_xx_s = test_gds_dtw(self.X_fit, X, self._sigma)
            elif self.kernel == 'rbf':
                X_ = rbf_kernel_fast_test(X, self.X_fit, self._sigma)
                self.train_gram = X_
                K_xx_s = np.ones(X.shape[0], dtype=np.float64, order='C')
            elif self.kernel == "rbf-gak":
                # sv_indices are passed to gram matrix computation
                sv_indices = np.sort(self.support_).astype(
                    dtype=np.int64, order='C')
                # libsvm starts counting with 1
                sv_indices = sv_indices - 1
                # special case when fit and predict data are equal
                if np.array_equal(self.X_fit.values, X.values):
                    gram_matrix = self.train_gram
                    K_xx_s = np.ones(
                        gram_matrix.shape[0], dtype=np.float64, order='C')
                else:
                    X_test = np.reshape(
                        X.values, (X.shape[0], X.shape[1], 1), order='C')
                    X_train = np.reshape(
                        self.X_fit.values, (self.X_fit.shape[0], self.X_fit.shape[1], 1), order='C')
                    gram_matrix = test_kernel_matrix(np.ascontiguousarray(X_train, dtype=np.float64), np.ascontiguousarray(
                        X_test, dtype=np.float64), self._sigma, self._triangular, self.normalization_method, sv_indices)
                    K_fe = compute_rbf_kernel(self.X_fit, X)
                    gram_matrix = self.alpha * gram_matrix + \
                        ((1 - self.alpha) * K_fe)

                    K_xx_s = np.ones(
                        X_test.shape[0], dtype=np.float64, order='C')
                X = gram_matrix

        if dec_vals:
            score = libsvdd.decision_function(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                cache_size=self.cache_size)
        else:
            score = libsvdd.predict(
                X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
                self.dual_coef_, self.intercept_,
                self._probA, self._probB, svm_type=5, kernel='precomputed',
                cache_size=self.cache_size)

        return score

    def set_params(self, **parameters):
        """Sets SVDD's parameters. Used for e.g., scikit-learn's hyperparameter optimizes.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Returns the models attributes as dict.

        Parameters
        ----------
        deep : bool, optional
            For legacy purposes, by default True.

        Returns
        -------
        dict
        """
        return {"kernel": self.kernel, "nu": self.nu, "C": self.C, "sigma": self.sigma, "alpha": self.alpha,
                "triangular": self.triangular, "normalization_method": self.normalization_method, "tol": self.tol,
                "shrinking": self.shrinking, "cache_size": self.cache_size, "max_iter": self.max_iter, "verbose": self.verbose}

    def decision_function(self, X, K_xx_s=None):
        """Calls predict() with dec_values=`True`. See predict() doc string.
        """

        p_val = self.predict(X, K_xx_s, dec_vals=True)
        return p_val

    def _info(self, line):
        """Print function.

        Parameters
        ----------
        line : str
            Message to output.
        """
        if self.verbose:
            print(line)

    def _check_arguments(self):
        """
        Check if C is valid. If \nu is set, calculate C from \nu.
        If sigma=`auto` or triangular=`auto`, calculate accordingly.
        """
        n_instances = self.fit_shape[0]
        n_length = self.fit_shape[1]
        if self.C is not None:
            if self.C < (1 / n_instances):
                self._C = (1 / n_instances)
                warnings.warn(f'C too small, set C to {self._C}', Warning)
            else:
                self._C = self.C
        elif self.nu:
            if self.nu <= 0 or self.nu > 1:
                raise ValueError(f'Invalid parameter `nu={self.nu}`.')
            else:
                self._C = 1 / (self.nu * n_instances)
        if self.kernel == 'tga' or self.kernel == 'rbf-gak':
            if self.sigma == 'auto':
                sigmas = sampled_gak_sigma(self.X_fit, 100)
                self._sigma = sigmas[3]
            else:
                self._sigma = self.sigma
            if self.triangular == 'auto':
                self._triangular = .5 * n_length
            else:
                self._triangular = self.triangular
        elif self.kernel == 'gds-dtw':
            self._sigma = self.sigma

    def _check_kernel_matrix(self, predict_matrix, is_fit=True, is_predict=False, K_xx=None):
        """
        Check if user provided gram matrix has correct shape and is c-contiguous.
        """
        # check shapes for predict; gram matrix --> predict gram matrix; self.fit_shape --> train gram matrix
        if is_predict:
            if K_xx is None:
                raise ValueError('K_xx can not be None')
            if predict_matrix.shape[0] != K_xx.shape[0]:
                raise ValueError(
                    f"Diagonal of predict matrix `gram matrix shape=({predict_matrix.shape})` has wrong length `diagonal length={K_xx.shape[0]}`.")
            if self.fit_shape[0] != predict_matrix.shape[1]:
                raise ValueError(
                    "Prediction matrix does not fit train matrix.")
            is_fit = False
        # check shapes during fit
        if is_fit:
            if predict_matrix.shape[0] != predict_matrix.shape[0]:
                raise ValueError("Kernel matrix should be symmetric.")
        # check if C_CONTIGUOUS
        if not predict_matrix.flags['C_CONTIGUOUS']:
            predict_matrix = np.ascontiguousarray(
                predict_matrix, dtype=np.float64)
        if K_xx is not None and not K_xx.flags['C_CONTIGUOUS']:
            K_xx = np.ascontiguousarray(K_xx, dtype=np.float64)
        if K_xx is not None and K_xx.ndim != 1:
            raise ValueError('K_xx_s ndim unequal 1')
        return predict_matrix, K_xx

    def _check_input_array(self, X):
        """
        Check if X has correct shape for kernel and is c-contiguous.
        """
        if isinstance(X, pd.DataFrame):
            if self.kernel == 'tga':
                X = np.reshape(
                    X.values, (X.shape[0], X.shape[1], 1), order='C')
            elif self.kernel == 'gds-dtw':
                X = np.reshape(X.values, (X.shape[0], X.shape[1]), order='C')
            elif self.kernel == 'rbf-gak':
                return X
        elif isinstance(X, np.ndarray):
            if self.kernel == 'tga':
                if X.ndim == 2:
                    X = np.reshape(
                        X, (X.shape[0], X.shape[1], 1), order='C')
            elif self.kernel in ["gds-dtw", "rbf"]:
                if X.ndim != 2:
                    raise ValueError(
                        "Input array X has wrong shape. Should be 2-tuple (n_instances, n_length)")
            elif self.kernel == 'rbf-gak':
                if X.ndim != 2:
                    raise ValueError(
                        "Input array X has wrong shape. Should be 2-tuple (n_instances, n_length)")
                else:
                    return pd.DataFrame(X)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X, dtype=np.float64)
        return X
