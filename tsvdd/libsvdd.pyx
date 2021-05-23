# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified to work with SVDD-libsvm.
#
# Haiko Thiessen 2021


import warnings
import  numpy as np
cimport numpy as np
from libc.stdlib cimport free


cdef extern from "svm.h":
    cdef struct svm_node
    cdef struct svm_model
    cdef struct svm_parameter:
        int svm_type
        int kernel_type
        int degree	# for poly
        double gamma	# for poly/rbf/sigmoid
        double coef0	# for poly/sigmoid

        # these are for training only
        double cache_size # in MB
        double eps	# stopping criteria
        double C	# for C_SVC, EPSILON_SVR and NU_SVR
        int nr_weight		# for C_SVC
        int *weight_label	# for C_SVC
        double* weight		# for C_SVC
        double nu	# for NU_SVC, ONE_CLASS, and NU_SVR
        double p	# for EPSILON_SVR
        int shrinking	# use the shrinking heuristics
        int probability # do probability estimates
        int max_iter  # ceiling on Solver runtime
        int random_seed  # seed for random generator in probability estimation

    cdef struct svm_problem:
        int l
        double *y
        svm_node *x
        double *W # instance weights

    char *svm_check_parameter(svm_problem *, svm_parameter *)
    svm_model *svm_train(svm_problem *, svm_parameter *) nogil
    void svm_free_and_destroy_model(svm_model** model_ptr_ptr)

cdef extern from "libsvm_helper.c":
    # this file contains methods for accessing libsvm 'hidden' fields
    # svm_node ** dense_to_sparse(char *, np.npy_intp *)
    void set_parameter(svm_parameter *, int, int, int, double, double,
                       double, double, double, double,
                       double, int, int, int, char *, char *, int)
    void set_problem(svm_problem *, char *, char *, char *, np.npy_intp *, int)

    svm_model *set_model(svm_parameter *, int, char *, np.npy_intp *,
                         char *, np.npy_intp *, np.npy_intp *, char *,
                         char *, char *, char *, char *)

    void copy_sv_coef(char *, svm_model *)
    void copy_intercept(char *, svm_model *, np.npy_intp *)
    void copy_SV(char *, svm_model *, np.npy_intp *)
    int copy_support(char *data, svm_model *model)
    int copy_predict(char *, svm_model *, np.npy_intp *, char *, char *) nogil
    #int copy_predict_proba(char *, svm_model *, np.npy_intp *, char *) nogil
    int copy_predict_values(char *, svm_model *, np.npy_intp *, char *, int, char *) nogil
    void copy_nSV(char *, svm_model *)
    void copy_probA(char *, svm_model *, np.npy_intp *)
    void copy_probB(char *, svm_model *, np.npy_intp *)
    np.npy_intp  get_l(svm_model *)
    np.npy_intp  get_nr(svm_model *)
    #int  free_problem(svm_problem *)
    int  free_model(svm_model *)
    double get_r_square(svm_model *)
    double get_rho(svm_model *)
    #void set_verbosity(int)


np.import_array()


################################################################################
# Internal variables
LIBSVM_KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']


################################################################################
# Wrapper functions

def fit(
    np.ndarray[np.float64_t, ndim=2, mode='c'] X,
    np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
    int svm_type=0, kernel='rbf',
    double tol=1e-3, double C=1.,
    np.ndarray[np.float64_t, ndim=1, mode='c']
        sample_weight=np.empty(0),
    int shrinking=1, double cache_size=100., int max_iter=1000000):
    """
    Train the model using libsvm (low-level method)

    Parameters
    ----------
    X : array-like, dtype=float64 of shape (n_samples, n_features)

    Y : array, dtype=float64 of shape (n_samples,)
        target vector

    svm_type : {0, 1, 2, 3, 4, 5}, optional
        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR, NuSVR, or
        SVDD-L1 respectively. 0 by default.

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed.

    degree : int32, default=3
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial).

    gamma : float64, default=0.1
        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    coef0 : float64, default=0
        Independent parameter in poly/sigmoid kernel.

    tol : float64, default=1e-3
        Numeric stopping criterion (WRITEME).

    C : float64, default=1
        C parameter in C-Support Vector Classification.

    nu : float64, default=0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Should be in the interval (0, 1].

    epsilon : double, default=0.1
        Epsilon parameter in the epsilon-insensitive loss function.

    class_weight : array, dtype=float64, shape (n_classes,), \
            default=np.empty(0)
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.

    sample_weight : array, dtype=float64, shape (n_samples,), \
            default=np.empty(0)
        Weights assigned to each sample.

    shrinking : int, default=1
        Whether to use the shrinking heuristic.

    probability : int, default=0
        Whether to enable probability estimates.

    cache_size : float64, default=100
        Cache size for gram matrix columns (in megabytes).


    Returns
    -------
    support : array of shape (n_support,)
        Index of support vectors.

    support_vectors : array of shape (n_support, n_features)
        Support vectors (equivalent to X[support]). Will return an
        empty array in the case of precomputed kernel.

    n_class_SV : array of shape (n_class,)
        Number of support vectors in each class.

    sv_coef : array of shape (n_class-1, n_support)
        Coefficients of support vectors in decision function.

    intercept : array of shape (n_class*(n_class-1)/2,)
        Intercept in decision function.

    probA, probB : array of shape (n_class*(n_class-1)/2,)
        Probability estimates, empty array for probability=False.
    """

    cdef svm_parameter param
    cdef svm_problem problem
    cdef svm_model *model
    cdef const char *error_msg
    cdef np.npy_intp SV_len
    cdef np.npy_intp nr

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] class_weight = np.empty(0)
    cdef int degree = 3
    cdef double gamma=0.1
    cdef double coef0=0.0
    cdef double epsilon=0.1
    cdef int probability=0
    cdef double nu = 0.5

    if len(sample_weight) == 0:
        sample_weight = np.ones(X.shape[0], dtype=np.float64)
    else:
        assert sample_weight.shape[0] == X.shape[0], \
               "sample_weight and X have incompatible shapes: " + \
               "sample_weight has %s samples while X has %s" % \
               (sample_weight.shape[0], X.shape[0])

    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)
    set_problem(
        &problem, X.data, Y.data, sample_weight.data, X.shape, kernel_index)
    if problem.x == NULL:
        raise MemoryError("Seems we've run out of memory")
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    set_parameter(
        &param, svm_type, kernel_index, degree, gamma, coef0, nu, cache_size,
        C, tol, epsilon, shrinking, probability, <int> class_weight.shape[0],
        class_weight_label.data, class_weight.data, max_iter)

    error_msg = svm_check_parameter(&problem, &param)
    if error_msg:
        # for SVR: epsilon is called p in libsvm
        error_repl = error_msg.decode('utf-8').replace("p < 0", "epsilon < 0")
        raise ValueError(error_repl)

    # this does the real work
    with nogil:
        model = svm_train(&problem, &param)

    # from here until the end, we just copy the data returned by
    # svm_train
    SV_len  = get_l(model)
    n_class = get_nr(model)
    r_square = get_r_square(model)
    rho = get_rho(model)

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] sv_coef
    sv_coef = np.empty((n_class-1, SV_len), dtype=np.float64)
    copy_sv_coef (sv_coef.data, model)

    # the intercept is just model.rho but with sign changed
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] intercept
    intercept = np.empty(int((n_class*(n_class-1))/2), dtype=np.float64)
    copy_intercept (intercept.data, model, intercept.shape)

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] support
    support = np.empty (SV_len, dtype=np.int32)
    copy_support (support.data, model)

    # copy model.SV
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] support_vectors
    if kernel_index == 4:
        # precomputed kernel
        support_vectors = np.empty((0, 0), dtype=np.float64)
    else:
        support_vectors = np.empty((SV_len, X.shape[1]), dtype=np.float64)
        copy_SV(support_vectors.data, model, support_vectors.shape)

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] n_class_SV
    if svm_type == 0 or svm_type == 1:
        n_class_SV = np.empty(n_class, dtype=np.int32)
        copy_nSV(n_class_SV.data, model)
    else:
        # OneClass and SVR are considered to have 2 classes
        n_class_SV = np.array([SV_len, SV_len], dtype=np.int32)

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] probA
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] probB
    if probability != 0:
        if svm_type < 2: # SVC and NuSVC
            probA = np.empty(int(n_class*(n_class-1)/2), dtype=np.float64)
            probB = np.empty(int(n_class*(n_class-1)/2), dtype=np.float64)
            copy_probB(probB.data, model, probB.shape)
        else:
            probA = np.empty(1, dtype=np.float64)
            probB = np.empty(0, dtype=np.float64)
        copy_probA(probA.data, model, probA.shape)
    else:
        probA = np.empty(0, dtype=np.float64)
        probB = np.empty(0, dtype=np.float64)

    svm_free_and_destroy_model(&model)
    free(problem.x)

    return (support, support_vectors, n_class_SV, sv_coef, intercept,
           probA, probB, r_square, rho)


cdef void set_predict_params(
    svm_parameter *param, int svm_type, kernel, int degree, double gamma,
    double coef0, double cache_size, int probability, int nr_weight,
    char *weight_label, char *weight) except *:
    """Fill param with prediction time-only parameters."""

    # training-time only parameters
    cdef double C = .0
    cdef double epsilon = .1
    cdef double nu = .5
    cdef int shrinking = 0
    cdef double tol = .1
    cdef int max_iter = 1000000

    kernel_index = LIBSVM_KERNEL_TYPES.index(kernel)

    set_parameter(param, svm_type, kernel_index, degree, gamma, coef0, nu,
                         cache_size, C, tol, epsilon, shrinking, probability,
                         nr_weight, weight_label, weight, max_iter)

def predict(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
            np.ndarray[np.float64_t, ndim=1, mode='c'] K_xx_s,
            np.ndarray[np.int32_t, ndim=1, mode='c'] support,
            np.ndarray[np.float64_t, ndim=2, mode='c'] SV,
            np.ndarray[np.int32_t, ndim=1, mode='c'] nSV,
            np.ndarray[np.float64_t, ndim=2, mode='c'] sv_coef,
            np.ndarray[np.float64_t, ndim=1, mode='c'] intercept,
            np.ndarray[np.float64_t, ndim=1, mode='c'] probA=np.empty(0),
            np.ndarray[np.float64_t, ndim=1, mode='c'] probB=np.empty(0),
            int svm_type=0, kernel='rbf',
            np.ndarray[np.float64_t, ndim=1, mode='c']
                class_weight=np.empty(0),
            np.ndarray[np.float64_t, ndim=1, mode='c']
                sample_weight=np.empty(0),
            double cache_size=100.):
    """
    Predict target values of X given a model (low-level method)

    Parameters
    ----------
    X : array-like, dtype=float of shape (n_samples, n_features)

    support : array of shape (n_support,)
        Index of support vectors in training set.

    SV : array of shape (n_support, n_features)
        Support vectors.

    nSV : array of shape (n_class,)
        Number of support vectors in each class.

    sv_coef : array of shape (n_class-1, n_support)
        Coefficients of support vectors in decision function.

    intercept : array of shape (n_class*(n_class-1)/2)
        Intercept in decision function.

    probA, probB : array of shape (n_class*(n_class-1)/2,)
        Probability estimates.

    svm_type : {0, 1, 2, 3, 4}, default=0
        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR
        respectively.

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed.

    degree : int32, default=3
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial).

    gamma : float64, default=0.1
        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    coef0 : float64, default=0.0
        Independent parameter in poly/sigmoid kernel.

    Returns
    -------
    dec_values : array
        Predicted values.
    """
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dec_values
    cdef svm_parameter param
    cdef svm_model *model
    cdef int rv
    cdef int degree = 3
    cdef double gamma = 0.1
    cdef double coef0 = 0.0

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    set_predict_params(&param, svm_type, kernel, degree, gamma, coef0,
                       cache_size, 0, <int>class_weight.shape[0],
                       class_weight_label.data, class_weight.data)

    model = set_model(&param, <int> nSV.shape[0], SV.data, SV.shape,
                      support.data, support.shape, sv_coef.strides,
                      sv_coef.data, intercept.data, nSV.data, probA.data, probB.data)
    #TODO: use check_model
    try:
        dec_values = np.empty(X.shape[0])
        with nogil:
            rv = copy_predict(X.data, model, X.shape, dec_values.data, K_xx_s.data)
        if rv < 0:
            raise MemoryError("We've run out of memory")
    finally:
        free_model(model)

    return dec_values


def decision_function(
    np.ndarray[np.float64_t, ndim=2, mode='c'] X,
    np.ndarray[np.float64_t, ndim=1, mode='c'] K_xx_s,
    np.ndarray[np.int32_t, ndim=1, mode='c'] support,
    np.ndarray[np.float64_t, ndim=2, mode='c'] SV,
    np.ndarray[np.int32_t, ndim=1, mode='c'] nSV,
    np.ndarray[np.float64_t, ndim=2, mode='c'] sv_coef,
    np.ndarray[np.float64_t, ndim=1, mode='c'] intercept,
    np.ndarray[np.float64_t, ndim=1, mode='c'] probA=np.empty(0),
    np.ndarray[np.float64_t, ndim=1, mode='c'] probB=np.empty(0),
    int svm_type=0, kernel='rbf', int degree=3,
    double gamma=0.1, double coef0=0.,
    np.ndarray[np.float64_t, ndim=1, mode='c']
        class_weight=np.empty(0),
    np.ndarray[np.float64_t, ndim=1, mode='c']
         sample_weight=np.empty(0),
    double cache_size=100.):
    """
    Predict margin (libsvm name for this is predict_values)

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.

    Parameters
    ----------
    X : array-like, dtype=float, size=[n_samples, n_features]

    support : array, shape=[n_support]
        Index of support vectors in training set.

    SV : array, shape=[n_support, n_features]
        Support vectors.

    nSV : array, shape=[n_class]
        Number of support vectors in each class.

    sv_coef : array, shape=[n_class-1, n_support]
        Coefficients of support vectors in decision function.

    intercept : array, shape=[n_class*(n_class-1)/2]
        Intercept in decision function.

    probA, probB : array, shape=[n_class*(n_class-1)/2]
        Probability estimates.

    svm_type : {0, 1, 2, 3, 4}, optional
        Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR
        respectively. 0 by default.

    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, optional
        Kernel to use in the model: linear, polynomial, RBF, sigmoid
        or precomputed. 'rbf' by default.

    degree : int32, optional
        Degree of the polynomial kernel (only relevant if kernel is
        set to polynomial), 3 by default.

    gamma : float64, optional
        Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
        kernels. 0.1 by default.

    coef0 : float64, optional
        Independent parameter in poly/sigmoid kernel. 0 by default.

    Returns
    -------
    dec_values : array
        Predicted values.
    """
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] dec_values
    cdef svm_parameter param
    cdef svm_model *model
    cdef np.npy_intp n_class

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    cdef int rv

    set_predict_params(&param, svm_type, kernel, degree, gamma, coef0,
                       cache_size, 0, <int>class_weight.shape[0],
                       class_weight_label.data, class_weight.data)

    model = set_model(&param, <int> nSV.shape[0], SV.data, SV.shape,
                      support.data, support.shape, sv_coef.strides,
                      sv_coef.data, intercept.data, nSV.data,
                      probA.data, probB.data)

    if svm_type > 1:
        n_class = 1
    else:
        n_class = get_nr(model)
        n_class = n_class * (n_class - 1) // 2

    try:
        dec_values = np.empty((X.shape[0], n_class), dtype=np.float64)
        with nogil:
            rv = copy_predict_values(X.data, model, X.shape, dec_values.data, n_class, K_xx_s.data)
        if rv < 0:
            raise MemoryError("We've run out of memory")
    finally:
        free_model(model)

    return dec_values