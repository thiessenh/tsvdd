import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
# from tsvdd.ga import tga_dissimilarity, train_kernel_matrix, test_kernel_matrix
#from tsvdd.utils import svmlib_kernel_format
#from sklearn.svm import SVDD
from tsvdd import libsvdd

class DirtySVDD:
    _kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    def __init__(self, kernel='linear', C=0.002, degree=3, gamma=1,
                 coef0=0.0, tol=1e-3, nu=0.5, shrinking=False, cache_size=200,
                 verbose=True):
        if kernel not in self._kernels:
            raise ValueError()
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
        self.shrinking=shrinking
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.cache_size = cache_size
        self.verbose = verbose
        self.class_weight_ = np.empty(0, dtype=np.float64)
        self.probability = False
        self.epsilon = 0
        self.is_fit = False
        self.n_SV = None

    def fit(self, X):
        y = np.ones(X.shape[0], dtype=np.float64)
        sample_weight = np.ones(X.shape[0], dtype=np.float64)

        self.support_, self.support_vectors_, self._n_support, \
        self.dual_coef_, self.intercept_, self._probA, \
        self._probB = libsvdd.fit(
                X, y,
                svm_type=5, sample_weight=sample_weight,
                class_weight=self.class_weight_, kernel=self.kernel, C=self.C,
                nu=self.nu, probability=self.probability, degree=self.degree,
                shrinking=self.shrinking, tol=self.tol,
                cache_size=self.cache_size, coef0=self.coef0,
                gamma=self.gamma, epsilon=self.epsilon)
        self.is_fit = True
        self.n_SV = self.support_vectors_.shape[0]

    def predict(self, X, K_xx_s=None):
        assert self.is_fit
        if K_xx_s is None and self.kernel == 'precomputed':
            K_xx_s = np.array([10]*X.shape[0], dtype=np.float64)
        else:
            K_xx_s = np.ascontiguousarray(K_xx_s, dtype=np.float64)
            if K_xx_s.ndim != 1:
                raise ValueError('K_xx_s ndim unequal 1')
        return libsvdd.predict(
            X, K_xx_s, self.support_, self.support_vectors_, self._n_support,
            self.dual_coef_, self.intercept_,
            self._probA, self._probB, svm_type=5, kernel=self.kernel,
            degree=self.degree, coef0=self.coef0, gamma=self.gamma,
            cache_size=self.cache_size)

n_samples = 1000
outlier_ratio = 0.5
X, y = make_circles(n_samples=n_samples, factor=.3, noise=0.05)
y[y == 0] = -1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
outlier_ratio_train = y_train[y_train == -1].shape[0] / (
            y_train[y_train == -1].shape[0] + y_train[y_train == 1].shape[0])
outlier_ratio_test = y_test[y_test == -1].shape[0] / (y_test[y_test == -1].shape[0] + y_test[y_test == 1].shape[0])
n_instances = X_train.shape[0]

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
plt.title('Train split')
reds = y_test == -1
blues = y_test == 1

axs[0].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
axs[0].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
axs[0].set_title('Test split')
axs[0].set_aspect(aspect='equal')
reds = y_train == -1
blues = y_train == 1

axs[1].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
axs[1].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
axs[1].set_title('Train split')
axs[1].set_aspect(aspect='equal')
plt.show()

gram_train = linear_kernel(X_train, X_train)
gram_test = linear_kernel(X_test, X_train)
gram_diagonal_test = np.diagonal(linear_kernel(X_test, X_test))
gram_diagonal_train = np.diagonal(gram_train)

# PRECOMPUTED = False
# if PRECOMPUTED:
#     X_libsvm_train = svmlib_kernel_format(gram_train)
#     X_libsvm_test = svmlib_kernel_format(gram_test)
#     prob = svmutil.svm_problem(np.ones(n_instances), X_libsvm_train, isKernel=True)
# else:
#     X_libsvm_train = X_train.tolist()
#     X_libsvm_test = X_test.tolist()
#     prob = svmutil.svm_problem(np.ones(n_instances), X_libsvm_train, isKernel=False)

C = 1 / (outlier_ratio_train * n_instances)
if C <= 1/n_samples or C >= 1:
    assert False
# if PRECOMPUTED:
#     param = svmutil.svm_parameter(f'-s 5 -t 4 -c {C} -e {10e-10} -h 0')
# else:
#     param = svmutil.svm_parameter(f'-s 5 -t 0 -c {C} -e {10e-10} -h 0')

#svdd = SVDD(kernel='linear', nu=outlier_ratio_train, verbose=True, tol=10e-10, shrinking=False)
#svdd.fit(X_train)

svdd = DirtySVDD(kernel='precomputed', C=C,verbose=True, tol=10e-10, shrinking=False)
svdd.fit(gram_train)
print("gram_train")
print(gram_train)
print("gram_diagonal_train")
print(gram_diagonal_train)
print("dual_coef_")
print(svdd.dual_coef_)
print("support_")
print(svdd.support_)
#model = svmutil.svm_train(prob, param)
# y_pred_test, p_val = svmutil.svm_predict(X_libsvm_test, gram_diagonal_test, model)
# y_pred_train, p_val = svmutil.svm_predict(X_libsvm_train, gram_diagonal_train, model)
y_pred_train = svdd.predict(gram_train, gram_diagonal_train)
y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
y_pred_test = np.array(y_pred_test)
y_pred_train = np.array(y_pred_train)

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
reds = y_pred_test == -1
blues = y_pred_test == 1

axs[0].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
axs[0].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
axs[0].set_title('Test split SVDD')
axs[0].set_aspect(aspect='equal')
reds = y_pred_train == -1
blues = y_pred_train == 1

axs[1].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
axs[1].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
axs[1].set_title('Train split SVDD')
axs[1].set_aspect(aspect='equal')
plt.show()

print(f"Inlier in y_pred_train: {y_pred_train[y_pred_train==1].shape[0]}")
print(f"Inlier in y_train: {y_train[y_train==1].shape[0]}")

1


# n_train = 30
# length_train = 10
# dim = 1
#
# rs = np.random.RandomState(1234)
# c_train_matrix = rs.rand(n_train, length_train, dim).astype(dtype=np.float64, order='c')
# res = np.zeros(shape=(n_train, n_train))
# sigma = 2
# triangular = 0
# for i in range(n_train):
#     seq_1 = c_train_matrix[i]
#     for j in range(n_train):
#         seq_2 = c_train_matrix[j]
#         res[i, j] = tga_dissimilarity(seq_1, seq_2, sigma, triangular)
# res = np.exp(-res)
# res_cython = train_kernel_matrix(c_train_matrix, sigma, triangular, 'exp')
# np.testing.assert_array_equal(res, res_cython)
#
# print(model.get_r2())
# print(model.get_rho())
