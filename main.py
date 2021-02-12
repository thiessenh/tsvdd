from tsvdd.libsvm import svmutil
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
import numpy as np
from tsvdd.ga import tga_dissimilarity, train_kernel_matrix, test_kernel_matrix
from tsvdd.utils import svmlib_kernel_format


n_samples = 2000
outlier_ratio = 0.5
X, y = make_circles(n_samples=n_samples, factor=.3, noise=0.01)
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

gram_train = rbf_kernel(X_train, X_train)
gram_test = rbf_kernel(X_test, X_train)
gram_diagonal_test = np.diagonal(rbf_kernel(X_test, X_test))
gram_diagonal_train = np.diagonal(gram_train)

X_libsvm_train = svmlib_kernel_format(gram_train)
X_libsvm_test = svmlib_kernel_format(gram_test)
prob = svmutil.svm_problem(np.ones(n_instances), X_libsvm_train, isKernel=True)
param = svmutil.svm_parameter(f'-s 5 -t 4 -n {outlier_ratio_train}')
model = svmutil.svm_train(prob, param)
y_pred_test, p_val = svmutil.svm_predict(X_libsvm_test, gram_diagonal_test, model)
y_pred_train, p_val = svmutil.svm_predict(X_libsvm_train, gram_diagonal_train, model)
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
