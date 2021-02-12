from tsvdd.libsvm import svmutil
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from tsvdd.utils import svmlib_kernel_format
import pytest


class TestLibsvm:
    @pytest.mark.parametrize("n_samples", [200, 500])
    def test_svdd(self, n_samples):
        X, y = make_circles(n_samples=n_samples, factor=.3, noise=0.01)
        y[y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        outlier_ratio_train = y_train[y_train == -1].shape[0] / (
                    y_train[y_train == -1].shape[0] + y_train[y_train == 1].shape[0])
        n_instances = X_train.shape[0]

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

        count_train =0
        count_test = 0
        for i in range(y_pred_train.shape[0]):
            if y_pred_train[i] != y_train[i]:
                count_train += 1

        for i in range(y_pred_test.shape[0]):
            if y_pred_test[i] != y_test[i]:
                count_test += 1

        assert not (count_test > 1 or count_train > 1)






