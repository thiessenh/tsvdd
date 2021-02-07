import numpy as np


def svmlib_kernel_format(kernel_matrix):
    """
    Transforms kernel_matrix to matrix in libsvm format.

    :param kernel_matrix:
    :return:
    """
    n_instances_test = kernel_matrix.shape[0]
    n_instances_train = kernel_matrix.shape[1]
    x_tmp = np.zeros((n_instances_test, n_instances_train + 1), dtype=np.float64)
    x_tmp[:, 1:] = kernel_matrix
    x_tmp[:, :1] = np.arange(n_instances_test, dtype=np.int64)[:, np.newaxis] + 1
    x_libsvm = [list(row) for row in x_tmp]
    return x_libsvm


def sampled_gak_sigma(X, n_samples, random_state: np.random.RandomState = None, multiplier=None):
    """
    Returns multiples between 0.1 and 10 of sigma estimation. Procedure according to Cuturi.

    :param random_state:
    :param X: numpy array
    :param n_samples: Number samples to use for sigma estimation.
    :param multiplier: list with multipliers for sigmas
    :return:
    """
    if len(X.shape) == 3:
        n_instances, n_length, n_dim = X.shape
        if n_dim == 1:
            _X = np.reshape(X, (n_instances, n_length))
        else:
            raise NotImplementedError()
    else:
        n_instances, n_length = X.shape
        _X = X
    random_state = np.random.RandomState() if random_state is None else random_state
    replace = True if n_samples > n_instances else False

    medians = np.empty(n_samples)
    norms = np.empty(n_length)
    for i in range(n_samples):
        x = random_state.choice(n_instances, replace=replace)
        y = random_state.choice(n_instances, replace=replace)
        for l in range(n_length):
            norms[l] = np.linalg.norm(_X[x][l] - _X[y][l])

        medians[i] = np.median(norms)
    median_sampled = np.median(medians)
    if multiplier is None:
        return list(np.array([0.1, 1, 2, 5, 10]) * median_sampled * np.sqrt(n_length))
    elif isinstance(multiplier, list):
        return np.array(multiplier) * median_sampled * np.sqrt(n_length)
    return multiplier * median_sampled * np.sqrt(n_length)
