import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne


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
    random_state = np.random.default_rng(42) if random_state is None else random_state
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


def plot_train_test_pred(X_train, X_test, y_train, y_test, y_pred_train=None, y_pred_test=None, r_square=None):
    if y_pred_test is not None and y_pred_train is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 12), dpi=600)
    fig.suptitle('Active learning Data with cirlce', fontsize=12)
    reds = y_train == -1
    blues = y_train == 1

    axs[0,0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
    axs[0,0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0,0].set_title('Train split')
    axs[0,0].set_aspect(aspect='equal')

    reds = y_test == -1
    blues = y_test == 1

    axs[0,1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
    axs[0,1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0,1].set_title('Test split')
    axs[0,1].set_aspect(aspect='equal')

    if y_pred_test is not None and y_pred_train is not None:
        reds = y_pred_train == -1
        blues = y_pred_train == 1

        if r_square:
            circle = plt.Circle((0, 0), r_square, color='b', fill=False)
            axs[1,0].add_patch(circle)
        axs[1,0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
        axs[1,0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
        axs[1,0].set_title('Train split alSVDD')
        axs[1,0].set_aspect(aspect='equal')

        reds = y_pred_test == -1
        blues = y_pred_test == 1

        if r_square:
            circle = plt.Circle((0, 0), r_square, color='b', fill=False)
            axs[1,1].add_patch(circle)
        axs[1,1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
        axs[1,1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
        axs[1,1].set_title('Test split alSVDD')
        axs[1,1].set_aspect(aspect='equal')

    plt.show()


def decision_boundary_train_test(X_train, X_test, y_pred_train, y_pred_test, xx, yy, Z):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Active learning Data with random cirlce', fontsize=12)

    axs[0].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs[0].contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs[0].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    axs[1].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs[1].contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs[1].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    reds = y_pred_train == -1
    blues = y_pred_train == 1

    axs[0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
    axs[0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0].set_title('Train split')
    axs[0].set_aspect(aspect='equal')

    reds = y_pred_test == -1
    blues = y_pred_test == 1

    axs[1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
    axs[1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
    axs[1].set_title('Test split')
    axs[1].set_aspect(aspect='equal')

    plt.show()


def decision_boundary(X, y, xx, yy, Z,title=None, sets=None):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    if title:
        fig.suptitle(f'{title}', fontsize=12)
    else:
        fig.suptitle('Active learning Data decision boundary', fontsize=12)

    axs.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    if sets is None:
        reds = y == -1
        blues = y == 1

        axs.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
        axs.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')

    else:
        (U, L_in, L_out) = sets
        # Unknown points
        reds = y[U] == -1
        blues = y[U] == 1

        axs.scatter(X[U][reds, 0], X[U][reds, 1], c="red", s=20, edgecolor='k')
        axs.scatter(X[U][blues, 0], X[U][blues, 1], c="blue", s=20, edgecolor='k')

        # Inliers
        reds = y[L_in] == -1
        blues = y[L_in] == 1

        if len(blues) > 0:
            axs.scatter(X[L_in][reds, 0], X[L_in][reds, 1], c="gold", s=20, marker='*', edgecolor='k')
        if len(reds) > 0:
            axs.scatter(X[L_in][blues, 0], X[L_in][blues, 1], c="gold", s=20, marker='*', edgecolor='k')

        # Outliers
        reds = y[L_out] == -1
        blues = y[L_out] == 1

        if len(blues) > 0:
            axs.scatter(X[L_out][reds, 0], X[L_out][reds, 1], c="gold", s=20, marker='*', edgecolor='k')
        if len(reds) > 0:
            axs.scatter(X[L_out][blues, 0], X[L_out][blues, 1], c="gold", s=20, marker='*', edgecolor='k')


    axs.set_aspect(aspect='equal')
    plt.show()

def normalize_0_1(data):
    """
    Returns normalized data between 0 and 1.

    :param data: numpy or DataFrame
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def rbf_kernel_fast(X, sigma):
    """
    Computes the gram RBF Kernel matrix

    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.
    """
    X_norm = -np.einsum('ij,ij->i', X, X)
    gamma = 1 / (2.0 * sigma ** 2)
    return ne.evaluate('exp(g * (A + B + 2 * C))', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': np.dot(X, X.T),
        'g': gamma
    })

def rbf_kernel_fast_test(test, sigma, train):
    """
    Computes the RBF Kernel matrix for prediction.

    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.
    """
    X_norm = np.einsum('ij,ij->i', test, test)
    if train is not None:
        Y_norm = np.einsum('ij,ij->i', train, train)
    else:
        Y = test
        Y_norm = X_norm

    return ne.evaluate('exp(-g * (A + B - 2 * C))', {
        'A': X_norm[:, None],
        'B': Y_norm[None, :],
        'C': np.dot(test, train.T),
        'g': 1 / (2 * sigma ** 2)
    })


def compute_rbf_kernel(X, X_test=None):
    from tsfresh import extract_features 
    from tsfresh.utilities.dataframe_functions import impute
    rbf_gak_features ={'fft_coefficient': [{'attr': 'real', 'coeff': 72},
   {'attr': 'abs', 'coeff': 64},
   {'attr': 'angle', 'coeff': 32},
   {'attr': 'real', 'coeff': 70},
   {'attr': 'real', 'coeff': 61},
   {'attr': 'real', 'coeff': 45},
   {'attr': 'imag', 'coeff': 81},
   {'attr': 'imag', 'coeff': 38}],
  'fft_aggregated': [{'aggtype': 'variance'}],
  'symmetry_looking': [{'r': 0.15000000000000002}, {'r': 0.45}],
  'value_count': [{'value': -1}],
  'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.2},
   {'f_agg': 'var', 'isabs': True, 'qh': 0.8, 'ql': 0.6}],
  'max_langevin_fixed_point': [{'m': 3, 'r': 30}]}

    X["id"] = X.index
    X = X.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)
    X_features = extract_features(X, default_fc_parameters=rbf_gak_features, column_id="id", column_sort="time", impute_function=impute) 
    X_features = X_features.apply(normalize_0_1, axis=0).fillna(0)
    X_features = X_features.values

    if X_test is not None:
        X_test["id"] = X_test.index
        X_test = X_test.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)
        X_test_features = extract_features(X_test, default_fc_parameters=rbf_gak_features, column_id="id", column_sort="time", impute_function=impute) 
        X_test_features = X_test_features.apply(normalize_0_1, axis=0).fillna(0)
        X_test_features = X_test_features.values

    K_s = []
    if X_test is not None:
         for train, test in zip(X_features.T, X_test_features.T):
            f_sigma = X.shape[0] ** (-1 / (1 + 4)) #* np.std(train.reshape((-1,1)))
            K_s.append(rbf_kernel_fast_test(test.reshape((-1,1)), f_sigma, train.reshape((-1,1))))
    else:
        for train in X_features.T:
            f_sigma = X.shape[0] ** (-1 / (1 + 4)) #* np.std(train.reshape((-1,1)))
            K_s.append(rbf_kernel_fast(train.reshape((-1,1)), f_sigma))

    K_s = np.stack(K_s)
    
    K_f = np.ones(shape=(K_s[0].shape[0], K_s[0].shape[1]), dtype=np.float64)
    for i in range(K_s.shape[0]):
        K_f = K_f * K_s[i]

    return K_f