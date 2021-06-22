import numpy as np
import matplotlib.pyplot as plt
import numexpr as ne
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from scipy.stats import zscore

rbf_gak_features = {'agg_linear_trend': [{'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'},
                                         {'attr': 'rvalue', 'chunk_len': 10,
                                             'f_agg': 'var'},
                                         {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'}],
                    'change_quantiles': [{'f_agg': 'mean', 'isabs': False, 'qh': 0.4, 'ql': 0.2},
                                         {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.4}]}


def sampled_gak_sigma(X, n_samples, random_state=None, multipliers=None):
    """Estimate optimal sigma according to Cuturi's Rule. When `multipliers` is specified, sigma multiples are returned.
    Otherwise, multiples in [0.1, 1, 2, 5, 10] are returned.

    Parameters
    ----------
    X : np.ndarray
        Time series array with, where one row corresponds to one time series.
    n_samples : int
        Amount of time series to consider for estimation.
    random_state : np.random.RandomState, optional
        If note provided default_rng(42) is used, by default None
    multiplier : float, list, optional
       Can be a list or a number, by default None

    Returns
    -------
    float or list of floats
        Estimated sigma multiple

    Raises
    ------
    NotImplementedError
        Raised when multivariate time series is detected.
    """
    if type(X).__name__ == 'DataFrame':
        X = X.values
    if len(X.shape) == 3:
        n_instances, n_length, n_dim = X.shape
        if n_dim == 1:
            _X = np.reshape(X, (n_instances, n_length))
        else:
            raise NotImplementedError()
    else:
        n_instances, n_length = X.shape
        _X = X
    random_state = np.random.default_rng(
        42) if random_state is None else random_state
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
    if multipliers is None:
        return list(np.array([0.1, 1, 2, 5, 10]) * median_sampled * np.sqrt(n_length))
    elif isinstance(multipliers, list):
        return np.array(multipliers) * median_sampled * np.sqrt(n_length)
    return multipliers * median_sampled * np.sqrt(n_length)


def plot_train_test_pred(X_train, X_test, y_train, y_test, y_pred_train=None, y_pred_test=None, r_square=None):
    if y_pred_test is not None and y_pred_train is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 12), dpi=600)
    fig.suptitle('Active learning Data with cirlce', fontsize=12)
    reds = y_train == -1
    blues = y_train == 1

    axs[0, 0].scatter(X_train[reds, 0], X_train[reds, 1],
                      c="red", s=20, edgecolor='k')
    axs[0, 0].scatter(X_train[blues, 0], X_train[blues, 1],
                      c="blue", s=20, edgecolor='k')
    axs[0, 0].set_title('Train split')
    axs[0, 0].set_aspect(aspect='equal')

    reds = y_test == -1
    blues = y_test == 1

    axs[0, 1].scatter(X_test[reds, 0], X_test[reds, 1],
                      c="red", s=20, edgecolor='k')
    axs[0, 1].scatter(X_test[blues, 0], X_test[blues, 1],
                      c="blue", s=20, edgecolor='k')
    axs[0, 1].set_title('Test split')
    axs[0, 1].set_aspect(aspect='equal')

    if y_pred_test is not None and y_pred_train is not None:
        reds = y_pred_train == -1
        blues = y_pred_train == 1

        if r_square:
            circle = plt.Circle((0, 0), r_square, color='b', fill=False)
            axs[1, 0].add_patch(circle)
        axs[1, 0].scatter(X_train[reds, 0], X_train[reds, 1],
                          c="red", s=20, edgecolor='k')
        axs[1, 0].scatter(X_train[blues, 0], X_train[blues, 1],
                          c="blue", s=20, edgecolor='k')
        axs[1, 0].set_title('Train split alSVDD')
        axs[1, 0].set_aspect(aspect='equal')

        reds = y_pred_test == -1
        blues = y_pred_test == 1

        if r_square:
            circle = plt.Circle((0, 0), r_square, color='b', fill=False)
            axs[1, 1].add_patch(circle)
        axs[1, 1].scatter(X_test[reds, 0], X_test[reds, 1],
                          c="red", s=20, edgecolor='k')
        axs[1, 1].scatter(X_test[blues, 0], X_test[blues, 1],
                          c="blue", s=20, edgecolor='k')
        axs[1, 1].set_title('Test split alSVDD')
        axs[1, 1].set_aspect(aspect='equal')

    plt.show()


def decision_boundary_train_test(X_train, X_test, y_pred_train, y_pred_test, xx, yy, Z):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Active learning Data with random circle', fontsize=12)

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

    axs[0].scatter(X_train[reds, 0], X_train[reds, 1],
                   c="red", s=20, edgecolor='k')
    axs[0].scatter(X_train[blues, 0], X_train[blues, 1],
                   c="blue", s=20, edgecolor='k')
    axs[0].set_title('Train split')
    axs[0].set_aspect(aspect='equal')

    reds = y_pred_test == -1
    blues = y_pred_test == 1

    axs[1].scatter(X_test[reds, 0], X_test[reds, 1],
                   c="red", s=20, edgecolor='k')
    axs[1].scatter(X_test[blues, 0], X_test[blues, 1],
                   c="blue", s=20, edgecolor='k')
    axs[1].set_title('Test split')
    axs[1].set_aspect(aspect='equal')

    plt.show()


def decision_boundary(X, y, xx, yy, Z, title=None, sets=None):
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
        axs.scatter(X[U][blues, 0], X[U][blues, 1],
                    c="blue", s=20, edgecolor='k')

        # Inliers
        reds = y[L_in] == -1
        blues = y[L_in] == 1

        if len(blues) > 0:
            axs.scatter(X[L_in][reds, 0], X[L_in][reds, 1],
                        c="gold", s=20, marker='*', edgecolor='k')
        if len(reds) > 0:
            axs.scatter(X[L_in][blues, 0], X[L_in][blues, 1],
                        c="gold", s=20, marker='*', edgecolor='k')

        # Outliers
        reds = y[L_out] == -1
        blues = y[L_out] == 1

        if len(blues) > 0:
            axs.scatter(X[L_out][reds, 0], X[L_out][reds, 1],
                        c="gold", s=20, marker='*', edgecolor='k')
        if len(reds) > 0:
            axs.scatter(X[L_out][blues, 0], X[L_out][blues, 1],
                        c="gold", s=20, marker='*', edgecolor='k')

    axs.set_aspect(aspect='equal')
    plt.show()


def normalize_0_1(data):
    """Use with apply(, axis=0) to normalize a dimension to values in [0,1].

    Parameters
    ----------
    data : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def rbf_kernel_fast(X, sigma):
    """Fast implementation to compute the RBF kernel. To set the numer of threads, use env variable OPENBLAS_NUM_THREADS.
    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.

    Parameters
    ----------
    X : np.ndarray
        Array with the shape (n_instances, n_dimensions)
    sigma : float
        RBF's bandwidth value.

    Returns
    -------
    np.ndarray
        Gram matrix.
    """
    X_norm = -np.einsum('ij,ij->i', X, X)
    return ne.evaluate('exp(g * (A + B + 2 * C))', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': np.dot(X, X.T),
        'g': 1 / (2.0 * sigma ** 2)
    })


def rbf_kernel_fast_test(X_test, sigma, X_train):
    """Fast implementation to compute the RBF kernel for prediction. To set the number of threads, use env variable OPENBLAS_NUM_THREADS.
    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.

    Parameters
    ----------
    X_test : np.ndarray
        Array with the shape (n_instances, n_dimensions)
    sigma : float
        RBF's bandwith value.
    X_train : np.ndarray
        Array with the shape (n_instances, n_dimensions)

    Returns
    -------
    np.ndarray
        Gram matrix.
    """
    Y_norm = np.einsum('ij,ij->i', X_test, X_test)
    if X_train is not None:
        X_norm = np.einsum('ij,ij->i', X_train, X_train)
    else:
        Y = X_test
        X_norm = Y_norm

    return ne.evaluate('exp(-g * (A + B - 2 * C))', {
        'A': Y_norm[:, None],
        'B': X_norm[None, :],
        'C': np.dot(X_test, X_train.T),
        'g': 1 / (2 * sigma ** 2)
    })


def compute_rbf_kernel(X, X_test=None):
    """Extracts the RBF-GAK features and computes the RBF kernel matrix/matrices.

    Parameters
    ----------
    X : DataFrame
        Needs to be a DataFrame so X can be transformed into a ts fresh time series container.
    X_test : DataFrame, optional
        Define for prediction matrix, by default None

    Returns
    -------
    np.ndarray
        Kernel matrices with shape (n_instances, n_instances) or (n_instances_test, n_instances_train).
    """

    n_instances = X.shape[0]

    X["id"] = X.index
    X = X.melt(id_vars="id", var_name="time")
    X["time"] = X["time"].astype(int)
    X = X.sort_values(["id", "time"]).reset_index(drop=True)

    X_features = extract_features(X, default_fc_parameters=rbf_gak_features,
                                  column_id="id", column_sort="time", impute_function=impute, disable_progressbar=True)
    X_features = X_features.apply(zscore, axis=0).fillna(0)
    X_features = X_features.values

    if X_test is not None:
        X_test["id"] = X_test.index
        X_test = X_test.melt(id_vars="id", var_name="time")
        X_test["time"] = X_test["time"].astype(int)
        X_test = X_test.sort_values(["id", "time"]).reset_index(drop=True)
        X_test_features = extract_features(
            X_test, default_fc_parameters=rbf_gak_features, column_id="id", column_sort="time", impute_function=impute, disable_progressbar=True)
        X_test_features = X_test_features.apply(zscore, axis=0).fillna(0)
        X_test_features = X_test_features.values

    K_s = []
    if X_test is not None:
        for train, test in zip(X_features.T, X_test_features.T):
            f_sigma = n_instances ** (-1 / (1 + 4))
            K_s.append(rbf_kernel_fast_test(test.reshape(
                (-1, 1)), f_sigma, train.reshape((-1, 1))))
    else:
        for train in X_features.T:
            f_sigma = n_instances ** (-1 / (1 + 4))
            K_s.append(rbf_kernel_fast(train.reshape((-1, 1)), f_sigma))

    K_s = np.stack(K_s)

    K_f = np.ones(shape=(K_s[0].shape[0], K_s[0].shape[1]), dtype=np.float64)
    for i in range(K_s.shape[0]):
        K_f = K_f * K_s[i]

    return K_f
