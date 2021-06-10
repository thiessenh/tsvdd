import numpy as np

# query strategies must provide signature: (U, L_in, L_out, y_vals):
# Then add clause to get_QS()

query_strategies = ["uncertainty_outside", "random_outlier"]


def _uncertainty_outside(U, L_in, L_out, y_vals):
    """Returns outlying sample closest to the decision boundary or lying on it.

    Parameters
    ----------
    U : list
        Set of indices referring to unknown samples.
    L_in : list
        Set of indices referring to annotated inliers.
    L_out : list
        Set of indices referring to annotated outliers.
    y_vals : list
        Values from decision function, i.e, distance from observations to decision boundary.

    Returns
    -------
    int
        Index corresponding to closest outlying observation.
    """
    idx_in_U = np.abs(np.where(y_vals[U] <= 0, np.inf, y_vals[U])).argmin()
    idx = U[idx_in_U]
    return idx

def _random_outlier(U, L_in=None, L_out=None, y_vals=None):
    """Selects a random observation that lies outside the decision boundary.

    Parameters
    ----------
    U : list
        Set of indices referring to unknown samples.
    L_in : list
        Set of indices referring to annotated inliers.
    L_out : list
        Set of indices referring to annotated outliers.
    y_vals : list
        Values from decision function, i.e, distance from observations to decision boundary.

    Returns
    -------
    int
        Index corresponding to closest outlying observation.
    """
    rng = np.random.default_rng()
    return rng.choice(U)


def get_QS(query_strategy):
    """Returns query function.

    Parameters
    ----------
    query_strategy : str
        string referring to query strategy

    Returns
    -------
    func
        Query Strategy

    """
    if query_strategy is "uncertainty_outside":
        return _uncertainty_outside
    if query_strategy is "random_outlier":
        return _random_outlier
    raise ValueError(f"Invalid query strategy: `{query_strategy}`.")