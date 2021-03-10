import numpy as np

# query strategies must provide signature: (U, L_in, L_out, y_vals):
# Then add clause to get_QS

query_strategies = ["uncertainty_outside", "random_outlier"]


def _uncertainty_outside(U, L_in, L_out, y_vals):
    """
    Returns samples closest to the decision boundary or lying on it.
    @param U: Set of indices referring to unknown samples
    @param L_in: Set of indices referring to annotated inliers
    @param L_out: Set of indices referring to annotated outliers
    @param y_vals: Values from decision function
    @return:
    """
    idx_in_U = np.abs(np.where(y_vals[U] <= 0, np.inf, y_vals[U])).argmin()
    idx = U[idx_in_U]
    return idx


def _random_outlier(U, L_in=None, L_out=None, y_vals=None):
    """
    Randomly selects a sample from U for annotation.
    @param U: Set of indices referring to unknown samples
    @param L_in: Set of indices referring to annotated inliers
    @param L_out: Set of indices referring to annotated outliers
    @param y_vals: Values from decision function
    @return:
    """
    rng = np.random.default_rng()
    return rng.choice(U)


def get_QS(query_strategy):
    """
    Returns query function.
    @param query_strategy: string referring to query strategy
    @return:
    """
    if query_strategy is "uncertainty_outside":
        return _uncertainty_outside
    if query_strategy is "random_outlier":
        return _random_outlier
    raise ValueError(f"Invalid query strategy: `{query_strategy}`.")


