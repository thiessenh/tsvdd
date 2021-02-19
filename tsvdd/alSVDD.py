from tsvdd.SVDD import SVDD
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import balanced_accuracy_score


class alSVDD(SVDD):

    _query_strategies = ["confidence_outside", "random_outlier"]

    def __init__(self,  kernel='tga', nu=0.05, C=0.02, degree=3, gamma=1,
                 coef0=0.0, tol=1e-5, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200,
                 verbose=True,
                 query_strategy='confidence_outside', update_in=10, update_out=0.01):
        super().__init__(kernel=kernel, nu=nu, C=C, degree=degree, gamma=gamma,
                 coef0=coef0, tol=tol, sigma=sigma, triangular=triangular,
                 normalization_method=normalization_method, shrinking=shrinking, cache_size=cache_size,
                 verbose=verbose)
        if query_strategy not in self._query_strategies:
            raise ValueError()
        self._query_strategy = query_strategy
        self.update_in = update_in
        self.update_out = update_out

        # active learning process
        self.radii = list()
        self.bal_accs = list()
        self.U = list()
        self.L_in = list()
        self.L_out = list()

    def learn(self, X, y, K_xx_s=None, n_iter_max=50):
        if self.kernel == 'precomputed' and K_xx_s is None:
            raise ValueError('If kernel=precomputed, then K_xx_s must be provided.')
        W = np.ones(y.shape[0])
        X = self._check_X(X)

        QS_function = self.get_QS(self._query_strategy)

        self.radii = list()
        self.bal_accs = list()
        if isinstance(X, np.ndarray):
            self.U = list(np.arange(X.shape[0]))
        else:
            raise ValueError()
        self.L_in = list()
        self.L_out = list()
        for i in range(n_iter_max):
            self._info(f'Iteration: {i}')

            # train / retrain
            self.fit(X, W=W)
            radius = np.sqrt(self.r_square)
            self._info(f'R = {radius}')
            self.radii.append(radius)
            y_pred = self.predict(X, K_xx_s)
            y_vals = self.decision_function(X, K_xx_s)
            bal_acc = balanced_accuracy_score(y, y_pred)
            self.bal_accs.append(bal_acc)
            # get most informative unknown sample outside decision boundary
            idx = QS_function(y_vals)
            self._info(f'Most informative sample is {idx} with decision value {y_vals[idx]}')
            # ask oracle for annotation
            annotation = y[idx]
            self._info(f'Annotation is {annotation} for most informative sample {idx}')
            # remove sample from U
            self.U.remove(idx)
            # adjust sample weight
            if annotation == 1:
                W[idx] = self.update_in
                self.L_in.append(idx)
            else:
                W[idx] = self.update_out
                self.L_out.append(idx)
            self._info(f'New sample weight W[{idx}]: {W[idx]}')
            self._info('')

    def get_QS(self, query_strategy):
        if query_strategy is "confidence_outside":
            return self._confidence_outside
        elif query_strategy is "random_outlier":
            return self._random_outlier
        raise ValueError(f"Invalid query strategy: `{query_strategy}`.")

    def _confidence_outside(self, y_vals):
        idx_in_U = np.abs(np.where(y_vals[self.U] <= 0, np.inf, y_vals[self.U])).argmin()
        idx = self.U[idx_in_U]
        return idx

    def _random_outlier(self, y_vals=None):
        rng = default_rng()
        return rng.choice(self.U)
