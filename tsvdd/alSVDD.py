from tsvdd.SVDD import SVDD
import numpy as np
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score


class alSVDD(SVDD):

    _query_strategies = ["uncertainty_outside", "random_outlier"]
    _metrics = ["MCC", "kappa"]

    def __init__(self,  kernel='tga', nu=0.05, C=0.02, degree=3, gamma=1,
                 coef0=0.0, tol=1e-5, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200,
                 verbose=True, query_strategy='uncertainty_outside',
                 update_in=10, update_out=0.01, metric='MCC'):
        super().__init__(kernel=kernel, nu=nu, C=C, degree=degree, gamma=gamma,
                 coef0=coef0, tol=tol, sigma=sigma, triangular=triangular,
                 normalization_method=normalization_method, shrinking=shrinking, cache_size=cache_size,
                 verbose=verbose)
        if query_strategy not in self._query_strategies:
            raise ValueError('Not a valid query strategy.')
        if metric not in self._metrics:
            raise ValueError('Not a valid metric.')
        self.query_strategy = query_strategy
        self.metric = metric
        self.update_in = update_in
        self.update_out = update_out

        # active learning process
        self.radii = list()
        self.quality_metrics = list()
        self.AEQs = list()
        self.LSs = list()
        # sets
        self.U = list()
        self.L_in = list()
        self.L_out = list()

    def learn(self, X, y, K_xx_s=None, n_iter_max=50, k=5):

        # QS selects most `informative` sample
        QS_function = self.load_QS()

        # keep track of radius throughout active learning
        self.radii = list()
        self.quality_metrics = list()
        # start quality
        # average end quality
        self.AEQs = list()
        # Learning Stability
        self.LSs = list()
        if isinstance(X, np.ndarray):
            self.U = list(np.arange(X.shape[0]))
        else:
            raise ValueError("Use numpy.")
        self.L_in = list()
        self.L_out = list()
        W = np.ones(X.shape[0], dtype=np.float64)

        # ugly, but needed for W
        if self.nu:
            self.C = 1.0 / (self.nu*X.shape[0])
        for i in range(n_iter_max):
            self._info(f'Iteration: {i}')

            # adjust instance weights
            # w_base = 1.0 / (len(self.U) + self.update_in * len(self.L_in) + self.update_out * len(self.L_out))
            # W_ = np.zeros(X.shape[0], dtype=np.longfloat)
            # W_[self.L_in] = self.update_in * w_base
            # W_[self.L_out] = self.update_out * w_base
            # W_[self.U] = w_base
            # if not np.isclose(W_.sum(dtype=np.longfloat), 1):
            #     raise ValueError(f"Bachelor Degree denied; {W_.sum(dtype=np.longfloat)}")
            # W = W_ / self.C
            # W = W.astype(dtype=np.float64, order='C')

            W[self.L_in] = self.update_in
            W[self.L_out] = self.update_out
            W[self.U] = 1

            # train / retrain
            self.fit(X, W=W)
            print(self.rho)
            # radius
            radius = np.sqrt(self.r_square)
            self._info(f'R = {radius}')
            self.radii.append(radius)
            # predict
            y_pred = self.predict(X, K_xx_s)
            y_vals = self.decision_function(X, K_xx_s)
            # quality score
            score = self._calc_quality(y, y_pred)
            self.quality_metrics.append(score)
            # Average End Quality
            average_end_quality = self._calc_AEQ(i, k)
            self.AEQs.append(average_end_quality)
            # Learning Stability
            learning_stability = self._calc_LS(i, k)
            self.LSs.append(learning_stability)
            self._info(f'BA: {balanced_accuracy_score(y, y_pred)}')
            # stop active learning cycle when learning stability reaches zero
            if learning_stability == 0 and i not in range(5):
                break
            # get most informative unknown sample outside decision boundary
            idx = QS_function(self.U, self.L_in, self.L_out, y_vals)
            self._info(f'Most informative sample is {idx} with decision value {y_vals[idx]}')
            # ask oracle for annotation
            annotation = y[idx]
            self._info(f'Annotation is {annotation} for most informative sample {idx}')
            # remove sample from U
            self.U.remove(idx)
            # adjust sample weight
            if annotation == 1:
                self.L_in.append(idx)
            else:
                self.L_out.append(idx)
            self._info('')

    def load_QS(self):
        from ._query_strategies import get_QS
        return get_QS(self.query_strategy)

    def _calc_quality(self, y_true, y_pred):
        if self.metric == 'MCC':
            return matthews_corrcoef(y_true, y_pred)
        if self.metric == 'kappa':
            return cohen_kappa_score(y_true, y_pred)
        raise ValueError(f'Metric {self.metric} is not supported.')

    def _calc_AEQ(self, iteration, k):
        """
        AEQ(k) = \frac{1}{k}\sum_{i=1}^k QM(t_{end-k})
        """
        # iteration might be smaller than k
        k_ = min(iteration, k)
        AEQ = 0
        for i in range(k_):
            AEQ += self.quality_metrics[iteration-k_]
        if k_ == 0:
            return AEQ
        else:
            return AEQ/k_

    def _calc_LS(self, iteration, k):
        """
        Learning Stability describes the influence of the last k iterations.
        High LS indicates improvement with further iterations.
        Low LS indicates that classifier might be saturated.
        \begin{equation}
            L S(k)=\left\{\begin{array}{ll}
            \frac{Q R(\text { end }-k, \text { end })}{k} / \frac{Q R(\text { init,end })}{\mid \mathcal{L}^{\text {end }} \backslash \mathcal{L}^{\text {init }\mid}} & \text { if } Q R(\text { init, end })>0 \\0 & \text { otherwise. }
            \end{array}\right.
        \end{equation}

        Note: \mid \mathcal{L}^{\text {end }} \backslash \mathcal{L}^{\text {init }\mid equals iteration. Because initial pool is empty.
        """

        def QR(i, j):
            if j<=i:
                return self.quality_metrics[i] - self.quality_metrics[j]
            else:
                raise ValueError('`i` is smaller than `j`, I cant go back in time.')
        k_ = min(iteration, k)
        # check if quality improved
        if QR(iteration, 0) > 0:
            return (QR(iteration, iteration-k_) / k_) / (QR(iteration, 0) / iteration)
        else:
            return 0


