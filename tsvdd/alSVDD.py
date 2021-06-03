from tsvdd.SVDD import SVDD
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, f1_score

from .query_strategies import query_strategies


class alSVDD(SVDD):
    """
    Active Learning Support Vector Data Description
    """
    _query_strategies = query_strategies
    _metrics = ["MCC", "kappa"]

    def __init__(self, kernel='tga', nu=None, C=0.02, degree=3, gamma=1,
                 coef0=0.0, tol=1e-5, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200, max_iter=100000,
                 verbose=True, query_strategy='uncertainty_outside',
                 update_in=10, update_out=0.01, metric='MCC', max_iter_AL=50, start_up_phase=5, paper=False):
        """
        @param kernel: Currently only tga supported for active learning.
        @param nu: Expected outlier ratio
        @param C: If nu not provided, C will be used. C can be calculated as 1/(outlier_ratio * n_instances)
        @param degree: To be removed
        @param gamma: To be removed
        @param coef0: To be removed
        @param tol: Stopping criteria for SMO optimization.
        @param sigma: Sigma for Gaussian-like kernels.
        @param triangular: For tga kernel
        @param normalization_method: Method to normalize tga kernel.
        @param shrinking: Whether to remove bounded \alphas from working set during optimization
        @param cache_size: Cache size
        @param max_iter: Maximum iterations for SMO optimization
        @param verbose: Whether active learning process should be communicated
        @param query_strategy: Query strategies are defined in file `_query_strategies`
        @param update_in: Weight update for annotated inliers
        @param update_out:Weight update for annotated outliers
        @param metric: Metric for monitoring and stopping learning process
        @param max_iter_AL: Upper bound for oracle acquisitions
        @param start_up_phase: Minimum number of iterations; Likewise parameter for AEQ and LS
        """
        super().__init__(kernel=kernel, nu=nu, C=C, tol=tol, sigma=sigma, triangular=triangular,
                         normalization_method=normalization_method, shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, verbose=verbose)
        if query_strategy not in self._query_strategies:
            raise ValueError('Not a valid query strategy.')
        if metric not in self._metrics:
            raise ValueError('Not a valid metric.')
        self.start_up_phase = start_up_phase
        self.query_strategy = query_strategy
        self.metric = metric
        self.update_in = update_in
        self.update_out = update_out
        self.max_iter_AL = max_iter_AL
        self.paper = paper

        # active learning process
        self.radii = list()
        self.quality_metrics = list()
        self.AEQs = list()
        self.LSs = list()
        self.al_iterations = 0
        # sets
        self.U = list()
        self.L_in = list()
        self.L_out = list()

        # evaluation
        self.qms_on_train = list()

        self.ba_on_train = list()
        self.ba_ramp_up_on_train = list()

        self.f1_on_train = list()
        self.f1_ramp_up_on_train = list()

        self.ba_on_annotations = list()
        self.ba_ramp_up_on_annotations = list()

        self.f1_on_annotations = list()
        self.f1_ramp_up_on_annotations = list()
        

    def learn(self, X, y):
        """
        Starts active Learning cycle.
        @param X: numpy array containg training data
        @param y: annotations, serves as oracle

        """

        # QS selects most `informative` sample
        QS_function = self._load_QS()

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
            self.C = 1.0 / (self.nu * X.shape[0])

        if self.kernel == 'precomputed':
            K_xx_s = np.diag(X)
        else:
            K_xx_s = None

        for i in range(self.max_iter_AL):

            self._info(f'Iteration: {i}')

            # train / retrain
            self.fit(X, W=W)
            self._info(f'Rho = {self.rho}')
            # radius
            radius = np.sqrt(self.r_square)
            self._info(f'R = {radius}')
            self.radii.append(radius)
            # predict
            y_pred = self.predict(X, K_xx_s)
            y_vals = self.decision_function(X, K_xx_s)
            if self.paper == True:
                self.ba_on_train.append(balanced_accuracy_score(y, y_pred))
                self.ba_ramp_up_on_train.append(balanced_accuracy_score(y, y_pred) - self.ba_on_train[0])
                self.f1_on_train.append(f1_score(y, y_pred))
                self.f1_ramp_up_on_train.append(f1_score(y, y_pred) - self.f1_on_train[0])

                self.ba_on_annotations.append(balanced_accuracy_score(y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]))
                self.ba_ramp_up_on_annotations.append(balanced_accuracy_score(y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]) - self.ba_on_annotations[0])
                self.f1_on_annotations.append(f1_score(y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]))
                self.f1_ramp_up_on_annotations.append(f1_score(y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]) - self.f1_on_annotations[0])
            # quality score
            score = self._calc_quality(y, y_pred)
            self.quality_metrics.append(score)
            # evaluation
            _mcc = matthews_corrcoef(y, y_pred)
            self.qms_on_train.append(_mcc)
            # Average End Quality
            average_end_quality = self._calc_AEQ(i, self.start_up_phase)
            self.AEQs.append(average_end_quality)
            # Learning Stability
            learning_stability = self._calc_LS(i, self.start_up_phase)
            self.LSs.append(learning_stability)
            self._info(f'BA: {balanced_accuracy_score(y, y_pred)}')
            # stop active learning cycle when learning stability reaches zero; but allow start-up phase
            if not self.paper and learning_stability <= 0 and i not in range(self.start_up_phase + 2): # first iteration does not count.
                break
            # get most informative unknown sample
            idx = QS_function(self.U, self.L_in, self.L_out, y_vals)
            self._info(f'Most informative sample is {idx} with decision value {y_vals[idx]}')
            # ask oracle for annotation
            annotation = y[idx]
            self._info(f'Annotation is {annotation} for most informative sample {idx}')
            # remove sample from U
            self.U.remove(idx)
            # add sample to L_in or L_out
            if annotation == 1:
                self.L_in.append(idx)
            else:
                self.L_out.append(idx)

            W[self.L_in] = self.update_in
            W[self.L_out] = self.update_out
            W[self.U] = 1
            self._info('')
        self.al_iterations = i

    def _load_QS(self):
        """
        Get Query Strategy.
        """
        from .query_strategies import get_QS
        return get_QS(self.query_strategy)

    def _calc_quality(self, y_true, y_pred):
        """
        Calculate quality metric from annotated data.
        @param y_true: True values; only values from annotated samples are used
        @param y_pred: Predicted from alSVDD
        @return:
        """
        if self.metric == 'MCC':
            annotated_indices = self.L_in + self.L_out
            if not self.verbose:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return matthews_corrcoef(y_true[annotated_indices], y_pred[annotated_indices])
            else:
                return matthews_corrcoef(y_true[annotated_indices], y_pred[annotated_indices])
        if self.metric == 'kappa':
            annotated_indices = self.L_in + self.L_out
            return cohen_kappa_score(y_true[annotated_indices], y_pred[annotated_indices])
        raise ValueError(f'Metric {self.metric} is not supported.')

    def _calc_AEQ(self, iteration, k):
        """
         Median average end quality (AEQ), the average quality over the last k iterations
        AEQ(k) = \frac{1}{k}\sum_{i=1}^k QM(t_{end-k})
        @param iteration: Current iteration
        @param k: Average Quality Metric over k iterations
        @return:
        """
        # iteration might be smaller than k
        k_ = min(iteration, k)
        AEQ = 0
        for i in range(k_):
            AEQ += self.quality_metrics[iteration - k_]
        if k_ == 0:
            return AEQ
        else:
            return AEQ / k_

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
        @param iteration: Current iteration
        @param k: Specifies the range to consider for Learning Stability
        @return:
        """

        def QR(start, end):
            """
            Quality Range from start to end
            """
            if start <= end:
                return self.quality_metrics[end] - self.quality_metrics[start]
            else:
                raise ValueError('`end` is smaller than `start`, I can\'t go back in time.')

        k_ = min(iteration, k)
        # check if quality improved
        if QR(0, iteration) > 0:
            return (QR(iteration - k_, iteration) / k_) / (QR(0, iteration) / iteration)
        else:
            return 0
