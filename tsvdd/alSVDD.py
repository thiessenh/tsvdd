from tsvdd.SVDD import SVDD
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, f1_score

from .query_strategies import query_strategies


class alSVDD(SVDD):
    """
    Active Learning Support Vector Data Description

    Initially, the active learning process starts without labeled data and with sample weights W = 1. At this point alSVDD is equivalent to SVDD.
    After fitting the alSVDD to the training data, a query strategy selects an observation x_i for annotation. According to the feedback, its sample weight w_i is updated with either update_in or update_out.
    Subsequently, the model is refitted. We repeat this until we either reach the feedback budget max_iter_AL which is user defined restriction depending on the availability of annotations, or the LS drops to 0.
    The start_up_phase prevents a premature interruption. That is, as long as the current active learning cycle is less than start_up_phase, the algorithm keeps learning. 
    """
    _query_strategies = query_strategies
    _metrics = ["MCC", "kappa"]

    def __init__(self, kernel='tga', nu=0.05, C=None,
                 tol=1e-5, sigma='auto', triangular='auto',
                 normalization_method='exp', shrinking=False, cache_size=200, max_iter=100000,
                 verbose=True, query_strategy='uncertainty_outside',
                 update_in=10, update_out=0.01, metric='MCC', max_iter_AL=50, start_up_phase=5, additional_metrics=False):
        """Active Learning Support Vector Data Description

        Parameters
        ----------
        kernel : str, optional
            Only tested for `tga`, but should work for other kernel methods as well, by default 'tga'.
        nu : [type], optional
            See SVDD's doc string, by default None.
        C : float, optional
            See SVDD's doc string, by default 0.02.
        tol : [type], optional
            See SVDD's doc string, by default 1e-5.
        sigma : str, optional
            See SVDD's doc string, by default 'auto'.
        triangular : str, optional
            See SVDD's doc string, by default 'auto'.
        normalization_method : str, optional
            See SVDD's doc string, by default 'exp'.
        shrinking : bool, optional
            See SVDD's doc string, by default False.
        cache_size : int, optional
            See SVDD's doc string, by default 200.
        max_iter : int, optional
            See SVDD's doc string, by default 100000.
        verbose : bool, optional
            See SVDD's doc string, by default True.
        query_strategy : str, optional
            Choose among a uncertainty and random-based query strategies, by default 'uncertainty_outside'.
        update_in : int, optional
            The factor with which an annotated inlier's weight is multiplied, by default 10.
        update_out : float, optional
           The factor with which an annotated outlier's weight is multiplied, by default 0.01
        metric : str, optional
            Choose between`MCC` and `kappa` for monitoring and stopping learning process, by default 'MCC'.
        max_iter_AL : int, optional
            Maximum active learning cycles, by default 50.
        start_up_phase : int, optional
           Minimum number of iterations, by default 5.
        additional_metrics : bool, optional
            Whether additional metrics should be logged, by default False.
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
        self.additional_metrics = additional_metrics

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
        """Starts active Learning process

        Parameters
        ----------
        X : np.ndarray
            The training data with shape (n_instances, n_time_series_length)
        y : np.ndarray
            True labels, servers as oracle.
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
        self.U = list(np.arange(X.shape[0]))
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
            if self.additional_metrics == True:
                self.ba_on_train.append(balanced_accuracy_score(y, y_pred))
                self.ba_ramp_up_on_train.append(
                    balanced_accuracy_score(y, y_pred) - self.ba_on_train[0])
                self.f1_on_train.append(f1_score(y, y_pred))
                self.f1_ramp_up_on_train.append(
                    f1_score(y, y_pred) - self.f1_on_train[0])

                self.ba_on_annotations.append(balanced_accuracy_score(
                    y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]))
                self.ba_ramp_up_on_annotations.append(balanced_accuracy_score(
                    y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]) - self.ba_on_annotations[0])
                self.f1_on_annotations.append(
                    f1_score(y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]))
                self.f1_ramp_up_on_annotations.append(f1_score(
                    y[self.L_in + self.L_out], y_pred[self.L_in + self.L_out]) - self.f1_on_annotations[0])
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
            # first iteration does not count.
            if not self.additional_metrics and learning_stability <= 0 and i not in range(self.start_up_phase + 2):
                break
            # get most informative unknown sample
            idx = QS_function(self.U, self.L_in, self.L_out, y_vals)
            self._info(
                f'Most informative sample is {idx} with decision value {y_vals[idx]}')
            # ask oracle for annotation
            annotation = y[idx]
            self._info(
                f'Annotation is {annotation} for most informative sample {idx}')
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
        """Gets the query strategy.

        Returns
        -------
        function
            The query strategy.
        """
        from .query_strategies import get_QS
        return get_QS(self.query_strategy)

    def _calc_quality(self, y_true, y_pred):
        """Computes the chosen metric based on L_o and L_in

        Parameters
        ----------
        y_true : array-based
            [description]
        y_pred : array-based
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
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
                raise ValueError(
                    '`end` is smaller than `start`, I can\'t go back in time.')

        k_ = min(iteration, k)
        # check if quality improved
        if QR(0, iteration) > 0:
            return (QR(iteration - k_, iteration) / k_) / (QR(0, iteration) / iteration)
        else:
            return 0
