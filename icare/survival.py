import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from icare.metrics import *


def drop_collinear_features(X, method, cutoff):
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    return np.array([x for x in X.columns if x not in to_drop])


def format_x(X):
    check_array(X, force_all_finite='allow-nan')

    assert isinstance(X, (pd.DataFrame, np.ndarray))
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(data=X,
                         columns=np.arange(X.shape[1]).astype('str'))
    X = X.astype('float32')
    return X.copy()


sign_eval_method = {
    'HR': hazard_ratio,
    'uno': uno_cindex,
    'tAUC': tAUC,
    'harrell': harrell_cindex,
}

sign_eval_criteria = {
    'HR': 0,
    'uno': 0.5,
    'tAUC': 0.5,
    'harrell': 0.5,
}

method_ignore_score = ['HR']


class IcareSurv(BaseEstimator):
    """
        IcareSurv estimator.
        This survival model assign weights of -1 or +1 to features to reduce
        overfitting risk. It evaluates features in univariate to reduce this risk
        even more.
        Predicts a continuous value that rank the samples (not calibrated).

        Parameters
        ----------
        rho : None or float in [0, 1[, default=0.5
            The threshold for the correlation removal. Is None, no feature
            removal is done. If defined, it will randomly drop a feature of
            each pair that has an absolute correlation > rho.

        correlation_method: 'pearson' or 'spearman', default='pearson'
            How to compute the correlation between feature

        sign_method: a list, default=['tAUC', 'harrell', 'uno']
            Which method to use to determine the signs of the feature.
            It should be list containing at least one of :
            - 'harrell': Harrell's concordance index (C-index)
            - 'uno': 'Uno's concordance index (attempt to make Harrell's C-index
            more robust)
            - 'tAUC': time-dependant ROC-AUC
            If multiple method are selected, only the features that have the
            same sign for all methods will be used by the model

        cmin: None or a float in [0.5, 1[, default=0.6
            The threshold for feature selection based on their scores with the
            signs method. For each score of each method abs_score is computed
            (defined by max(score, 1-score)). Then mean_abs_score is computed
            by averaging all abs_score. If mean_abs_score < cmin, the feature
            is dropped

        max_features: float in [0,1], default=0.9
            Proportion of feature to randomly select at the beginning of the
            fitting. It happens before any feature selection.

        features_groups_to_use: None or list, default=None
            In a context where features are grouped (e.g. radiomic features)
            from the same mask, this is used to specify which groups should be
            used by the model. The random feature selection of max_features
            happens before this step. If None, all the features are kept.
            If this is defined, the argument feature_groups of the fit()
            function should be defined.

        mandatory_features: None or list, default=None
            List of feature to always include in the model.

        random_state : int, RandomState instance or None, default=None
            Control all the random steps of the model

        Attributes
        ----------

        Examples
        --------
        """

    def __init__(self,
                 rho=0.5,
                 correlation_method='pearson',
                 sign_method='tAUC,harrell,uno',
                 cmin=0.6,
                 max_features=1.,
                 features_groups_to_use=None,
                 mandatory_features=None,
                 random_state=None):

        self.correlation_method = correlation_method
        self.rho = rho
        self.cmin = cmin
        self.sign_method = sign_method
        self.max_features = max_features
        self.random_state = random_state
        self.mandatory_features = mandatory_features
        self.features_groups_to_use = features_groups_to_use

    def _more_tags(self):
        return {
            'allow_nan': True,
        }

    def fit(self, X, y, feature_groups=None):
        """
            Fit the model on the given data.
            Parameters
            ----------
            X : numpy array or pandas DataFrame with shape = (n_samples,n_features)
                Feature set
            y : 1d numpy-array, or structured array, shape = (n_samples,))
                Target event to predict
                OR
                Survival times of test data. A structured array containing
                the binary event indicator as first field,
                and time of event or time of censoring as second field.
            feature_groups : array-like of shape (n_features,), default=None
                Indicates to which group each feature belongs. If None,
                no feature group selection is done.
                If the parameter features_groups_to_use is not None, this argument
                should be specified.

            Returns
            -------
            The fitted model.
            """

        X = format_x(X)
        original_X = X.copy()
        X = X.copy()

        self.n_features_in_ = X.shape[1]

        if feature_groups is not None:
            assert len(feature_groups) == X.shape[1]
            assert self.features_groups_to_use is not None
            feature_groups = np.array(feature_groups)
        if self.features_groups_to_use is not None:
            assert feature_groups is not None

        self.sign_method_lst_ = self.sign_method.split(',')

        self.parameters_ = {}
        self.rng_ = np.random.default_rng(self.random_state)
        self.used_features_ = None
        self.weights_ = None
        self.std_f_ = None
        self.mean_f_ = None
        self.used_features_ = X.columns.values.copy()
        self.feature_groups_ = feature_groups

        # randomly select some features and keep the ones that are mandatory
        # or in defined features groups
        if self.max_features < 1:
            rand_nb_features = np.max([1, int(self.max_features * len(self.used_features_))])
            rand_feature_idx = self.rng_.choice(np.arange(len(self.used_features_)),
                                                rand_nb_features,
                                                replace=False)
            if self.mandatory_features is not None:
                mandatory_features_idx = []
                feature_list = self.used_features_.tolist()
                for f in self.mandatory_features:
                    mandatory_features_idx.append(feature_list.index(f))
                rand_feature_idx = np.concatenate([rand_feature_idx, np.array(mandatory_features_idx)])

            self.used_features_ = self.used_features_[rand_feature_idx]
            if self.feature_groups_ is not None:
                self.feature_groups_ = self.feature_groups_[rand_feature_idx]

            if len(self.used_features_) == 1:
                self.used_features_ = [self.used_features_]

        if self.features_groups_to_use is not None:
            sfeatures = []
            for fid in range(len(self.used_features_)):
                g = self.feature_groups_[fid]
                if g in self.features_groups_to_use or \
                        self.used_features_[fid] in self.mandatory_features:
                    sfeatures.append(fid)

            self.used_features_ = self.used_features_[sfeatures]
            self.feature_groups_ = self.feature_groups_[sfeatures]

            if len(self.used_features_) == 1:
                self.used_features_ = [self.used_features_]

        X = X[self.used_features_]

        # drop highly correlated features
        self.rng_.shuffle(self.used_features_)
        X = X[self.used_features_]
        if self.rho is not None and self.rho < 1:
            self.used_features_ = drop_collinear_features(X,
                                                          method=self.correlation_method,
                                                          cutoff=self.rho)
            X = X[self.used_features_]

        # evaluate weight and score of each feature
        self.weights_ = np.zeros(len(self.used_features_))
        for feature_id in range(len(self.used_features_)):
            feature = self.used_features_[feature_id]
            mask_not_nan = ~X[feature].isna().values

            signs, scores = [], []
            for method in self.sign_method_lst_:
                try:
                    score = sign_eval_method[method](y[mask_not_nan],
                                                     X.iloc[mask_not_nan][feature].values)
                    if score < sign_eval_criteria[method]:
                        signs.append(-1.)
                    else:
                        signs.append(1.)
                except:
                    score = np.nan

                if method not in method_ignore_score:
                    scores.append(score)

            if len(scores) > 0 and not np.isnan(np.unique(scores)[0]):
                mean_score = np.nanmean(scores)
                abs_mean_score = np.max([1. - mean_score, mean_score])
                if self.cmin is not None and abs_mean_score >= self.cmin:
                    if len(np.unique(signs)) == 1:
                        self.weights_[feature_id] = signs[0]

        # select features with a determied weight and score > cmin
        mask_feature_non_zero = self.weights_ != 0
        self.weights_ = self.weights_[mask_feature_non_zero]
        self.used_features_ = self.used_features_[mask_feature_non_zero]
        X = X[self.used_features_]

        # features normalization
        self.mean_f_, self.std_f_ = X.mean(), X.std()
        self.std_f_[self.std_f_ == 0] = 1e-6

        # output normalization
        self.mean_pred_, self.std_pred_ = 0, 1.
        train_pred = self.predict(original_X)
        self.mean_pred_, self.std_pred_ = np.nanmean(train_pred), np.nanstd(train_pred)
        if self.std_pred_ == 0:
            self.std_pred_ = 1e-6

        self.features_signs_ = self.used_features_, self.weights_

        return self

    def predict(self, X):
        X = format_x(X)
        X = X.copy()
        if len(self.used_features_) == 0:
            return np.ones(X.shape[0])

        X = X[self.used_features_]

        X = (X - self.mean_f_.values) / self.std_f_.values
        X = X * self.weights_
        pred = np.nanmean(X, axis=1)

        pred = (pred - self.mean_pred_) / self.std_pred_

        return pred

    def get_params(self, deep=False):
        return {
            'correlation_method': self.correlation_method,
            'rho': self.rho,
            'cmin': self.cmin,
            'sign_method': self.sign_method,
            'max_features': self.max_features,
            'random_state': self.random_state,
        }


class BaggedIcareSurv(IcareSurv):
    """
        Bagged IcareSurv estimator.
        This survival model ensemble a set of IcareSurv models with bootstrap
        resampling of their respective train sets. Once fitted, each model
        make a prediction and all teir prediction are aggregated.
        Predicts a continuous value that rank the samples (not calibrated).

        Parameters
        ----------
        n_estimators : int >= 1, default=10
            How many IcareSurv estimators should be used by the model

        parameters_sets : list of dictionaries or None, default=None
            A list a dictionaries. Each dictionary is a set of hyperparameters
            for the IcareSurv estimator. If the length of this list is > 1, then
            each estimator will randomly pick one of the hyperparameters sets of
            the list. If None, the default parameters of IcareSurv is used for
            all estimator.

        aggregation_method='mean' or 'median, default='mean'
            How to aggreate the predictions of the estimator.

        n_jobs=int, default=None
        The number of jobs to run in parallel. :meth:`fit` and :meth:`predict`
        are all parallelized over the estimators.
        None means 1. -1 means using all processors.

        random_state : int, RandomState instance or None, default=None
            Control all the random steps of the model

        Attributes
        ----------

        Examples
        --------
        """

    def __init__(self,
                 n_estimators=10,
                 parameters_sets=None,
                 aggregation_method='mean',
                 n_jobs=1,
                 random_state=None):

        self.n_estimators = n_estimators
        self.parameters_sets = parameters_sets
        self.aggregation_method = aggregation_method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def create_estimators(self):
        self.estimators_ = []
        self.estimators_random_state_ = []
        for _ in range(self.n_estimators):
            if self.random_state is None:
                estimator_random_state = None
            else:
                estimator_random_state = self.rng_.integers(low=0, high=99999999, size=1)[0]
            self.estimators_random_state_.append(estimator_random_state)

            if self.parameters_sets is None:
                model = IcareSurv(random_state=estimator_random_state)
                self.estimators_.append(model)
            else:
                params = self.rng_.choice(self.parameters_sets, 1)[0]
                params['random_state'] = estimator_random_state
                model = IcareSurv(**params)
                self.estimators_.append(model)

    def fit_estimator(self, estimator_id, X, y, feature_groups):
        estimator = self.estimators_[estimator_id]
        estimators_random_state = self.estimators_random_state_[estimator_id]
        estimator_rng = np.random.default_rng(estimators_random_state)
        resample_idx = estimator_rng.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        estimator.fit(X.iloc[resample_idx], y[resample_idx], feature_groups)
        return estimator

    def fit(self, X, y, feature_groups=None):
        """
            Fit the model on the given data.
            Parameters
            ----------
            X : numpy array or pandas DataFrame with shape = (n_samples,n_features)
                Feature set
            y : 1d numpy-array, or structured array, shape = (n_samples,))
                Target event to predict
                OR
                Survival times of test data. A structured array containing
                the binary event indicator as first field,
                and time of event or time of censoring as second field.
            feature_groups : array-like of shape (n_features,), default=None
                Indicates to which group each feature belongs. If None,
                no feature group selection is done.
                If the parameter features_groups_to_use is not None, this argument
                should be specified.

            Returns
            -------
            The fitted model.
        """

        X = format_x(X)
        X = X.copy()

        self.n_features_in_ = X.shape[1]

        self.rng_ = np.random.default_rng(self.random_state)

        if self.aggregation_method == 'mean':
            self.aggregation_method_fn_ = np.nanmean
        elif self.aggregation_method == 'median':
            self.aggregation_method_fn_ = np.nanmedian

        self.create_estimators()

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_estimator)(estimator_id, X, y, feature_groups)
            for estimator_id in range(self.n_estimators))

        self.average_feature_signs_ = self.get_feature_importance()

        return self

    def predict_estimator(self, estimator, X):
        return estimator.predict(X)

    def predict(self, X):
        X = format_x(X)
        X = X.copy()

        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_estimator)(estimator, X)
            for estimator in self.estimators_)

        preds = np.array(preds)
        return self.aggregation_method_fn_(preds,
                                           axis=0)

    def get_params(self, deep=False):
        return {
            'n_estimators': self.n_estimators,
            'parameters_sets': self.parameters_sets,
            'aggregation_method': self.aggregation_method,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }

    def get_feature_importance(self):
        all_importances = {}
        for m in self.estimators_:
            features, weights = m.features_signs_
            if len(features) == 0:
                continue
            for fid in range(len(features)):
                if features[fid] not in all_importances:
                    all_importances[features[fid]] = [weights[fid]]
                else:
                    all_importances[features[fid]].append(weights[fid])
        if len(all_importances) == 0:
            return None
        rows = []
        for f in all_importances:
            rows.append({
                'feature': f,
                'average sign': np.nanmean(all_importances[f])
            })
        importance = pd.DataFrame(data=rows).sort_values(by='average sign', ascending=False)
        return importance
