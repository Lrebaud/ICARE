import pandas as pd
import numpy as np
from icare.metrics import *
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


def drop_collinear_features(X, method, cutoff):
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    return np.array([x for x in X.columns if x not in to_drop])


def format_X(X):
    check_array(X, force_all_finite='allow-nan')
    #self.cols_name_ = X.columns.tolist()
    #X = pd.DataFrame(data=X, columns=self.cols_name_)

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
    def __init__(self,
                 rho=None,
                 correlation_method='pearson',
                 sign_method=['harrell'],
                 cmin=None,
                 normalize_output=False,
                 max_features=1.,
                 features_groups_to_use=None,
                 mandatory_features=None,
                 random_state=None):

        self.correlation_method = correlation_method
        self.rho = rho
        self.cmin = cmin
        self.sign_method = sign_method
        self.normalize_output = normalize_output
        self.max_features = max_features
        self.random_state = random_state
        self.mandatory_features = mandatory_features
        self.features_groups_to_use = features_groups_to_use

    def _more_tags(self):
        return {
            'allow_nan': True,
        }

    def fit(self, X, y, feature_groups=None):
        X = format_X(X)
        original_X = X.copy()
        X = X.copy()

        if feature_groups is not None:
            assert len(feature_groups) == X.shape[1]
            assert self.features_groups_to_use is not None
            feature_groups = np.array(feature_groups)
        if self.features_groups_to_use is not None:
            assert feature_groups is not None

        self.parameters_ = {}
        self.rng_ = np.random.default_rng(self.random_state)
        self.used_features_ = None
        self.weights_ = None
        self.std_f_ = None
        self.mean_f_ = None
        self.used_features_ = X.columns.values.copy()
        self.feature_groups_ = feature_groups

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
            for method in self.sign_method:
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

            mean_score = np.nanmean(scores)
            abs_mean_score = np.max([1. - mean_score, mean_score])
            if self.cmin is not None and abs_mean_score >= self.cmin:
                if len(np.unique(signs)) == 1:
                    self.weights_[feature_id] = signs[0]

        #  select features with a determied weight and score > cmin
        mask_feature_non_zero = self.weights_ != 0
        self.weights_ = self.weights_[mask_feature_non_zero]
        self.used_features_ = self.used_features_[mask_feature_non_zero]

        X = X[self.used_features_]

        # features normalization
        self.mean_f_, self.std_f_ = X.mean(), X.std()
        self.std_f_[self.std_f_ == 0] = 1e-6

        #  output normalization
        if self.normalize_output:
            self.mean_pred, self.std_pred = 0, 1.
            train_pred = self.predict(original_X)
            self.mean_pred, self.std_pred = np.nanmean(train_pred), np.nanstd(train_pred)
            if self.std_pred == 0:
                self.std_pred = 1e-6

        return self

    def predict(self, X):
        X = format_X(X)
        X = X.copy()
        if len(self.used_features_) == 0:
            return np.ones(X.shape[0])

        X = X[self.used_features_]


        X = (X - self.mean_f_.values) / self.std_f_.values
        X = X * self.weights_
        pred = np.nanmean(X, axis=1)

        if self.normalize_output:
            pred = (pred - self.mean_pred) / self.std_pred

        return pred

    def get_params(self, deep=False):
        return {
            'correlation_method': self.correlation_method,
            'rho': self.rho,
            'cmin': self.cmin,
            'sign_method': self.sign_method,
            'normalize_output': self.normalize_output,
            'max_features': self.max_features,
            'random_state': self.random_state,
        }

    def get_feature_importance(self):
        return self.used_features_, self.weights_


from joblib import Parallel, delayed


class BaggedIcareSurv(IcareSurv):
    def __init__(self,
                 n_estimators=1,
                 parameters_sets=None,
                 aggregation_method='mean',
                 normalize_estimators=True,
                 n_jobs=1,
                 random_state=None):

        self.n_estimators = n_estimators
        self.parameters_sets = parameters_sets
        self.aggregation_method = aggregation_method
        self.normalize_estimators = normalize_estimators
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
                params['normalize_output'] = self.normalize_estimators
                params['random_state'] = estimator_random_state
                model = IcareSurv(**params)
                self.estimators_.append(model)

    def fit_estimator(self, estimator_id, X, y, feature_groups):
        estimator = self.estimators_[estimator_id]
        estimators_random_state = self.estimators_random_state_[estimator_id]
        estimator_rng = np.random.default_rng(estimators_random_state)
        resample_idx = estimator_rng.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        estimator.fit(X.iloc[resample_idx], y[resample_idx], feature_groups)

        # print(estimator.get_feature_importance())
        return estimator

    def fit(self, X, y, feature_groups=None):
        X = format_X(X)
        X = X.copy()

        self.rng_ = np.random.default_rng(self.random_state)

        if self.aggregation_method == 'mean':
            self.aggregation_method_fn_ = np.nanmean
        elif self.aggregation_method == 'median':
            self.aggregation_method_fn_ = np.nanmedian

        self.create_estimators()

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_estimator)(estimator_id, X, y, feature_groups)
            for estimator_id in range(self.n_estimators))

        return self

    def predict_estimator(self, estimator, X):
        return estimator.predict(X)

    def predict(self, X):
        X = format_X(X)
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
            'normalize_estimators': self.normalize_estimators,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }

    def get_feature_importance(self):
        all_importances = {}
        for m in self.estimators_:
            features, weights = m.get_feature_importance()
            for fid in range(len(features)):
                if features[fid] not in all_importances:
                    all_importances[features[fid]] = [weights[fid]]
                else:
                    all_importances[features[fid]].append(weights[fid])
        rows = []
        for f in all_importances:
            rows.append({
                'feature': f,
                'average sign': np.nanmean(all_importances[f])
            })
        importance = pd.DataFrame(data=rows).sort_values(by='average sign', ascending=False)
        return importance
