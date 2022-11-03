import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from joblib import Parallel, delayed


def format_x(X):
    X = X.copy()
    check_array(X, force_all_finite='allow-nan')

    assert isinstance(X, (pd.DataFrame, np.ndarray))
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(data=X,
                         columns=np.arange(X.shape[1]).astype('str'))
    X = X.astype('float32')
    return X


def drop_collinear_features(X, method, cutoff):
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    return np.array([x for x in X.columns if x not in to_drop])


class IcareBase(BaseEstimator):

    def __init__(self,
                 rho,
                 correlation_method,
                 sign_method,
                 cmin,
                 max_features,
                 features_groups_to_use,
                 mandatory_features,
                 random_state,
                 sign_eval_method,
                 sign_eval_criteria,
                 abs_fnc):

        self.correlation_method = correlation_method
        self.rho = rho
        self.cmin = cmin
        self.sign_method = sign_method
        self.max_features = max_features
        self.random_state = random_state
        self.mandatory_features = mandatory_features
        self.features_groups_to_use = features_groups_to_use
        self.sign_eval_method = sign_eval_method
        self.sign_eval_criteria = sign_eval_criteria
        self.abs_fnc = abs_fnc

    def _more_tags(self):
        return {
            'allow_nan': True,
        }

    def fit(self, X, y, feature_groups=None):
        X = format_x(X)
        original_X = X.copy()

        if isinstance(y, list):
            y = np.array(y)

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

            with open("testoutput.txt", "a") as myfile:
                myfile.write('cc before\n')

            if self.mandatory_features is not None:
                mandatory_features_idx = []
                with open("testoutput.txt", "a") as myfile:
                    myfile.write('cc ok\n')
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
        self.scores = {}
        for method in self.sign_method_lst_:
            self.scores[method] = np.zeros(len(self.used_features_))
        for feature_id in range(len(self.used_features_)):
            feature = self.used_features_[feature_id]
            mask_not_nan = ~X[feature].isna().values

            signs, scores, abs_scores = [], [], []
            for method in self.sign_method_lst_:

                score = self.sign_eval_method[method](y[mask_not_nan], X.iloc[mask_not_nan][feature].values)
                abs_score = self.abs_fnc[method](score)

                if score < self.sign_eval_criteria[method]:
                    signs.append(-1.)
                else:
                    signs.append(1.)

                self.scores[method][feature_id] = score
                scores.append(score)
                abs_scores.append(abs_score)

            if len(np.unique(signs)) == 1:
                if self.cmin is None:
                    self.weights_[feature_id] = signs[0]
                else:
                    if len(scores) > 0 and not np.isnan(np.unique(scores)[0]):
                        abs_mean_score = np.nanmean(abs_scores)
                        if abs_mean_score >= self.cmin:
                            self.weights_[feature_id] = signs[0]

        self.df_scores = pd.DataFrame()
        for method in self.sign_method_lst_:
            self.df_scores[method] = self.scores[method]
        self.df_scores.index = self.used_features_

        self.df_signs = pd.DataFrame()
        for method in self.sign_method_lst_:
            self.df_signs[method] = (self.df_scores[method] > self.sign_eval_criteria[method]) * 2. - 1.
        self.df_signs.index = self.used_features_

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

        return self

    def predict(self, X):
        X = format_x(X)
        if len(self.used_features_) == 0:
            return np.ones(X.shape[0])

        X = X[self.used_features_]

        X = (X - self.mean_f_.values) / self.std_f_.values
        X = X * self.weights_
        pred = np.nanmean(X, axis=1)
        pred = (pred - self.mean_pred_) / self.std_pred_
        return pred


class BaggedIcareBase(IcareBase):

    def __init__(self,
                 estimator,
                 n_estimators=10,
                 parameters_sets=None,
                 aggregation_method='mean',
                 n_jobs=1,
                 random_state=None):

        self.estimator = estimator
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
                model = self.estimator(random_state=estimator_random_state)
                self.estimators_.append(model)
            else:
                params = self.rng_.choice(self.parameters_sets, 1)[0]
                params['random_state'] = estimator_random_state
                model = self.estimator(**params)
                self.estimators_.append(model)

    def fit_estimator(self, estimator, estimators_random_state, X, y, feature_groups):
        estimator_rng = np.random.default_rng(estimators_random_state)
        resample_idx = estimator_rng.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        estimator.fit(X.iloc[resample_idx], y[resample_idx], feature_groups)
        return estimator

    def fit(self, X, y, feature_groups=None):
        X = format_x(X)

        self.n_features_in_ = X.shape[1]

        self.rng_ = np.random.default_rng(self.random_state)

        if self.aggregation_method == 'mean':
            self.aggregation_method_fn_ = np.nanmean
        elif self.aggregation_method == 'median':
            self.aggregation_method_fn_ = np.nanmedian

        self.create_estimators()

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_estimator)(self.estimators_[estimator_id], self.estimators_random_state_[estimator_id], X,
                                        y, feature_groups)
            for estimator_id in range(self.n_estimators))

        self.average_feature_signs_ = self.get_feature_importance()

        return self

    def predict(self, X):
        X = format_x(X)

        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(estimator.predict)(X)
            for estimator in self.estimators_)

        preds = np.array(preds)
        return self.aggregation_method_fn_(preds,
                                           axis=0)

    def get_feature_importance(self):
        all_importances = {}
        for m in self.estimators_:
            df_signs = m.df_signs
            df_signs = df_signs[df_signs.sum(axis=1).abs() == df_signs.shape[1]]  # keep features with consistent signs
            features, weights = df_signs.index.values, df_signs.iloc[:, 0].values
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
