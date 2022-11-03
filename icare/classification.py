from icare.ranker import IcareRanker, BaggedIcareRanker
from sklearn.base import clone
import numpy as np


def one_vs_rest_fit(X, y, feature_groups, n_classes, estimator, classes):
    estimators = []
    for i in range(n_classes):
        ce = clone(estimator)
        cy = (y == classes[i]).astype('int8')
        ce.fit(X, cy, feature_groups)
        estimators.append(ce)
    return estimators


def one_vs_rest_predict(classes, n_classes, estimators, X):
    all_preds = []
    for i in range(n_classes):
        all_preds.append(estimators[i].predict(X))
    all_preds = np.stack(all_preds)
    pred_id = np.argmax(all_preds, axis=0)
    return classes[pred_id]


class IcareClassifier(IcareRanker):
    """
        Icare estimator adapted for classification. One Icare ranker mdoel is trained for each class.
        The class of the model with the highest predicted value is the predicted class.
        This model assign weights of -1 or +1 to features to reduce overfitting risk.
        It evaluates features in univariate to reduce this risk even more.
        Parameters
        ----------
        rho : None or float in [0, 1[, default=None
            The threshold for the correlation removal. Is None, no feature
            removal is done. If defined, it will randomly drop a feature of
            each pair that has an absolute correlation > rho.
        correlation_method: 'pearson' or 'spearman', default='pearson'
            How to compute the correlation between feature
        sign_method: a string, default='pearson,spearman'
            Which method to use to determine the signs of the feature.
            It should be string containing at least one of :
            - 'pearson': Pearson's correlation with the target
            - 'spearman': Spearman's correlation with the target
            If multiple method are selected, they should be separated by a comma. In such cases,
            only the features that have the same sign for all methods will be used by the model
        cmin: None or a float in [0.5, 1[, default=None
            The threshold for feature selection based on their scores with the
            signs method. For each score of each method abs_score is computed
            (defined by max(score, 1-score)). Then mean_abs_score is computed
            by averaging all abs_score. If mean_abs_score < cmin, the feature
            is dropped
        max_features: float in ]0,1], default=1
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
    """

    def fit(self, X, y, feature_groups=None):
        self.estimator_ = IcareRanker(rho=self.rho,
                                      correlation_method=self.correlation_method,
                                      sign_method=self.sign_method,
                                      cmin=self.cmin,
                                      max_features=self.max_features,
                                      features_groups_to_use=self.features_groups_to_use,
                                      mandatory_features=self.mandatory_features,
                                      random_state=self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = one_vs_rest_fit(X, y, feature_groups,
                                           n_classes=self.n_classes_,
                                           estimator=self.estimator_,
                                           classes=self.classes_)
        return self

    def predict(self, X):
        return one_vs_rest_predict(self.classes_, self.n_classes_, self.estimators_, X)


class BaggedIcareClassifier(BaggedIcareRanker):
    """
        Icare estimator adapted for classification. One bagged Icare ranker mdoel is trained for each class.
        The class of the model with the highest predicted value is the predicted class.
        This model assign weights of -1 or +1 to features to reduce overfitting risk.
        It evaluates features in univariate to reduce this risk even more.
                Parameters
        ----------
        n_estimators : int >= 1, default=10
            How many IcareSurvival estimators should be used by the model
        parameters_sets : list of dictionaries or None, default=None
            A list a dictionaries. Each dictionary is a set of hyperparameters
            for the IcareSurvival estimator. If the length of this list is > 1, then
            each estimator will randomly pick one of the hyperparameters sets of
            the list. If None, the default parameters of IcareSurvival is used for
            all estimator.
        aggregation_method='mean' or 'median, default='mean'
            How to aggregate the predictions of the estimator.
        n_jobs=int, default=None
        The number of jobs to run in parallel. :meth:`fit` and :meth:`predict`
        are all parallelized over the estimators.
        None means 1. -1 means using all processors.
        random_state : int, RandomState instance or None, default=None
            Control all the random steps of the model
    """

    def fit(self, X, y, feature_groups=None):
        self.estimator_ = BaggedIcareRanker(n_estimators=self.n_estimators,
                                            parameters_sets=self.parameters_sets,
                                            aggregation_method=self.aggregation_method,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = one_vs_rest_fit(X, y, feature_groups,
                                           n_classes=self.n_classes_,
                                           estimator=self.estimator_,
                                           classes=self.classes_)
        return self

    def predict(self, X):
        return one_vs_rest_predict(self.classes_, self.n_classes_, self.estimators_, X)
