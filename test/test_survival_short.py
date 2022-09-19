from sksurv import datasets
from icare.survival import IcareSurv, BaggedIcareSurv
from sksurv.preprocessing import OneHotEncoder
import numpy as np
from icare.metrics import *
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_score, ShuffleSplit
import pandas as pd
from joblib import Parallel, delayed


def generate_random_param_set():
    return {
        'rho': np.random.uniform(0.01, 1.),
        'correlation_method': np.random.choice(['spearman', 'pearson']),
        'sign_method': np.random.choice(['harrell', 'tAUC', 'uno'],
                                        np.random.randint(1, 4),
                                        replace=False).tolist(),
        'cmin': np.random.uniform(0.5, 1.),
        'normalize_output': True,
        'max_features': np.random.uniform(0.5, 1.),
        'random_state': None,
    }


def generate_random_param_set_bagged():
        return {
            'n_estimators': np.random.randint(1, 10),
            'parameters_sets': [generate_random_param_set() for x in range(np.random.randint(1, 3))],
            'aggregation_method': np.random.choice(['mean', 'median']),
            'normalize_estimators': np.random.choice([True, False]),
            'n_jobs': np.random.randint(1, 5),
            'random_state': None,
        }

def test_feature_group():
    X, y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = IcareSurv()
    ml.fit(X, y)

    feature_groups = None
    ml.fit(X,y, feature_groups=feature_groups)

    ml = IcareSurv(features_groups_to_use=[0,1])
    feature_groups = np.ones(int(X.shape[1]/2))
    feature_groups = np.concatenate([feature_groups, np.zeros(X.shape[1]-len(feature_groups))])
    ml.fit(X, y, feature_groups=feature_groups)
    harrell_cindex_scorer(ml, X, y)

def test_scikit_simple():
    check_estimator(IcareSurv())


def test_scikit_bagged():
    check_estimator(BaggedIcareSurv())

def test_no_censoring():
    X,y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = IcareSurv()
    ml.fit(X, y)

    y = [x[1] for x in y]
    ml.fit(X, y)

def test_scorer():
    X, y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = IcareSurv()
    ml.fit(X, y)
    harrell_cindex_scorer(ml, X, y)
    uno_cindex_scorer(ml, X, y)
    tAUC_scorer(ml, X, y)

def test_sksurv_datasets_simple():
    for X, y in [datasets.load_veterans_lung_cancer(),
                 datasets.load_whas500(),
                 datasets.load_gbsg2(),
                 # datasets.load_flchain(),
                 datasets.load_breast_cancer(),
                 datasets.load_aids()]:
        X = OneHotEncoder().fit_transform(X)

        for _ in range(100):
            # try:
            params = generate_random_param_set()
            ml = IcareSurv(**params)
            ml.fit(X, y)
            score = harrell_cindex_scorer(ml, X, y)

            # except:
            #    pass
            if score > 0.5:
                break
        assert score > 0.5

def test_sksurv_datasets_bagged():
    for X, y in [datasets.load_veterans_lung_cancer(),
                 datasets.load_whas500(),
                 datasets.load_gbsg2(),
                 # datasets.load_flchain(),
                 datasets.load_breast_cancer(),
                 datasets.load_aids()]:
        X = OneHotEncoder().fit_transform(X)

        for _ in range(10):
            # try:
            params = generate_random_param_set_bagged()
            ml = BaggedIcareSurv(**params)
            ml.fit(X, y)
            score = harrell_cindex_scorer(ml, X, y)
            # except:
            #    pass
            if score > 0.5:
                break
        assert score > 0.5