import pandas as pd
from sklearn.model_selection import cross_val_score, ShuffleSplit
from icare.metrics import *
from icare.ranker import IcareRanker, BaggedIcareRanker
from sklearn import datasets
import statsmodels.api as sm
from sklearn import preprocessing


dt_regres_list = [
    [sm.datasets.committee, 'BILLS104'],
    [sm.datasets.copper, 'WORLDCONSUMPTION'],
    [sm.datasets.grunfeld, 'invest'],
    [sm.datasets.longley, 'TOTEMP'],
]
dt_dict = []
for dt, target in dt_regres_list:
    name = dt.__name__.split('.')[-1]
    data = dt.load_pandas().data
    y = data[target]
    X = data.drop(columns=target)
    dt_dict.append({
        'data': X.values,
        'target': y.values
    })
dt_dict += [datasets.fetch_california_housing(),
            datasets.load_diabetes()]

def concordant_cindex_scorer(ml, X, y):
    return 1. - harrell_cindex_scorer(ml, X, y)

def generate_random_param_set():
    return {
        'rho': np.random.uniform(0.01, 1.),
        'correlation_method': np.random.choice(['spearman', 'pearson']),
        'sign_method': ','.join(np.random.choice(['pearson', 'spearman'],
                                                 np.random.randint(1, 3),
                                                 replace=False).tolist()),
        'cmin': np.random.uniform(0., 1.),
        'max_features': np.random.uniform(0.5, 1.),
        'random_state': None,
    }


def generate_random_param_set_bagged():
    return {
        'n_estimators': np.random.randint(1, 10),
        'parameters_sets': [generate_random_param_set() for x in range(np.random.randint(1, 3))],
        'aggregation_method': np.random.choice(['mean', 'median']),
        'n_jobs': np.random.randint(1, 16),
        'random_state': None,
    }



def test_scikit_datasets_simple():
    for data in dt_dict:
        X,y = data['data'], data['target']

        for j in range(X.shape[1]):
            if not np.issubdtype(X[:,j].dtype, np.number):
                X[:,j] = preprocessing.LabelEncoder().fit_transform(X[:,j])

        for _ in range(100):
            params = generate_random_param_set()
            ml = IcareRanker(**params)
            ml.fit(X, y)
            score = concordant_cindex_scorer(ml, X, y)
            if score > 0.5:
                break
        assert score > 0.5


def test_scikit_datasets_bagged():
    for data in dt_dict:
        X,y = data['data'], data['target']

        for j in range(X.shape[1]):
            if not np.issubdtype(X[:,j].dtype, np.number):
                X[:,j] = preprocessing.LabelEncoder().fit_transform(X[:,j])

        for _ in range(100):
            params = generate_random_param_set_bagged()
            ml = BaggedIcareRanker(**params)
            ml.fit(X, y)
            score = concordant_cindex_scorer(ml, X, y)
            if score > 0.5:
                break
        assert score > 0.5


def test_signs():
    for _ in range(100):
        X,y, coef = datasets.make_regression(n_samples=200,
                                             n_features=1,
                                             n_informative=1,
                                             coef=True)
        coef = [coef]
        real_sign = np.random.choice([-1,1], len(coef))
        coef = coef * real_sign
        y = (X * coef).sum(axis=1)

        ml = BaggedIcareRanker(n_estimators=200, n_jobs=-1)
        ml.fit(X,y)
        avg_signs = ml.get_feature_importance().sort_values(by='feature')['average sign'].values
        found_signs = (avg_signs > 0).astype('int8') * 2 - 1
        assert np.mean(found_signs == real_sign) == 1.