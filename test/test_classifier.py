from icare.metrics import *
from icare.classification import IcareClassifier, BaggedIcareClassifier
from sklearn import datasets
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import f1_score


dt_classif_list = [
    [sm.datasets.anes96, 'vote'],
    [sm.datasets.ccard, 'OWNRENT'],
]
dt_dict = []
for dt, target in dt_classif_list:
    name = dt.__name__.split('.')[-1]
    data = dt.load_pandas().data
    y = data[target]
    X = data.drop(columns=target)
    dt_dict.append({
        'data': X.values,
        'target': y.values
    })
dt_dict += [datasets.load_iris(), datasets.load_breast_cancer(), datasets.load_wine()]


def concordant_cindex_scorer(ml, X, y):
    return 1. - harrell_cindex_scorer(ml, X, y)


def generate_random_param_set():
    return {
        'rho': np.random.uniform(0.01, 1.),
        'correlation_method': np.random.choice(['spearman', 'pearson']),
        'sign_method': ','.join(np.random.choice(['pearson', 'spearman'],
                                                 np.random.randint(1, 3),
                                                 replace=False).tolist()),
        'cmin': np.random.uniform(0., 1),
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
            ml = IcareClassifier(**params)
            ml.fit(X, y)
            pred = ml.predict(X)
            score_f1 = f1_score(y, pred, average='weighted')
            score_cindex = concordant_cindex_scorer(ml, X, y)
            if score_f1 > 0.5 and score_cindex > 0.5:
                break
        assert score_f1 > 0.5 and score_cindex > 0.5


def test_scikit_datasets_bagged():
    for data in dt_dict:
        X,y = data['data'], data['target']
        for j in range(X.shape[1]):
            if not np.issubdtype(X[:,j].dtype, np.number):
                X[:,j] = preprocessing.LabelEncoder().fit_transform(X[:,j])


        for _ in range(10):
            params = generate_random_param_set_bagged()
            ml = BaggedIcareClassifier(**params)
            ml.fit(X, y)
            pred = ml.predict(X)
            score_f1 = f1_score(y, pred, average='weighted')
            score_cindex = concordant_cindex_scorer(ml, X, y)
            if score_f1 > 0.5 and score_cindex > 0.5:
                break
        assert score_f1 > 0.5 and score_cindex > 0.5



