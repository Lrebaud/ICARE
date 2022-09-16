from sksurv import datasets
from icare.survival import IcareSurv, BaggedIcareSurv
from sksurv.preprocessing import OneHotEncoder
import numpy as np
from icare.metrics import harrell_cindex_scorer
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_score, ShuffleSplit


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


def test_scikit_simple():
    check_estimator(IcareSurv())


def test_scikit_bagged():
    check_estimator(BaggedIcareSurv())



def test_sksurv_datasets_simple():
    for X, y in [datasets.load_veterans_lung_cancer(),
                 datasets.load_whas500(),
                 datasets.load_gbsg2(),
                 # datasets.load_flchain(),
                 datasets.load_breast_cancer(),
                 datasets.load_aids()]:
        X = OneHotEncoder().fit_transform(X)

        for _ in range(10):
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


def test_reproducible_simple():
    for _ in range(3):
        for X, y in [datasets.load_veterans_lung_cancer(),
                     datasets.load_whas500(),
                     datasets.load_gbsg2(),
                     # datasets.load_flchain(),
                     datasets.load_breast_cancer(),
                     datasets.load_aids()]:
            X = OneHotEncoder().fit_transform(X)

            rand_params = generate_random_param_set()
            rand_state = np.random.randint(0, 1000)
            all_scores = []

            for _ in range(10):
                rand_params['random_state'] = rand_state
                ml = IcareSurv(**rand_params)
                score = cross_val_score(ml, X, y,
                                        cv=ShuffleSplit(n_splits=4, test_size=.25, random_state=42),
                                        n_jobs=4,
                                        scoring=harrell_cindex_scorer).mean()

                all_scores.append(score.round(4))

            assert len(np.unique(all_scores)) == 1


def test_reproducible_bagged():
    for _ in range(3):
        for X, y in [datasets.load_veterans_lung_cancer(),
                     datasets.load_whas500(),
                     datasets.load_gbsg2(),
                     # datasets.load_flchain(),
                     datasets.load_breast_cancer(),
                     datasets.load_aids()]:
            X = OneHotEncoder().fit_transform(X)

            rand_params = generate_random_param_set_bagged()
            rand_state = np.random.randint(0, 1000)
            all_scores = []

            for _ in range(10):
                rand_params['random_state'] = rand_state
                ml = BaggedIcareSurv(**rand_params)
                score = cross_val_score(ml, X, y,
                                        cv=ShuffleSplit(n_splits=1, test_size=.25, random_state=42),
                                        n_jobs=1,
                                        scoring=harrell_cindex_scorer).mean()

                all_scores.append(score.round(4))
            assert len(np.unique(all_scores)) == 1
