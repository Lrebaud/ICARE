import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.utils.estimator_checks import check_estimator
from sksurv import datasets
from sksurv.preprocessing import OneHotEncoder

from icare.metrics import *
from icare.survival import IcareSurv, BaggedIcareSurv


def generate_random_param_set():
    return {
        'rho': np.random.uniform(0.01, 1.),
        'correlation_method': np.random.choice(['spearman', 'pearson']),
        'sign_method': ','.join(np.random.choice(['harrell', 'tAUC', 'uno'],
                                        np.random.randint(1, 4),
                                        replace=False).tolist()),
        'cmin': np.random.uniform(0.5, 1.),
        'max_features': np.random.uniform(0.5, 1.),
        'random_state': None,
    }



def generate_random_param_set_bagged():
    return {
        'n_estimators': np.random.randint(1, 10),
        'parameters_sets': [generate_random_param_set() for x in range(np.random.randint(1, 3))],
        'aggregation_method': np.random.choice(['mean', 'median']),
        'n_jobs': np.random.randint(1, 5),
        'random_state': None,
    }


def test_feature_group():
    X, y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = IcareSurv()
    ml.fit(X, y)

    feature_groups = None
    ml.fit(X, y, feature_groups=feature_groups)

    ml = IcareSurv(features_groups_to_use=[0, 1])
    feature_groups = np.ones(int(X.shape[1] / 2))
    feature_groups = np.concatenate([feature_groups, np.zeros(X.shape[1] - len(feature_groups))])
    ml.fit(X, y, feature_groups=feature_groups)
    harrell_cindex_scorer(ml, X, y)


def test_scikit_simple():
    check_estimator(IcareSurv())


def test_scikit_bagged():
    check_estimator(BaggedIcareSurv())


def test_no_censoring():
    X, y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = IcareSurv()
    ml.fit(X, y)

    y = [x[1] for x in y]
    ml.fit(X, y)

def test_feature_sign():
    X, y = datasets.load_veterans_lung_cancer()
    X = OneHotEncoder().fit_transform(X)

    ml = BaggedIcareSurv(n_estimators=50,
                         n_jobs=-1,
                         parameters_sets=[{'max_features':1,
                                           'rho' : None,
                                           'cmin': 0.5}])
    ml.fit(X, y)
    fs = ml.average_feature_signs_
    fs_kar = fs[fs['feature']=='Karnofsky_score']['average sign'].values[0]
    fs_celltype = fs[fs['feature']=='Celltype=smallcell']['average sign'].values[0]
    assert fs_kar < 0
    assert fs_celltype > 0


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


def test_hecktor():
    df = pd.read_csv('data/df_train.csv', index_col='PatientID')
    features = list(set(df.columns.tolist()) - set(['Relapse', 'RFS', 'Task 1', 'Task 2', 'CenterID']))
    features = [x for x in features if 'lesions_merged' not in x and 'lymphnodes_merged' not in x]
    extra_features = ['Gender',
                      'Age',
                      'Weight',
                      'Tobacco',
                      'Alcohol',
                      'Performance status',
                      'HPV status (0=-, 1=+)',
                      'Surgery',
                      'Chemotherapy', 'nb_lesions', 'nb_lymphnodes', 'whole_body_scan']
    features_groups = np.unique([x.split('_shape_')[0].split('_PT_')[0].split('_CT_')[0] for x in features])
    features_groups = list(set(features_groups) - set(extra_features))
    features_groups = [x + '_' for x in features_groups]
    features_groups.append('extra_features')

    groups_to_use = ['everything_mergedshell4mm_',
                     'everything_merged_',
                     'everything_merged4_',
                     'everything_mergeddilat2mm_', ]

    params = {'max_features': 20 / df.shape[1],
              'rho': 0.66,
              'cmin': 0.53,
              'mandatory_features': extra_features,
              'features_groups_to_use': [features_groups.index(x) for x in groups_to_use]
              }

    y = Surv.from_arrays(event=df['Relapse'].values,
                         time=df['RFS'].values)
    X = df[features]
    mask_keep = (X.isna().sum(axis=1) < 1000).values
    X, y = X.iloc[mask_keep], y[mask_keep]

    features_groups_id = []
    for f in X.columns:
        if f in extra_features:
            features_groups_id.append(features_groups.index('extra_features'))
        else:
            group = f.split('_shape_')[0].split('_PT_')[0].split('_CT_')[0] + '_'
            features_groups_id.append(features_groups.index(group))

    def worker_cv_paral(model, X, y, feature_group, train_index, test_index):
        try:
            model.fit(X.iloc[train_index], y[train_index], feature_groups=feature_group)
            pred = model.predict(X.iloc[test_index])
            return harrell_cindex(y[test_index], pred)
        except:
            return np.nan

    def cv_paral(model, X, y, feature_group, n_folds):
        cv = ShuffleSplit(n_splits=n_folds, test_size=.5)
        scores = Parallel(n_jobs=-1)(delayed(worker_cv_paral)(model, X, y, feature_group, train_index, test_index)
                                     for train_index, test_index in cv.split(X))

        return np.nanmean(scores)

    score = cv_paral(IcareSurv(**params), X, y, features_groups_id, n_folds=64)
    assert score > 0.65

    ml = BaggedIcareSurv(n_estimators=100,
                         parameters_sets=[params],
                         aggregation_method='median',
                         n_jobs=1,
                         random_state=None)
    score = cv_paral(ml, X, y, features_groups_id, n_folds=64)
    assert score > 0.67


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
