# ICARE

This package contains the Individual Coefficient Approximation for Risk Estimation
(ICARE) survival model. It also includes a dedicated bagging aggregator.

## Description

The Individual Coefficient Approximation for Risk Estimation (ICARE) survival model
use a minimal learning strategy to reduce to risk of overfitting on the often 
noisy and censored survival data.


This makes the model 


## Getting Started

### Dependencies

* Python 3.6 or later
* scikit-survival

### Installing

* Via PyPI:
```
$ pip install icare
```
or via GitHub
```
$ pip install git+https://github.com/Lrebaud/ICARE.git
```

### Utilisation

The model is used like any other scikit-learn estimator:
```
from sksurv import datasets
from sksurv.preprocessing import OneHotEncoder
from icare.metrics import harrell_cindex
from icare.survival import IcareSurv, BaggedIcareSurv


X, y = datasets.load_veterans_lung_cancer()
X = OneHotEncoder().fit_transform(X)

model = IcareSurv()
model.fit(X, y)
pred = model.predict(X)
print(pred[:5])
harrell_cindex(y, pred)
```

It can be used with all scikit-learn function:
```
from sklearn.model_selection import cross_val_score, ShuffleSplit

model = IcareSurv()
score = cross_val_score(model, X, y,
                        cv=ShuffleSplit(n_splits=20, test_size=.25),
                        n_jobs=-1,
                        scoring=harrell_cindex_scorer).mean()

```

If you are working with a censored target, you need to create a
structured array containing both the time and the censoring:

```
from sksurv.util import Surv
y = Surv.from_arrays(event=np.array(event_happened).astype('bool'),
                     time=time)
```


## Authors

Louis Rebaud : louis.rebaud[at]gmail.com

## Version History

* 0.0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
