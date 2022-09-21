[![PyPI version](https://badge.fury.io/py/icare.svg)](https://badge.fury.io/py/icare)

<h1>
  <img align="right" height="80" src="https://raw.githubusercontent.com/Lrebaud/ICARE/main/img/logo.png">
   <br> ICARE
</h1>

> Rebaud, L.\*, Escobar, T.\*, Khalid, F., Girum, K., Buvat, I.: Head and Neck Tumor and Lymph Node 
Segmentation and Outcome Prediction from 18F-FDG PET/CT Images: Simplicity is All You Need. 
In: Lecture Notes in Computer Science (LNCS) Challenges (in press)


This package contains the Individual Coefficient Approximation for Risk Estimation
(ICARE) survival model.
For ensembling strategies, it also includes a dedicated bagging aggregator.


## Description

The Individual Coefficient Approximation for Risk Estimation (ICARE) survival model
uses a minimal learning strategy to reduce to risk of overfitting on the often 
noisy and censored survival data.
To do so:
 * drop highly correlated features
 * for each feature:
   * evaluate feature sign using an univariate approach
   * normalize the feature
   * multiply the feature by its sign
 * the prediction is computed as the mean of all signed features

This makes the model more robust to overfitting. It also makes it
resilient to the curse of dimensionality. We hypothesize that it is 
better to have too many features than too few for this model.
This algorithm is implemented in the `IcareSurv` estimator in this
package.

To improve the performance, this model can be bagged. The package
provides `BaggedIcareSurv` estimator that does the ensembling of 
multiple `IcareSurv` estimators. 

The models make predictions that are **anti-concordants** with the target. 
For instance, if the target is the survival in days since baseline, the
prediction corresponds to the **ranking risk** of death.


## Getting Started

### Dependencies

* Python 3.6 or later
* scikit-survival

### Installing

* Via PyPI:
```shell
pip install icare
```
or via GitHub
```shell
pip install git+https://github.com/Lrebaud/ICARE.git
```

### Documentation
Coming soon.

### Utilisation

The model is used as any other scikit-learn estimator:
```python
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

It can be used with all scikit-learn functions:
```python
from sklearn.model_selection import cross_val_score, ShuffleSplit

model = IcareSurv()
score = cross_val_score(model, X, y,
                        cv=ShuffleSplit(n_splits=20, test_size=.25),
                        n_jobs=-1,
                        scoring=harrell_cindex_scorer).mean()

```

If you are working with a censored target, you need to create a
structured array containing both the time and the censoring:

```python
from sksurv.util import Surv
y = Surv.from_arrays(event=np.array(event_happened).astype('bool'),
                     time=time)
```


## Author

Louis Rebaud: [louis.rebaud@gmail.com](mailto:louis.rebaud@gmail.com)

## Version History

* 0.0.1
    * Initial Release

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details

## Acknowledgements

This package was created as a part of the HEad and neCK TumOR segmentation and outcome prediction in PET/CT images challenge, 3<sup>rd</sup> edition (HECKTOR 2022) and presented by Louis Rebaud and Thibault Escobar at the 25<sup>th</sup> international conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) congress in Singapore.

## Citation

If you use this package for your research, please cite the following paper:

```blockquote
Rebaud, L.*, Escobar, T.*, Khalid, F., Girum, K., Buvat, I.: Head and Neck Tumor and Lymph Node 
Segmentation and Outcome Prediction from 18F-FDG PET/CT Images: Simplicity is All You Need. 
In: Lecture Notes in Computer Science (LNCS) Challenges (in press)
```
