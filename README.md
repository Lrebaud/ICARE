[![PyPI version](https://badge.fury.io/py/icare.svg)](https://badge.fury.io/py/icare)
[![Downloads](https://pepy.tech/badge/icare/month)](https://pepy.tech/project/icare)
[![Python package](https://github.com/Lrebaud/ICARE/actions/workflows/python-package.yml/badge.svg)](https://github.com/Lrebaud/ICARE/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/github/Lrebaud/ICARE/branch/main/graph/badge.svg?token=W4D5C373NR)](https://codecov.io/github/Lrebaud/ICARE)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6--3.10-blue)](https://www.python.org/downloads/release/python-360/)

<h1>
  <img align="right" height="80" src="https://raw.githubusercontent.com/Lrebaud/ICARE/main/img/logo.png">
   <br> ICARE
</h1>

> Rebaud, L., Escobar, T., Khalid, F., Girum, K., Buvat, I. (2023).
Simplicity Is All You Need: Out-of-the-Box nnUNet Followed by Binary-Weighted Radiomic Model for Segmentation and Outcome Prediction in Head and Neck PET/CT.
In: Andrearczyk, V., Oreiller, V., Hatt, M., Depeursinge, A. (eds) Head and Neck Tumor Segmentation and Outcome Prediction.
HECKTOR 2022. Lecture Notes in Computer Science, vol 13626. Springer, Cham.

This package contains the Individual Coefficient Approximation for Risk Estimation
(ICARE) survival model. For ensembling strategies, it also includes a dedicated bagging aggregator.

This model was the winning solution for the [MICCAI 2022](https://conferences.miccai.org/2022/en/) challenge: HEad and neCK TumOR ([HECKTOR](https://hecktor.grand-challenge.org/)) for the outcome prediction task from PET/CT.

## Description

The Individual Coefficient Approximation for Risk Estimation (ICARE) model
uses a minimal learning strategy to reduce to risk of overfitting.
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
This algorithm is implemented in the following classes:
 * `IcareClassifier` estimator for classification tasks
 * `IcareRanker` estimator for ranking tasks (not calibrated regression)
 * `IcareSurvival` estimator for survival prediction tasks package.

To improve the performance, this package also provides bagged versions
of these estimators:
 * `BaggedIcareClassifier` estimator for classification tasks
 * `BaggedIcareRanker` estimator for ranking tasks (not calibrated regression)
 * `BaggedIcareSurvival` estimator for survival prediction tasks package.


The survival models (`IcareSurvival` and `BaggedIcareSurvival`) predict
a risk score. Therefore, there predictions are **anti-concordants** with
the target.

On the other hand, the ranking models (`IcareRanker` and `BaggedIcareRanker`)
try to correctly the samples according to the target, so there predictions
are **concordants** with the target.


## Getting Started

### Dependencies

* Python 3.6 or later
* pandas
* seaborn
* scikit-learn
* scikit-survival

### Installing

Via PyPI:
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

The model is used as any other scikit-learn estimator.

You can find detailed notebooks in the `notebooks` folder
of this repository showing how to use the package for each type of datasets.

## Author

Louis Rebaud: [louis.rebaud@gmail.com](mailto:louis.rebaud@gmail.com)

## Version History

* 0.1.0
    * Add classification and ranking support

* 0.0.1
    * Initial Release

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details

## Acknowledgements

This package was created as a part of the HEad and neCK TumOR segmentation and outcome prediction in PET/CT images challenge, 3<sup>rd</sup> edition (HECKTOR 2022) and presented by Louis Rebaud and Thibault Escobar at the 25<sup>th</sup> international conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) congress in Singapore.

## Citation

If you use this package for your research, please cite the following [paper](http://dx.doi.org/10.1007/978-3-031-27420-6_13):

```blockquote
Rebaud, L., Escobar, T., Khalid, F., Girum, K., Buvat, I. (2023). Simplicity Is All You Need: Out-of-the-Box nnUNet Followed by Binary-Weighted Radiomic Model for Segmentation and Outcome Prediction in Head and Neck PET/CT. In: Andrearczyk, V., Oreiller, V., Hatt, M., Depeursinge, A. (eds) Head and Neck Tumor Segmentation and Outcome Prediction. HECKTOR 2022. Lecture Notes in Computer Science, vol 13626. Springer, Cham.
```
