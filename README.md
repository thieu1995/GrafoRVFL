
# GrafoRVFL (GRAdient Free Optimized Random Vector Functional Link)

---

[![GitHub release](https://img.shields.io/badge/release-2.2.0-yellow.svg)](https://github.com/thieu1995/GrafoRVFL/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/graforvfl) 
[![PyPI version](https://badge.fury.io/py/graforvfl.svg)](https://badge.fury.io/py/graforvfl)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graforvfl.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/graforvfl.svg)
[![Downloads](https://pepy.tech/badge/graforvfl)](https://pepy.tech/project/graforvfl)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yml/badge.svg)](https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yml)
[![Documentation Status](https://readthedocs.org/projects/graforvfl/badge/?version=latest)](https://graforvfl.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10258280.svg)](https://doi.org/10.5281/zenodo.10258280)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## 📑 Overview

**GrafoRVFL** is an open-source Python library designed to optimize Random Vector Functional Link (RVFL) networks using 
various **gradient-free metaheuristic algorithms** such as GA, PSO, WOA, TLO, DE, etc. It is fully implemented in 
**NumPy** and seamlessly integrates with the **Scikit-Learn** interface, making it easy to plug into standard 
ML workflows. GrafoRVFL enables hyperparameter tuning for RVFL networks without relying on gradient-based methods.


## ✨ Features

- ✅ Free software under **GNU GPL v3**
- 📘 Full documentation: [https://graforvfl.readthedocs.io](https://graforvfl.readthedocs.io)
- 🧠 Estimators:
  - `RvflRegressor`
  - `RvflClassifier`
  - `GfoRvflCV`
  - `GfoRvflTuner`
  - `GfoRvflComparator`
- 🐍 Python compatibility: `>= 3.8`
- 🧩 Dependencies:
  - `numpy`, `scipy`, `scikit-learn`, `pandas`, `mealpy`, `permetrics`, `matplotlib`


## 📖 Citation Request 

Please include these citations if you plan to use this library:

```bibtex
@software{nguyen_van_thieu_2023_10258280,
  author       = {Nguyen Van Thieu},
  title        = {GrafoRVFL: A Gradient-Free Optimization Framework for Boosting Random Vector Functional Link Network},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10258280},
  url          = {https://github.com/thieu1995/GrafoRVFL}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@inproceedings{nguyen2019building,
  title={Building resource auto-scaler with functional-link neural network and adaptive bacterial foraging optimization},
  author={Nguyen, Thieu and Nguyen, Binh Minh and Nguyen, Giang},
  booktitle={International Conference on Theory and Applications of Models of Computation},
  pages={501--517},
  year={2019},
  organization={Springer}
}

@inproceedings{nguyen2018resource,
  title={A resource usage prediction system using functional-link and genetic algorithm neural network for multivariate cloud metrics},
  author={Nguyen, Thieu and Tran, Nhuan and Nguyen, Binh Minh and Nguyen, Giang},
  booktitle={2018 IEEE 11th conference on service-oriented computing and applications (SOCA)},
  pages={49--56},
  year={2018},
  organization={IEEE},
  doi={10.1109/SOCA.2018.00014}
}
```

* Learn more about Random Vector Functional Link from [this paper](https://doi.org/10.1016/j.ins.2015.09.025)

* Learn more about on how to use Gradient Free Optimization to fine-tune the hyper-parameter of RVFL networks from 
[this paper](https://doi.org/10.1016/j.neucom.2018.07.080)


## 🔧 Installation

Install the latest version from PyPI:

```bash
$ pip install graforvfl
```

Verify installation:

```bash
$ python
>>> import graforvfl
>>> graforvfl.__version__
```

## 🧪 Example Usage

Below is a simple example code of how to use Gradient Free Optimization to tune hyper-parameter of RVFL network.

```python
from sklearn.datasets import load_breast_cancer
from graforvfl import Data, GfoRvflCV, StringVar, IntegerVar, FloatVar


## Load data object
X, y = load_breast_cancer(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

# Design the boundary (parameters)
my_bounds = [
    IntegerVar(lb=3, ub=50, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu",
                          "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
                          "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
                          "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink",
                          "softmin", "softmax", "log_softmax"), name="act_name"),
    StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform",
                          "glorot_normal", "lecun_uniform", "lecun_normal", "random_uniform",
                          "random_normal"), name="weight_initializer"),
    FloatVar(lb=0, ub=10., name="reg_alpha"),
]

model = GfoRvflCV(problem_type="classification", bounds=my_bounds,
                  optim="OriginalWOA", optim_params={"name": "WOA", "epoch": 10, "pop_size": 20},
                  scoring="AS", cv=3, seed=42, verbose=True)
model.fit(data.X_train, data.y_train)
print(model.best_params)
print(model.best_estimator)
print(model.best_estimator.scores(data.X_test, data.y_test, list_metrics=("PS", "RS", "NPV", "F1S", "F2S")))
```

👉 The more complicated cases in the folder: [examples](/examples). You can also read the [documentation](https://graforvfl.readthedocs.io/) 
for more detailed installation instructions, explanations, and examples.


## 📎 Official channels 

* 🔗 [Official source code repository](https://github.com/thieu1995/GrafoRVFL)
* 📘 [Official document](https://graforvfl.readthedocs.io/)
* 📦 [Download releases](https://pypi.org/project/graforvfl/) 
* 🐞 [Issue tracker](https://github.com/thieu1995/GrafoRVFL/issues) 
* 📝 [Notable changes log](/ChangeLog.md)
* 💬 [Official discussion group](https://t.me/+fRVCJGuGJg1mNDg1)

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=GrafoRVFL_QUESTIONS) @ 2023
