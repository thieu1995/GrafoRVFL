
# GrafoRVFL (GRAdient Free Optimized Random Vector Functional Link)

---

[![GitHub release](https://img.shields.io/badge/release-1.1.0-yellow.svg)](https://github.com/thieu1995/GrafoRVFL/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/graforvfl) 
[![PyPI version](https://badge.fury.io/py/graforvfl.svg)](https://badge.fury.io/py/graforvfl)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graforvfl.svg)
![PyPI - Status](https://img.shields.io/pypi/status/graforvfl.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/graforvfl.svg)
[![Downloads](https://pepy.tech/badge/graforvfl)](https://pepy.tech/project/graforvfl)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/graforvfl.svg)
[![Documentation Status](https://readthedocs.org/projects/graforvfl/badge/?version=latest)](https://graforvfl.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/graforvfl.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/676088001.svg)](https://zenodo.org/doi/10.5281/zenodo.10251021)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


GrafoRVFL is an open-source library in Python that employs gradient-free optimization ((GA, PSO, WOA, TLO, DE, ...) to 
optimize Random Vector Functional Link Networks. It is entirely implemented based on Numpy and fully compatible 
with the interfaces of the Scikit-Learn library. With GrafoRVFL, you can fine-tune the hyper-parameters of network 
or optimize weights in the network using gradient-free optimizers.


* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: RvflRegressor, RvflClassifier, MhaTuneRvfl
* **Total Gradient Free based RVFL Regressor**: > 200 Models 
* **Total Gradient Free based RVFL Classifier**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://graforvfl.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics


# Citation Request 

Learn more about Random Vector Functional Link from [this paper](https://doi.org/10.1016/j.ins.2015.09.025)

Learn more about on how to use Gradient Free Optimization to fine-tune the hyper-parameter of RVFL networks from 
[this paper](https://doi.org/10.1109/TCSS.2022.3146974)

Learn more about on how to use Gradient Free Optimization to optimize the weights of RVFL netweorks from [this paper](https://doi.org/10.1109/SOCA.2018.00014)



Please include these citations if you plan to use this library:

```code

@software{nguyen_van_thieu_2023_10251022,
  author       = {Nguyen Van Thieu},
  title        = {GrafoRVFL: A Python Library for Maximizing Performance of Random Vector Functional Link Network with Gradient-Free Optimization},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {},
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

# Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/graforvfl):
```sh 
$ pip install graforvfl==1.0.0
```

* Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/GrafoRVFL.git
$ cd GrafoRVFL
$ python setup.py install
```

* In case, you want to install the development version from Github:
```sh 
$ pip install git+https://github.com/thieu1995/GrafoRVFL 
```

After installation, you can import GrafoRVFL as any other Python module:

```sh
$ python
>>> import graforvfl
>>> graforvfl.__version__
```

### Examples

Please check all use cases and examples in folder [examples](examples).

Current provided classes:

```python
from graforvfl import DataTransformer, Data
from graforvfl import RvflRegressor, RvflClassifier, MhaTuneRvfl
```

##### `DataTransformer` class

We provide many scaler classes that you can select and make a combination of transforming your data via 
DataTransformer class. For example: scale data by `Loge` and then `Sqrt` and then `MinMax`:

```python
from graforvfl import DataTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)
```

##### `Data` class

+ You can load your dataset into Data class
+ You can split dataset to train and test set
+ You can scale dataset without using DataTransformer class
+ You can scale labels using LabelEncoder

```python
from graforvfl import Data
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

data = Data(X, y, name="position_salaries")

#### Split dataset into train and test set
data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

#### Feature Scaling
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
data.y_test = scaler_y.transform(data.y_test)
```

##### Network class

```python
from graforvfl import RvflRegressor, RvflClassifier, MhaTuneRvfl
from mealpy import IntegerVar, StringVar

## 1. Use standard RVFL model for regression problem
model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", trainer="OLS", alpha=0.5)

## 2. Use standard RVFL model for classification problem 
model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="OLS", alpha=0.5)


## 3. Use Gradient Free Optimization to fine-tune the hyper-parameter of RVFL network for regression problem
# Design the boundary (parameters)
my_bounds = [
    IntegerVar(lb=2, ub=1000, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu",
                          "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act_name"),
    StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                          "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"), name="weight_initializer")
]
opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
model = MhaTuneRvfl(problem_type="regression", bounds=my_bounds, cv=3, scoring="MSE",
                      optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
```

##### Supported functions in model object

```python
from graforvfl import RvflRegressor, Data 

data = Data()       # Assumption that you have provided this object like above

model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", trainer="OLS", alpha=0.5)

## Train the model
model.fit(data.X_train, data.y_train)

## Predicting a new result
y_pred = model.predict(data.X_test)

## Calculate metrics using score or scores functions.
print(model.score(data.X_test, data.y_test, method="MAE"))
print(model.scores(data.X_test, data.y_test, list_methods=["MAPE", "NNSE", "KGE", "MASE", "R2", "R", "R2S"]))

## Calculate metrics using evaluate function
print(model.evaluate(data.y_test, y_pred, list_metrics=("MSE", "RMSE", "MAPE", "NSE")))

## Save performance metrics to csv file
model.save_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv")

## Save training loss to csv file
model.save_loss_train(save_path="history", filename="loss.csv")

## Save predicted label
model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

## Save model
model.save_model(save_path="history", filename="traditional_mlp.pkl")

## Load model 
trained_model = RvflRegressor.load_model(load_path="history", filename="traditional_mlp.pkl")
```

# Support (questions, problems)

### Official Links 

* Official source code repo: https://github.com/thieu1995/GrafoRVFL
* Official document: https://graforvfl.readthedocs.io/
* Download releases: https://pypi.org/project/graforvfl/
* Issue tracker: https://github.com/thieu1995/GrafoRVFL/issues
* Notable changes log: https://github.com/thieu1995/GrafoRVFL/blob/main/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/IntelELM
    * https://github.com/thieu1995/reflame
    * https://github.com/thieu1995/MetaPerceptron
    * https://github.com/aiir-team
