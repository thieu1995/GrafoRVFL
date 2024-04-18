#!/usr/bin/env python
# Created by "Thieu" at 19:31, 04/12/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from mealpy import StringVar, IntegerVar
from graforvfl import Data, GfoRvflTuner


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))

# Design the boundary (parameters)
my_bounds = [
    IntegerVar(lb=2, ub=1000, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu",
                          "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act_name"),
    StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                          "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"), name="weight_initializer")
]

opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
model = GfoRvflTuner(problem_type="regression", bounds=my_bounds, cv=3, scoring="MSE",
                      optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
model.fit(data.X_train, data.y_train)
print(model.best_params)
print(model.best_estimator)
print(model.best_estimator.scores(data.X_test, data.y_test, list_methods=("MSE", "RMSE", "MAPE", "NSE", "R2", "KGE")))
