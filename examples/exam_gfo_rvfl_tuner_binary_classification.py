#!/usr/bin/env python
# Created by "Thieu" at 11:10, 03/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_breast_cancer
from graforvfl import Data, GfoRvflTuner, StringVar, IntegerVar, FloatVar


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
    IntegerVar(lb=2, ub=40, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax"), name="act_name"),
    StringVar(valid_sets=("orthogonal",), name="weight_initializer"),
    FloatVar(lb=0, ub=10., name="reg_alpha"),
]

optim_param_grid = {
    "epoch": [10, 20, 30],
    "pop_size": [15, 20, 25, 30],
}

tuner = GfoRvflTuner(problem_type="classification", bounds=my_bounds,
                     optim="OriginalWOA", optim_param_grid=optim_param_grid,
                     scoring="AS", cv=3,
                     search_type="random", n_iter=5, seed=42, verbose=True,
                     mode="single", n_workers=None, termination=None)
tuner.fit(data.X_train, data.y_train)
print(tuner.best_score)
print(tuner.best_optim_params)
print(tuner.best_searcher)
print(tuner.best_searcher.scores(data.X_test, data.y_test, list_metrics=("PS", "RS", "NPV", "F1S", "F2S")))

print(tuner.best_searcher.best_params)
print(tuner.best_searcher.best_estimator)
print(tuner.best_searcher.best_estimator.scores(data.X_test, data.y_test, list_metrics=("PS", "RS", "NPV", "F1S", "F2S")))
