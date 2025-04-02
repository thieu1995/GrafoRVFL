#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from graforvfl import GfoRvflCV, IntegerVar, StringVar, FloatVar

np.random.seed(42)


def test_GfoRvflCV_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    # Design the boundary (parameters)
    my_bounds = [
        IntegerVar(lb=2, ub=1000, name="size_hidden"),
        StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu",
                              "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act_name"),
        StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                              "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"),
                  name="weight_initializer"),
        FloatVar(lb=0.0, ub=1.0, name="reg_alpha")
    ]

    model = GfoRvflCV(problem_type="regression", bounds=my_bounds,
                      optim="OriginalWOA", optim_params={"name": "WOA", "epoch": 5, "pop_size": 10},
                      scoring="MSE", cv=3, seed=42, verbose=True)
    model.fit(X, y)
    print(model.best_params)
    print(model.best_estimator)

    pred = model.predict(X)
    assert GfoRvflCV.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert len(pred) == X.shape[0]
