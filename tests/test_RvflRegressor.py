#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from graforvfl import RvflRegressor

np.random.seed(42)


def test_RvflRegressor_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal",
                          trainer="MPI", seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert RvflRegressor.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert len(pred) == X.shape[0]
