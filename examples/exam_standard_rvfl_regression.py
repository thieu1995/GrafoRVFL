#!/usr/bin/env python
# Created by "Thieu" at 10:12, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from graforvfl import Data, RvflRegressor
from sklearn.datasets import load_diabetes


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

## Create network
model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", reg_alpha=0.5, seed=42)

## Train the network
model.fit(data.X_train, data.y_train)

## Test the network
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["R", "NSE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R", "NSE", "MAPE", "NNSE"]))
