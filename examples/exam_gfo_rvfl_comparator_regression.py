#!/usr/bin/env python
# Created by "Thieu" at 17:00, 03/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from graforvfl import Data, StringVar, IntegerVar, FloatVar, GfoRvflComparator


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
    IntegerVar(lb=2, ub=40, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax"), name="act_name"),
    StringVar(valid_sets=("orthogonal",), name="weight_initializer"),
    FloatVar(lb=0, ub=10., name="reg_alpha"),
]

# Danh sách thuật toán cần so sánh
optim_list = ["OriginalPSO", "BaseGA", "OriginalWOA"]
# Mỗi thuật toán có tham số khác nhau
optim_params_list = [
    {"epoch": 10, "pop_size": 30, "c1": 1.5, "c2": 1.5, "w": 0.7, "name": "PSO"},  # PSO
    {"epoch": 10, "pop_size": 30, "pc": 0.8, "pm": 0.2, "name": "GA"},  # GA
    {"epoch": 10, "pop_size": 30, "name": "WOA"}  # WOA
]

# Khởi tạo comparator
comparator = GfoRvflComparator(problem_type="regression", bounds=my_bounds,
                               optim_list=optim_list, optim_params_list=optim_params_list,
                               scoring="MSE", cv=3, seed=42, verbose=True,
                               mode='single', n_workers=None, termination=None)
# Chạy so sánh
_, _, results = comparator.run(data.X_train, data.y_train, data.X_test, data.y_test, n_trials=3,
               list_metrics=("MSE", "RMSE", "MAPE", "NSE", "R2", "KGE"),
               save_results=True, save_models=True, path_save="history")

# Xem kết quả
print(results)

# Plotting
comparator.plot_loss_train_per_trial(path_read="history", path_save="history")
comparator.plot_loss_train_average(path_read="history", path_save="history")
comparator.plot_metric_boxplot(path_read="history", path_save="history")
comparator.plot_average_runtime(path_read="history", path_save="history")
