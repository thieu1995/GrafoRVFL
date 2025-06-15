#!/usr/bin/env python
# Created by "Thieu" at 17:50, 15/06/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from permetrics import RegressionMetric
from graforvfl import RvflRegressor, DataTransformer


def fully_online_learning():
    # 1. Load and preprocess the dataset
    X, y = load_diabetes(return_X_y=True)

    # 2: Split train and test sets
    X_train_stream, X_test, y_train_stream, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3: Scale data
    dt_x = DataTransformer(scaling_methods=('standard',),).fit(X_train_stream)
    X_train_stream = dt_x.transform(X_train_stream)
    X_test = dt_x.transform(X_test)

    dt_y = DataTransformer(scaling_methods=('minmax',),).fit(y_train_stream)
    y_train_stream = dt_y.transform(y_train_stream)
    y_test = dt_y.transform(y_test)

    # 4. Define the model
    model = RvflRegressor(size_hidden=20, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.01, seed=42)

    # 5: Streaming loop (real-time input simulation)
    print("Start streaming training...")
    for i in range(X_train_stream.shape[0]):
        x_i = X_train_stream[i].reshape(1, -1)
        y_i = y_train_stream[i].reshape(1, -1)

        # === Streaming: partial_fit for each sample ===
        model.partial_fit(x_i, y_i)

        # === Evaluate one after 10 samples ===
        if (i + 1) % 10 == 0:
            y_pred = model.predict(X_test)
            reg = RegressionMetric(y_true=y_test, y_pred=y_pred)
            res = reg.get_metrics_by_list_names(['RMSE', 'MAPE', 'NSE', 'R', 'KGE'])
            print(f"[Step {i + 1}], {res}")

        # === Sleep for delay simulation (real-time) ===
        time.sleep(0.2)  # Delay to simulate real-time streaming after get new data

    print("Done streaming.")


def hybrid_learning():
    # Step 1: Load dataset and encode
    db = load_diabetes()
    X_all = db.data
    y_all = db.target.reshape(-1, 1)

    # Step 2: Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Step 3: Scale data
    dt_x = DataTransformer(scaling_methods=('standard',),).fit(X_train)
    X_train = dt_x.transform(X_train)
    X_test = dt_x.transform(X_test)

    dt_y = DataTransformer(scaling_methods=('minmax',),).fit(y_train)
    y_train = dt_y.transform(y_train)
    y_test = dt_y.transform(y_test)

    # Step 4: Define model
    model = RvflRegressor(size_hidden=20, act_name="tanh", weight_initializer="random_normal", reg_alpha=0.001, seed=42)

    # Step 5: Batch training
    print("Start batch training...")
    model.fit(X_train, y_train)

    # Step 5: Simulate real-time streaming with new data
    for i in range(40):
        x_new = X_test[i].reshape(1, -1)
        y_new = y_test[i].reshape(1, -1)

        # Update model with new data
        model.partial_fit(x_new, y_new)

        # Evaluate after each new sample
        y_pred = model.predict(X_test)
        reg = RegressionMetric(y_true=y_test, y_pred=y_pred)
        res = reg.get_metrics_by_list_names(['RMSE', 'MAPE', 'NSE', 'R', 'KGE'])
        print(f"[Step {i + 1}], {res}")

        time.sleep(0.2)  # Delay to simulate real-time streaming after get new data

    print("Done hybrid learning.")


if __name__ == "__main__":
    print("Running fully online learning...")
    fully_online_learning()

    print("\nRunning hybrid learning...")
    hybrid_learning()
