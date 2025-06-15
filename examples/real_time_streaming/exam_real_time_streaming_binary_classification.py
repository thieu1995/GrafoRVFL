#!/usr/bin/env python
# Created by "Thieu" at 23:10, 15/06/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from permetrics import ClassificationMetric
from graforvfl import RvflClassifier, DataTransformer



def fully_online_learning():
    """
    Fully online learning with RvflClassifier on Breast Cancer dataset.
    This simulates real-time streaming of data, training the model incrementally.
    """

    # Step 1: Load dataset and encode
    iris = load_breast_cancer()
    X_all = iris.data
    y_all = iris.target.reshape(-1, 1)
    classes = np.unique(y_all)

    # Step 2: Split train and test sets
    X_train_stream, X_test, y_train_stream, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Step 3: Scale features
    dt = DataTransformer().fit(X_train_stream)
    X_train_stream = dt.transform(X_train_stream)
    X_test = dt.transform(X_test)

    # Step 4: Define model
    model = RvflClassifier(size_hidden=15, act_name="elu", weight_initializer="random_uniform", reg_alpha=0, seed=42)

    # Step 5: Streaming loop (real-time input simulation)
    print("Start streaming training...")
    for i in range(X_train_stream.shape[0]):
        x_i = X_train_stream[i].reshape(1, -1)
        y_i = y_train_stream[i].reshape(1, -1)

        # === Streaming: partial_fit for each sample ===
        model.partial_fit(x_i, y_i, classes=classes)

        # === Evaluate one after 10 samples ===
        if (i + 1) % 10 == 0:
            y_pred = model.predict(X_test)
            cls = ClassificationMetric(y_true=y_test, y_pred=y_pred)
            res = cls.get_metrics_by_list_names(['AS', 'PS', 'RS', 'F1S'])
            print(f"[Step {i + 1}], {res}")

        # === Sleep for delay simulation (real-time) ===
        time.sleep(0.2)  # Delay to simulate real-time streaming after get new data

    print("Done streaming.")


def hybrid_learning():
    """
    Hybrid learning with RvflClassifier on Breast Cancer dataset.
    This simulates a scenario where the model is trained in batches and then updated with new data.
    """

    # Step 1: Load dataset and encode
    iris = load_breast_cancer()
    X_all = iris.data
    y_all = iris.target.reshape(-1, 1)

    # Step 2: Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Step 3: Scale features
    dt = DataTransformer().fit(X_train)
    X_train = dt.transform(X_train)
    X_test = dt.transform(X_test)

    # Step 3: Define model
    model = RvflClassifier(size_hidden=15, act_name="tanh", weight_initializer="random_normal", reg_alpha=0.001, seed=42)

    # Step 4: Batch training
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
        cls = ClassificationMetric(y_true=y_test, y_pred=y_pred)
        res = cls.get_metrics_by_list_names(['AS', 'PS', 'RS', 'F1S'])
        print(f"[Step {i + 1}], {res}")

        time.sleep(0.2)  # Delay to simulate real-time streaming after get new data

    print("Done hybrid learning.")


if __name__ == "__main__":
    print("=== Fully Online Learning ===")
    fully_online_learning()

    print("\n=== Hybrid Learning ===")
    hybrid_learning()
