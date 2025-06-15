#!/usr/bin/env python
# Created by "Thieu" at 10:02, 15/06/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from time import sleep
from collections import deque
from graforvfl import RvflClassifier

# Step 1: Load dataset and encode
iris = load_iris()
X_all = iris.data
y_all = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_all_encoded = encoder.fit_transform(y_all)

# Step 2: Split train and test sets
X_train_stream, X_test, y_train_stream, y_test = train_test_split(X_all, y_all_encoded, test_size=0.3, random_state=42)

# Step 3: Define model
model = RvflClassifier(size_hidden=20, act_name="relu", weight_initializer="random_normal", reg_alpha=0.001, seed=42)

# Step 4: Streaming loop (real-time input simulation)
print("Start streaming training...")
stream_window = deque()

for i in range(X_train_stream.shape[0]):
    x_i = X_train_stream[i].reshape(1, -1)
    y_i = y_train_stream[i].reshape(1, -1)

    # === Streaming: partial_fit for each sample ===
    model.partial_fit(x_i, y_i)

    # === Evaluate one after 10 samples ===
    if (i + 1) % 10 == 0:
        y_pred = model.predict_proba(X_test)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        print(f"[Step {i+1}] Accuracy: {acc:.4f}")

    # === Sleep for delay simulation (real-time) ===
    sleep(0.2)  # Delay to simulate real-time streaming after get new data

print("Done streaming.")
