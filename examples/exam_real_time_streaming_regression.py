#!/usr/bin/env python
# Created by "Thieu" at 17:50, 15/06/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from graforvfl import RvflRegressor  # class bạn đã refactor

# 1. Load and preprocess the dataset
X, y = load_diabetes(return_X_y=True)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 2. Define the model
model = RvflRegressor(size_hidden=20, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.01, seed=42)

# 3. Streaming training
y_preds, y_trues = [], []
for i, (x_new, y_new) in enumerate(zip(X, y), start=1):
    x_new = x_new.reshape(1, -1)
    y_new = np.array([[y_new]])

    if i > 10:
        # Predict after 10 samples to have more meaningful output
        y_pred = model.predict(x_new)
        y_preds.append(y_pred.item())
        y_trues.append(y_new.item())

    # Train with partial_fit
    model.partial_fit(x_new, y_new)

    # In thông tin trạng thái
    print(f"[{i}] y_true: {y_new.item():.4f}", end="")
    if i > 10:
        print(f" | y_pred: {y_pred.item():.4f}")
    else:
        print(" | warming up...")

    # Ngủ 0.5s để mô phỏng dữ liệu thời gian thực
    time.sleep(0.5)

# 4. Sau streaming, đánh giá
mse = mean_squared_error(y_trues, y_preds)
print(f"\nFinal MSE after streaming: {mse:.4f}")



# import numpy as np
# from sklearn.datasets import load_diabetes
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from graforvfl import RvflRegressor
#
# # 1. Load và chuẩn hóa dữ liệu
# X, y = load_diabetes(return_X_y=True)
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
#
# # 2. Chia dữ liệu: 80% offline training, 20% streaming data
# split_idx = int(0.8 * len(X))
# X_train, y_train = X[:split_idx], y[:split_idx]
# X_stream, y_stream = X[split_idx:], y[split_idx:]
#
# # 3. Khởi tạo mô hình RVFL và huấn luyện offline ban đầu
# model = RvflRegressor(size_hidden=20, reg_alpha=0.01, seed=42)
# model.fit(X_train, y_train)
#
# # 4. Giả lập môi trường streaming (real-time)
# y_preds, y_trues = [], []
# for x_new, y_new in zip(X_stream, y_stream):
#     x_new = x_new.reshape(1, -1)
#     y_new = np.array([[y_new]])
#
#     # Dự đoán trước khi cập nhật
#     y_pred = model.predict(x_new)
#     y_preds.append(y_pred.item())
#     y_trues.append(y_new.item())
#
#     # Cập nhật mô hình với dữ liệu mới
#     model.partial_fit(x_new, y_new)
#
# # 5. Đánh giá hiệu suất sau online update
# mse = mean_squared_error(y_trues, y_preds)
# print(f"Online MSE after streaming: {mse:.4f}")
