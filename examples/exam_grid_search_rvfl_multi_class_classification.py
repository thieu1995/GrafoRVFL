#!/usr/bin/env python
# Created by "Thieu" at 16:50, 18/04/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris
from graforvfl import Data, RvflClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Step 1: Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the parameter grid
param_grid = {
    'size_hidden': list(range(2, 1000)),
    'act_name': ["none", "relu", "leaky_relu", "celu", "prelu", "gelu",
                          "elu", "selu", "rrelu", "tanh", "sigmoid"],
    'weight_initializer': ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                          "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"]
}

# Step 4: Create the model
model = RvflClassifier()

# Step 5: Perform grid search
grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Step 6: Evaluate results
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Step 7: Test set evaluation
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Set Score:", test_score)
