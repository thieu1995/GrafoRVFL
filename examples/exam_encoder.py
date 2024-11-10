#!/usr/bin/env python
# Created by "Thieu" at 19:45, 10/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from graforvfl.shared.scaler import OneHotEncoder


# Input data
data = np.array(['cat', 'dog', 'bird', 'cat', 'dog', 'None', 1.4, 10, np.nan])

# Create and fit-transform
encoder = OneHotEncoder()
one_hot_encoded = encoder.fit_transform(data)

# Results
print("Categories:", encoder.categories_)
print("One-Hot Encoded Matrix:\n", one_hot_encoded)

# Inverse transform
original_data = encoder.inverse_transform(one_hot_encoded)

# Results
print("Categories:", encoder.categories_)
print("One-Hot Encoded Matrix:\n", one_hot_encoded)
print("Inverse Transformed Data:", original_data)
