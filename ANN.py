# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:14:02 2024

@author: Sandy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
data = pd.read_csv('D:/EDA/ESDA_0091_assignment/cleaned_data.csv')
X = data.drop('Prod_LatestAvg_TotActPwr', axis=1)
y = data['Prod_LatestAvg_TotActPwr']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an ANN regressor
ann_regressor = MLPRegressor(max_iter=1000)

param_grid = {
    'hidden_layer_sizes': [(50, 50), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01],
    'beta_1': [0.9, 0.95],  # Adjusted beta_1
    'beta_2': [0.999, 0.995],  # Adjusted beta_2
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=ann_regressor, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error',verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')
