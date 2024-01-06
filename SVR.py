#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:02:12 2023

@author: wangxuan
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from joblib import dump
from joblib import load


# Load data
data = pd.read_csv('cleaned_data.csv')
X = data.drop('Prod_LatestAvg_TotActPwr', axis=1)
y = data['Prod_LatestAvg_TotActPwr']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVR model
svr = SVR()

# Define the hyperparameter grid
param_grid = {
'C': [50000,100000,200000], # Consider a wider range and more values
'gamma': ['scale', 'auto'], # More options for gamma
'kernel': ['rbf'], # Try different kernels
'epsilon': [1000] # Include epsilon in the search
}

# Set up GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=2, scoring='neg_mean_squared_error',verbose=3)

# Train the model
grid_search.fit(X_train_scaled, y_train)

# Best parameters
best_parameters = grid_search.best_params_
print("Best Parameters:", best_parameters)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)


print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared: {r_squared}')