# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:33:18 2023

@author: Jichu
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor  # Importing XGBoost
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
data = pd.read_csv('cleaned_data.csv')
X = data.drop('Prod_LatestAvg_TotActPwr', axis=1)
y = data['Prod_LatestAvg_TotActPwr']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb = XGBRegressor()

# %% Setup the parameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9, 11],
    'subsample': [0.7, 0.8, 0.9, 1]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# %% Load the model
# best_xgb = load('xgboost_model_25122023.joblib')

# %% Evaluate the model
# Make predictions on the test set
predictions = best_xgb.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')

# %% Save the Model
dump(best_xgb, 'xgboost_model_25122023.joblib')
