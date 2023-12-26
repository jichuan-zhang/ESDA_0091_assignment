# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:00:27 2023

@author: Jichu
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
data = pd.read_csv('cleaned_data.csv')
X = data.drop('Prod_LatestAvg_TotActPwr', axis=1)
y = data['Prod_LatestAvg_TotActPwr']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf = RandomForestRegressor()

# %% Setup the parameter grid
param_grid = {
    'n_estimators': [10, 20, 50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# %% Load the model
best_rf = load('random_forest_model_25122023.joblib')

# %% Evaluate the model
# Make predictions on the test set
predictions = best_rf.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')


# %% Feature importances
importances = best_rf.feature_importances_

# Map to feature names and sort
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot
plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance in Random Forest Model')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()
# %% Save the Model

dump(best_rf, 'random_forest_model_25122023.joblib')