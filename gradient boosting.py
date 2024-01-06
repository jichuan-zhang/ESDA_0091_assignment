#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


file_path = '/Users/leah/Desktop/github repository/ESDA_0091_assignment/cleaned_data.csv' 
data = pd.read_csv(file_path)


# In[8]:


X = data.drop('Prod_LatestAvg_TotActPwr', axis=1)
y = data['Prod_LatestAvg_TotActPwr']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


#best hyper parametre?
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}


# In[11]:


gbr = GradientBoostingRegressor(random_state=42)


# In[12]:


grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')


# In[13]:


grid_search.fit(X_train, y_train)


# In[15]:


# Best parameters
best_params = grid_search.best_params_
print("Best Parameters: ", best_params)


# In[16]:


# Initialize the best model
best_gbr = GradientBoostingRegressor(**best_params, random_state=42)


# In[17]:


best_gbr.fit(X_train, y_train)


# In[18]:


# Predict and evaluate
y_pred = best_gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[ ]:




