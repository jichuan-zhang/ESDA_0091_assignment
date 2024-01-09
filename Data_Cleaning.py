# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:25:14 2023

@author: Jichu
"""

# Import Libraries and Load Data
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

scada_data_path = 'Wind-Turbine-SCADA-signals-2017_modified.csv'
failures_data_path = 'opendata-wind-failures-2017.csv'

scada_data = pd.read_csv(scada_data_path, parse_dates=['Timestamp'])
failures_data = pd.read_csv(failures_data_path, parse_dates=['Timestamp'])

clean_data = pd.DataFrame()

# %% Sort SCADA Data
scada_data_sorted = scada_data.sort_values(by=['Turbine_ID', 'Timestamp']).reset_index(drop=True)

# %% Process Each Turbine
for turbine in scada_data_sorted['Turbine_ID'].unique():
    turbine_scada_data = scada_data_sorted[scada_data_sorted['Turbine_ID'] == turbine].reset_index(drop=True)
    turbine_failures = failures_data[failures_data['Turbine_ID'] == turbine]

    if turbine_failures.empty:
        # If there are no failures for this turbine, append its data as is
        clean_data = pd.concat([clean_data, turbine_scada_data], ignore_index=True)
    else:
        # %% Process Each Failure
        for _, failure in turbine_failures.iterrows():
            # Rounding failure timestamp to the nearest 10 minutes
            rounded_failure_time = failure['Timestamp'].round('10min')
            nearest_index = turbine_scada_data['Timestamp'].sub(rounded_failure_time).abs().idxmin()
    
            # Initialize start and end indices for the failure
            start_index, end_index = nearest_index, nearest_index
    
            # %% Look Backward to Find Start of Failure
            consecutive_positives = 0
            for i in range(nearest_index, 0, -1):
                if turbine_scada_data.iloc[i]['Prod_LatestAvg_TotActPwr'] > 0:
                    consecutive_positives += 1
                    if consecutive_positives == 5:
                        start_index = i + consecutive_positives  # Corrected start index
                        break
                else:
                    consecutive_positives = 0  # Reset if a non-positive value is found
            
            # Look Forward to Find End of Failure
            consecutive_positives = 0
            for i in range(nearest_index, len(turbine_scada_data)):
                if turbine_scada_data.iloc[i]['Prod_LatestAvg_TotActPwr'] > 0:
                    consecutive_positives += 1
                    if consecutive_positives == 5:
                        end_index = i + consecutive_positives  # Corrected end index
                        break
                else:
                    consecutive_positives = 0  # Reset if a non-positive value is found
    
            # %% Exclude Failure Period
            if start_index >= 0 and end_index < len(turbine_scada_data):
                start_time = turbine_scada_data.iloc[start_index]['Timestamp']
                end_time = turbine_scada_data.iloc[end_index]['Timestamp']
                turbine_scada_data = turbine_scada_data[
                    ~((turbine_scada_data['Timestamp'] >= start_time) &
                      (turbine_scada_data['Timestamp'] <= end_time))
                ]
    
        # %% Append Cleaned Data for This Turbine
        clean_data = pd.concat([clean_data, turbine_scada_data], ignore_index=True)

# %% Drop NaN Values
rows_before_drop = len(clean_data)
clean_data.dropna(inplace=True)
rows_after_drop = len(clean_data)
rows_dropped = rows_before_drop - rows_after_drop

print(f"Number of rows before dropping NaNs: {rows_before_drop}")
print(f"Number of rows after dropping NaNs: {rows_after_drop}")
print(f"Number of rows dropped: {rows_dropped}")

# %% Convert Direction to Sine and Cosine
clean_data['Amb_WindDir_Abs_Avg_sin'] = np.sin(np.radians(clean_data['Amb_WindDir_Abs_Avg']))
clean_data['Amb_WindDir_Abs_Avg_cos'] = np.cos(np.radians(clean_data['Amb_WindDir_Abs_Avg']))
clean_data.drop('Amb_WindDir_Abs_Avg', axis=1, inplace=True)


# %% Save Clean Data
cols_to_exclude = ['Turbine_ID', 'Timestamp']
clean_data.loc[:, ~clean_data.columns.isin(cols_to_exclude)].to_csv('cleaned_data.csv', index=False)























