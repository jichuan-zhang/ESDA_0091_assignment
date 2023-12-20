## Data Source
EDA: [Wind-Turbine-SCADA-signals-2017_0](https://www.edp.com/en/innovation/open-data/turbine-scada-signals-2017) contains the SCADA recorded by its wind turbines

EDA: [opendata-wind-failures-2017](https://www.edp.com/en/innovation/open-data/historical-failure-logbook-2017) contains the event cause the wind turbines to fail

## Data Cleaning

 1. Use insight and past literatures to select most relevant features (as opposed to PCA)
 2. Use the failures data to remove the SCADA signal during the failure period as they are outliers
 3. Check if there is any outstanding values and correct/remove them

## Model Training

 1. A Random Forest Model is trained as a benchmark model
 2. Each team member choose one algorithm they prefer to train

## Performance Evaluation
- Model Performance will be compared using R-squared and MAPE matrix
- Draw some conclusion as what model is better
## Workload Distribution
| Workload | Code | Report |
|--|--|--|
| Data Cleaning | Jichuan | Jichuan |
| Literature Review | N/A |  |
| Benchmark - Random Forest | Jichuan | Jichuan |
| Model 1 - XG Boost | Jichuan | Jichuan |
| Model 2 -  |  |  |
| Model 3 -  |  |  |
| Model 4 -  |  |  |
| Report - Introduction and Summary | N/A |  |
| Report - Discussion | N/A |  |

> Written with [StackEdit](https://stackedit.io/).
