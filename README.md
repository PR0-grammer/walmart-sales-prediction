# walmart-sales-prediction
A machine learning model to forecast Walmart sales based on historical data.

- `Walmart_Store_sales.csv`: The dataset used for training and evaluation.
- `Walmart_Sales_Prediction.ipynb`: The main Jupyter Notebook with all the preprocessing, modeling, and analysis.

## Problem Statement

The goal is to accurately predict future sales for Walmart stores using a supervised learning model for the purpose of improving inventory management.

## Features Used

- Store ID
- Day
- Month
- Year
- Weekly Sales (The target)
- Temperature
- Fuel Price
- CPI (Consumer Price Index)
- Unemployment Rate
- Holiday Indicator

## Libraries Used

- Python
- Pandas
- NumPy
- joblib
- xgboost
- lightgbm
- Scikit-learn
- Matplotlib

## Model(s) Implemented

- RandomForestRegressor
- XGBRegressor (XGBoost)
- LGBMRegressor (LightGBM)

## Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error
- R² Score

## Results

The best-performing model wich was LightGBM achieved the LightGBM [MAE: 36303.68, RMSE: 52129.36, R²: 0.9905].
