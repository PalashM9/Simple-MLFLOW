# Simple-MLFLOW

## Time Series Prediction with MLflow

This project demonstrates time series prediction using multiple machine learning models and hyperparameter tuning, with experiments tracked using MLflow. The dataset used in this project is a synthetic sales revenue dataset spanning 200 months.

## Project Structure

```plaintext
time_series_mlflow/
│
├── data/
│   └── synthetic_sales_data.csv  # Synthetic dataset for time series prediction
│
├── main.py                       # Main script to execute the training pipeline
├── train.py                      # Script for model training, hyperparameter tuning, and MLflow logging
├── requirements.txt              # Python dependencies required for the project
└── README.md                     # Project description and instructions

## Dataset

The dataset (`time_series_data.csv`) contains 200 rows of synthetic sales revenue data, with two columns:

- **Month**: The date of the observation, ranging from January 2020 to August 2036.
- **SalesRevenue**: The sales revenue for the corresponding month, generated around a mean of 10,000 with added variability.

## Models and Hyperparameters

The following models are used in this project, along with their respective hyperparameters:

### Linear Regression

- **Hyperparameters**:
  - `fit_intercept`: [True, False]

### Random Forest Regressor

- **Hyperparameters**:
  - `n_estimators`: [50, 100]
  - `max_depth`: [5, 10]

### XGBoost Regressor

- **Hyperparameters**:
  - `n_estimators`: [50, 100]
  - `max_depth`: [3, 6]

## Results

The results of the experiments, including the best model and hyperparameters, are logged and can be analyzed through the MLflow UI. The following metrics are tracked for each experiment:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

These metrics help in evaluating and comparing the performance of different models and hyperparameter settings.
