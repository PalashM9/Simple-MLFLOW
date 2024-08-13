import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Month'])
    df.set_index('Month', inplace=True)
    return df


# Preprocess data
def preprocess_data(df):
    X = np.array([i for i in range(len(df))]).reshape(-1, 1)
    y = df['SalesRevenue'].values
    return X, y


# Train model
def train_model(X_train, y_train, model, params):
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    grid_search = GridSearchCV(model, params, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Evaluate model
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae


# Main training function
def main(file_path):
    df = load_data(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        'LinearRegression': (LinearRegression(), {'fit_intercept': [True, False]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
        'XGBoost': (XGBRegressor(), {'n_estimators': [50, 100], 'max_depth': [3, 6]})
    }

    input_example = pd.DataFrame(X_test).iloc[:1]  # Example input for model signature

    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            best_model, best_params = train_model(X_train, y_train, model, params)
            mse, mae = evaluate_model(best_model, X_test, y_test)

            mlflow.log_param('best_params', best_params)
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('mae', mae)

            # Log model with an input example
            mlflow.sklearn.log_model(best_model, model_name, input_example=input_example)


if __name__ == "__main__":
    main("data/time_series_data.csv")
