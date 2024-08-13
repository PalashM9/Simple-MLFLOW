import mlflow
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Generate synthetic time series data if not available
    try:
        df = pd.read_csv("data/time_series_data.csv")
    except FileNotFoundError:
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        values = np.sin(np.linspace(0, 3.14 * 2, 100)) + np.random.normal(0, 0.1, 100)
        df = pd.DataFrame({'date': dates, 'value': values})
        df.to_csv("data/time_series_data.csv", index=False)

    # Run training script
    mlflow.start_run()
    os.system("python train.py")
    mlflow.end_run()
