import sys
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

def train_test_split(time_series, test_size=0.2):
    split_index = int(len(time_series) * (1 - test_size))
    train, test = time_series[:split_index], time_series[split_index:]
    return train, test

def find_optimal_arima_model(time_series):
    model = auto_arima(time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model

def main(past_event_times):
    time_series = pd.Series([int(time.split(':')[0]) * 60 + int(time.split(':')[1]) for time in past_event_times])
    train, test = train_test_split(time_series)

    model = find_optimal_arima_model(train)
    test_predictions = model.predict(n_periods=len(test))
    
    mae = mean_absolute_error(test, test_predictions)
    print(f"Mean Absolute Error on Test Data: {mae:.2f} minutes")
    
    next_event_minutes = list(model.predict(n_periods=1))[0]
    next_event_time = f'{int(next_event_minutes // 60):02d}:{int(next_event_minutes % 60):02d}'
    
    return next_event_time

if __name__ == '__main__':
    try:
        past_event_times = sys.argv[1:]
        next_event_time = main(past_event_times)
        print(f"The predicted time of the next event: {next_event_time}")
    except Exception as e:
        print(f"Error encountered: {e}")

