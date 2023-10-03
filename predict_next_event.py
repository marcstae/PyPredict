import sys
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

def train_test_split(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    train, test = data[:split_index], data[split_index:]
    return train, test

def evaluate_model(time_series):
    train, test = train_test_split(time_series)
    
    model = auto_arima(train, seasonal=True, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)
    
    predictions = model.predict(n_periods=len(test))
    
    mae = mean_absolute_error(test, predictions)
    
    return model, mae

def main(past_event_times):
    # Preprocessing: Convert time string to minutes
    time_series = pd.Series([int(time.split(':')[0]) * 60 + int(time.split(':')[1]) for time in past_event_times])
    
    model, mae = evaluate_model(time_series)
    print(f'Mean Absolute Error on Test Data: {mae:.2f} minutes')
    
    # Analyzing residuals
    residuals = pd.Series(model.resid())
    if residuals.autocorr() > 0.05: # this threshold can be adjusted
        print("Warning: Residuals have some correlation left. Model may benefit from further tuning.")
    
    # Forecast the next event time
    next_event_minutes = model.predict(n_periods=1)[0]
    next_event_time = f'{int(next_event_minutes // 60):02d}:{int(next_event_minutes % 60):02d}'
    
    return next_event_time

if __name__ == '__main__':
    past_event_times = sys.argv[1:]
    next_event_time = main(past_event_times)
    print(f'The predicted time of the next event: {next_event_time}')
