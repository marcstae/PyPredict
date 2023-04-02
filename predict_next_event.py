import sys
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def parse_time(time_str):
    hh, mm = time_str.split(':')
    return int(hh) * 60 + int(mm)

def format_time(minutes):
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def find_optimal_d(time_series, max_d=2):
    for d in range(max_d + 1):
        test_result = adfuller(time_series.diff(d).dropna())
        if test_result[1] < 0.05:
            return d
    return max_d

def find_optimal_arima_order(time_series, max_p=3, max_q=3, max_d=2):
    optimal_order = (0, 0, 0)
    best_aic = float("inf")

    d = find_optimal_d(time_series, max_d)

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue

        try:
            model = ARIMA(time_series, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic

            if aic < best_aic:
                best_aic = aic
                optimal_order = (p, d, q)
        except:
            continue

    return optimal_order

def main(past_event_times):
    # Convert the input event times to minutes and numpy arrays
    past_event_minutes = [parse_time(t) for t in past_event_times]
    
    # Convert the list of event times to a pandas Series
    time_series = pd.Series(past_event_minutes)

    # Find the optimal ARIMA order using AIC
    optimal_order = find_optimal_arima_order(time_series)

    # Fit the ARIMA model with the optimal order
    model = ARIMA(time_series, order=optimal_order)
    model_fit = model.fit()

    # Predict the next event time
    next_event_minutes = model_fit.forecast(steps=1).iloc[0]

    return format_time(int(round(next_event_minutes)))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_next_event.py event_time1 event_time2 ...")
        sys.exit(1)
    
    past_event_times = sys.argv[1:]
    next_event_time = main(past_event_times)
    
    print(f"The predicted time of the next event: {next_event_time}")
