import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def parse_time(time_str):
    hh, mm = time_str.split(':')
    return int(hh) * 60 + int(mm)

def format_time(minutes):
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def main(past_event_times):
    # Convert the input event times to minutes and numpy arrays
    past_event_minutes = [parse_time(t) for t in past_event_times]
    
    # Convert the list of event times to a pandas Series
    time_series = pd.Series(past_event_minutes)

    # Fit the ARIMA model
    model = ARIMA(time_series, order=(1, 1, 0))
    model_fit = model.fit(method_kwargs={'maxiter': 500})

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
