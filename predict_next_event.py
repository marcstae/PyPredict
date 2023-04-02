import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")


def find_optimal_d(time_series, max_d):
    best_d = 0
    best_aic = np.inf

    for d in range(max_d + 1):
        try:
            test_result = adfuller(time_series.diff(d).dropna())
            if test_result[1] < 0.05:
                return d
        except ValueError:
            continue

    return best_d


def find_optimal_arima_order(time_series, max_p=3, max_d=2, max_q=3):
    d = find_optimal_d(time_series, max_d)
    best_order = (0, d, 0)
    best_aic = np.inf

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(time_series, order=(p, d, q))
                model_fit = model.fit()
                current_aic = model_fit.aic
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = (p, d, q)
            except:
                continue

    return best_order


def main(past_event_times):
    time_series = pd.Series([int(time.split(':')[0]) * 60 + int(time.split(':')[1]) for time in past_event_times])

    arima_order = find_optimal_arima_order(time_series)
    model = ARIMA(time_series, order=arima_order)
    model_fit = model.fit()

    next_event_minutes = model_fit.forecast(steps=1).iloc[0]
    next_event_time = f'{int(next_event_minutes // 60):02d}:{int(next_event_minutes % 60):02d}'
    return next_event_time


if __name__ == '__main__':
    past_event_times = sys.argv[1:]
    next_event_time = main(past_event_times)
    print(f'The predicted time of the next event: {next_event_time}')
