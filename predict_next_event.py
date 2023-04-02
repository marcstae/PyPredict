import sys
import numpy as np
from sklearn.linear_model import LinearRegression

def parse_time(time_str):
    hh, mm = time_str.split(':')
    return int(hh) * 60 + int(mm)

def format_time(minutes):
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def main(past_event_times):
    # Convert the input event times to minutes and numpy arrays
    past_event_times = np.array([parse_time(t) for t in past_event_times]).reshape(-1, 1)
    
    # Create an array with event indices
    event_indices = np.arange(len(past_event_times)).reshape(-1, 1)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(event_indices, past_event_times)
    
    # Predict the next event index
    next_event_index = len(past_event_times)
    next_event_minutes = model.predict([[next_event_index]])[0][0]
    
    return format_time(int(round(next_event_minutes)))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_next_event.py event_time1 event_time2 ...")
        sys.exit(1)
    
    past_event_times = sys.argv[1:]
    next_event_time = main(past_event_times)
    
    print(f"The predicted time of the next event: {next_event_time}")

#python predict_next_event.py 01:30 03:00 04:30 06:00
