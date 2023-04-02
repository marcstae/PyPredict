# PyPredict

PyPredict is a Python script that uses the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model to predict the time of the next event. The script takes a list of past event times in the format hh:mm (24-hour time format) as input and outputs the predicted time of the next event.

## Requirements
- Python 3.6 or higher
- NumPy
- pandas
- statsmodels

## Installation

Clone the repository:
git clone https://github.com/marcstae/PyPredict.git

Install the required Python packages:
```bash
pip install numpy pandas statsmodels
```

## Usage
Run the script by providing a list of past event times as command-line arguments. Ensure that each time is in the hh:mm format (24-hour time format).

```bash
python predict_next_event.py 09:00 10:30 12:00 13:30
```
The script will output the predicted time of the next event:
The predicted time of the next event: 15:00







