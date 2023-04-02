# PyPredict

PyPredict is a Python script that predicts the time of the next event based on past event times using the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model. The script accepts input times in the hh:mm format in the 24-hour time format.

## Installation

To run PyPredict, you will need to install the statsmodels library. You can install it using pip:

```bash
pip install statsmodels
```

If you're using a virtual environment, make sure to activate it before installing the package.

## Usage
Run the script from the command line with the past event times as arguments:

```bash
python predict_next_event.py 01:30 03:00 04:30 06:00
```

The script will output the predicted time of the next event in the hh:mm format using the ARIMA model. The ARIMA model's accuracy depends on the nature of your data and the chosen order parameters (p, d, q). You might need to fine-tune these parameters for better predictions.





