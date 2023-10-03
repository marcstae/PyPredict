# PyPredict

PyPredict is a Python script that leverages the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model to predict the time of the next event. This tool processes a list of past event times in the format `hh:mm` (24-hour time format) as input, evaluates the model's accuracy using a test set, and outputs the predicted time of the next event along with the model's Mean Absolute Error (MAE).

## Requirements

- Python 3.6 or higher
- NumPy
- pandas
- pmdarima
- scikit-learn

## Installation

Clone the repository:
```bash
git clone https://github.com/marcstae/PyPredict.git

Install the required Python packages:
```bash
pip install numpy pandas pmdarima scikit-learn
```

## Usage
Run the script by providing a list of past event times as command-line arguments. Make sure that each time follows the hh:mm format (24-hour time format).

```bash
python predict_next_event.py 09:00 10:30 12:00 13:30
```
The script will then display the Mean Absolute Error on the test data to provide insight into the model's accuracy. Following that, it will output the predicted time of the next event. For example:

```
Mean Absolute Error on Test Data: 5.20 minutes
The predicted time of the next event: 15:00
```

Note:
Model performance depends on the nature of the data. The provided Mean Absolute Error offers a hint about the model's accuracy on historical data, but as with any forecast, there's no guarantee of future performance. Users are encouraged to understand the model's assumptions and limitations and apply expert judgment when needed.



