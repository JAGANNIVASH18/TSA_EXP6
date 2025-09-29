# Ex.No: 6               HOLT WINTERS METHOD
### Date: 29-09-2025



### AIM:  
To holy winters method of Indian Ocean

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# ----------------------------
# Load and prepare the dataset
# ----------------------------
file_path = 'Sunspots.csv'      # your uploaded file
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Rename value column
data.rename(columns={'Monthly Mean Total Sunspot Number': 'Value'}, inplace=True)

# Ensure numeric and drop NaNs
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data = data.dropna(subset=['Value'])

# Resample to monthly frequency (if not already monthly)
monthly_data = data['Value'].resample('MS').mean()

# ----------------------------
# Split into Train & Test
# ----------------------------
train_data = monthly_data[:int(0.9 * len(monthly_data))]
test_data = monthly_data[int(0.9 * len(monthly_data)):]

# ----------------------------
# Fit Holt-Winters model (Additive trend)
# ----------------------------
fitted_model = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()

# Forecast for test period
test_predictions = fitted_model.forecast(len(test_data))

# ----------------------------
# Graph 1: Test Data vs Predictions
# ----------------------------
plt.figure(figsize=(12,6))
test_data.plot(label='Actual Test Data', marker='o')
test_predictions.plot(label='Predicted Test Data', marker='x')
plt.title('Sunspots: Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Sunspot Number')
plt.legend()
plt.show()

# ----------------------------
# Evaluate performance
# ----------------------------
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.2f}")
print(f"Mean Squared Error = {mse:.2f}")

# ----------------------------
# Fit on Full Data & Forecast 12 Months Ahead
# ----------------------------
final_model = ExponentialSmoothing(monthly_data, trend='add', seasonal=None).fit()
forecast_predictions = final_model.forecast(steps=12)

# ----------------------------
# Graph 2: Final 12-Month Forecast
# ----------------------------
plt.figure(figsize=(12,6))
monthly_data.plot(label='Original Data', legend=True)
forecast_predictions.plot(label='12-Month Forecast', marker='o', color='green')
plt.title('Sunspots: Final 12-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Sunspot Number')
plt.legend()
plt.show()
```

### OUTPUT:


#### TEST_PREDICTION

<img width="1017" height="545" alt="image" src="https://github.com/user-attachments/assets/65225ff5-03d4-4298-89a0-6f69da720a74" />


#### FINAL_PREDICTION
<img width="1013" height="545" alt="image" src="https://github.com/user-attachments/assets/9d09fbc5-016c-44ba-b32a-16d1e62d6db2" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
