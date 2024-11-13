import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the S&P 500 data
ticker = '^GSPC'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
df = pd.DataFrame(data)

# Plot the stock price over time
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price History')
plt.title('S&P 500 Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create a dataset with sequences of 60 days
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predicting and plotting
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
actual_train_values = scaler.inverse_transform(y_train.reshape(-1, 1))
actual_test_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot training predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(actual_train_values, label='Actual Prices')
plt.plot(train_predict, label='Training Predictions')
plt.title('Training Predictions vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot testing predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(actual_test_values, label='Actual Prices')
plt.plot(test_predict, label='Testing Predictions')
plt.title('Testing Predictions vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Calculate and plot residual errors
train_errors = actual_train_values - train_predict
test_errors = actual_test_values - test_predict

plt.figure(figsize=(14, 7))
plt.hist(train_errors, bins=50, alpha=0.5, label='Training Errors')
plt.hist(test_errors, bins=50, alpha=0.5, label='Testing Errors')
plt.title('Residual Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Print mean squared error
train_mse = mean_squared_error(actual_train_values, train_predict)
test_mse = mean_squared_error(actual_test_values, test_predict)
print(f'Training Mean Squared Error: {train_mse}')
print(f'Testing Mean Squared Error: {test_mse}')
