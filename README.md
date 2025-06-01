# Jupyter Notebook: Gold Price Prediction with LSTM

This Jupyter Notebook demonstrates how to build and train an LSTM (Long Short-Term Memory) neural network to predict the price of gold. It uses historical financial data, including gold prices, the US Dollar Index (DXY), 10-year bond yields, and an inflation proxy (TIP ETF), to train the model.

## 📍 Overview

The notebook covers the following key steps:

1.  **Data Acquisition:**
    *   Fetching historical data for Gold (GC=F), DXY (DX-Y.NYB), 10-year US Treasury yields (^TNX), and an inflation proxy ETF (TIP) using the `yfinance` library.
2.  **Feature Engineering:**
    *   Calculating new features such as:
        *   Gold/Dollar Ratio
        *   Implied Inflation (monthly percentage change of TIP)
        *   Real Interest Rate (10-year bond yield - implied inflation)
        *   200-day Simple Moving Average (SMA) of Gold Price
        *   Bollinger Bands (Upper and Lower) for Gold Price
        *   Volatility (difference between Upper and Lower Bollinger Bands)
    *   Combining all features into a single DataFrame and handling missing values.
3.  **Data Preprocessing:**
    *   Defining input features and the target variable (Gold Price).
    *   Scaling both features and the target variable using `RobustScaler` to handle outliers.
4.  **Sequence Preparation:**
    *   Creating sequences of historical data (lookback window of 90 days) to predict future gold prices (forecast horizon of 5 days).
5.  **Data Splitting:**
    *   Chronologically splitting the data into training (70%), validation (15%), and test (15%) sets.
    *   A `TimeseriesGenerator` is also set up, although the manual sequence preparation is primarily used for training the hierarchical LSTM.
6.  **LSTM Model Architecture:**
    *   Defining a sequential LSTM model with:
        *   An LSTM layer with 192 units, L2 regularization, and `return_sequences=True`.
        *   A Dropout layer (0.35).
        *   A second LSTM layer with 96 units and L2 regularization.
        *   A Dense output layer with 5 units (to predict the 5-day forecast horizon).
7.  **Model Compilation:**
    *   Compiling the model with the Adam optimizer, Mean Squared Error (MSE) loss, and Mean Absolute Error (MAE) metric.
    *   Implementing callbacks: `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to adjust the learning rate during training.
8.  **Model Training:**
    *   Training the LSTM model using the prepared sequences and the `TimeseriesGenerator`.
    *   Visualizing the training loss.
9.  **Prediction:**
    *   Making a sample prediction using the last 30 days of available data.
    *   Making predictions on the entire test set.
10. **Evaluation:**
    *   Calculating MAE and RMSE in percentage terms on the test set to evaluate model performance.
    *   Visualizing the real vs. predicted gold prices.
11. **Model Saving:**
    *   Saving the trained Keras model (`modelo_lstm_oro.keras`).
    *   Saving the `RobustScaler` objects for features and target using `joblib`.
    *   Saving metadata (features used, training period) as a JSON file.

## ⚙️ Prerequisites

*   Python 3.x
*   Jupyter Notebook or a compatible environment.
*   The following Python libraries:
    *   `yfinance`
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `tensorflow` (which includes Keras)
    *   `matplotlib`
    *   `joblib`

You can typically install these libraries using pip:
```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib joblib
