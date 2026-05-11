import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------------
# Load Models
# -------------------------------
arima_model = pickle.load(open("arima_model.pkl", "rb"))
lstm_model = load_model("lstm_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# App Title
# -------------------------------           
st.title("📈 Stock Prediction App")
st.write("Compare ARIMA and LSTM predictions")

# -------------------------------
# User Inputs
# -------------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# -------------------------------
# Fetch Data
# -------------------------------
if st.button("Predict"):

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Try another ticker.")
    else:
        st.subheader("Stock Data")
        st.write(data.tail())

        close_data = data['Close'].values.reshape(-1, 1)

        # -------------------------------
        # ARIMA Prediction
        # -------------------------------
        arima_pred = arima_model.forecast(steps=10)

        # -------------------------------
        # LSTM Prediction
        # -------------------------------
        # Scale data
        scaled_data = scaler.transform(close_data)

        # Create sequences (last 60 days)
        sequence_length = 60
        X_test = []

        for i in range(sequence_length, len(scaled_data)):
            X_test.append(scaled_data[i-sequence_length:i])

        X_test = np.array(X_test)

        # Predict
        lstm_pred = lstm_model.predict(X_test)

        # Inverse scale
        lstm_pred = scaler.inverse_transform(lstm_pred)

        # -------------------------------
        # Display Predictions
        # -------------------------------
        st.subheader("Predictions")

        st.write("ARIMA Prediction (Next 10 Days):")
        st.write(arima_pred)

        st.write("LSTM Prediction:")
        st.write(lstm_pred[-10:])  # last 10 predictions

        # -------------------------------
        # Plot Graph
        # -------------------------------
        st.subheader("Prediction Graph")

        plt.figure(figsize=(10, 5))

        plt.plot(data['Close'].values, label='Actual Prices')

        # Align predictions for plotting
        arima_plot = np.full(len(data), np.nan)
        arima_plot[-10:] = arima_pred

        lstm_plot = np.full(len(data), np.nan)
        lstm_plot[-len(lstm_pred):] = lstm_pred.flatten()

        plt.plot(arima_plot, label='ARIMA Prediction')
        plt.plot(lstm_plot, label='LSTM Prediction')

        plt.legend()
        st.pyplot(plt)