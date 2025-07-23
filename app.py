import streamlit as st
from src.data_loader import download_data, preprocess_data
from src.model_builder import build_lstm_model
from src.predictor import predict_next
from src.utils import plot_predictions
import numpy as np
import pandas as pd

# --- Streamlit Config ---
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
sequence_length = st.sidebar.slider("Lookback Window (Days)", 30, 100, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 20)
predict_button = st.sidebar.button("Train & Predict")

if predict_button:
    with st.spinner("🔄 Downloading data..."):
        close_data, raw_data = download_data(ticker, str(start_date), str(end_date))
        st.success(f"Downloaded {len(close_data)} records.")
        st.line_chart(raw_data['Close'])

    with st.spinner("⚙️ Preprocessing data..."):
        X, y, scaler = preprocess_data(close_data, sequence_length)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    with st.spinner("🧠 Training model..."):
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    with st.spinner("📈 Making predictions..."):
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.subheader("📊 Actual vs Predicted Prices")
    plot_predictions(actual, predictions, ticker)

    with st.spinner("🔮 Predicting next-day price..."):
        last_60 = scaler.transform(close_data[-sequence_length:])
        next_price = predict_next(model, last_60, scaler)
        st.success(f"Predicted next-day closing price for {ticker}: **${next_price:.2f}**")
