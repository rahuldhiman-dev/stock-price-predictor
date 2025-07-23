from src.data_loader import download_data, preprocess_data
from src.model_builder import build_lstm_model
from src.trainer import train_model
from src.predictor import predict_next
from src.utils import plot_predictions
import numpy as np

# CONFIG
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
MODEL_PATH = "model/lstm_model.h5"

# Step 1: Load + preprocess
close_data, raw_data = download_data(TICKER, START_DATE, END_DATE)
X, y, scaler = preprocess_data(close_data)

# Step 2: Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 3: Build + train model
model = build_lstm_model((X_train.shape[1], 1))
train_model(model, X_train, y_train, save_path=MODEL_PATH)

# Step 4: Predict + plot
model = build_lstm_model((X_train.shape[1], 1))
model.load_weights(MODEL_PATH)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plot_predictions(actual, predictions, TICKER)

# Step 5: Predict next day
last_60 = scaler.transform(close_data[-60:])
next_day_price = predict_next(MODEL_PATH, last_60, scaler)
print(f"Predicted next {TICKER} closing price: ${next_day_price:.2f}")
