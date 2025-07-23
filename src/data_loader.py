import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data['Close'].values.reshape(-1, 1), data

def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler
