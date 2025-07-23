import numpy as np
from tensorflow.keras.models import load_model

def predict_next(model, last_60_scaled, scaler):
    last_60_scaled = np.reshape(last_60_scaled, (1, 60, 1))
    predicted_scaled = model.predict(last_60_scaled)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]
