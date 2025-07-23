import os
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, epochs=20, batch_size=32, save_path='model/lstm_model.h5'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='loss', mode='min')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
