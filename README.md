# 📈 Stock Price Predictor using LSTM

>A complete deep learning-based solution for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. Includes both a Streamlit web app and a Python script for training, prediction, and visualization.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Streamlit Web App](#1-streamlit-web-app)
  - [Script Mode](#2-script-mode)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

---

## Features

- **Download stock data** from Yahoo Finance using `yfinance`
- **Preprocess and normalize** time-series data for LSTM
- **Train LSTM neural network** on historical prices
- **Predict and visualize** actual vs. predicted prices
- **Next-day price prediction** for selected stocks
- **Streamlit web interface** for interactive use
- **Script-based workflow** for automation and reproducibility

---

## Project Structure

```
├── app.py                # Streamlit web app
├── main.py               # Script for training and prediction
├── requirements.txt      # Python dependencies
├── model/                # Saved model weights
│   └── lstm_model.h5
├── src/                  # Source code modules
│   ├── data_loader.py    # Data download & preprocessing
│   ├── model_builder.py  # LSTM model construction
│   ├── predictor.py      # Prediction utilities
│   ├── trainer.py        # Model training logic
│   └── utils.py          # Plotting and helpers
```

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahuldhiman509/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Streamlit Web App

Launch the interactive app:
```bash
streamlit run app.py
```

**Web App Features:**
- Enter stock ticker, date range, lookback window, and epochs
- Visualize downloaded data
- Train LSTM model and view actual vs. predicted prices
- Predict next-day closing price

---

### 2. Script Mode

Run the main script for batch training and prediction:
```bash
python main.py
```

**Script Output:**
- Saves trained model to `model/lstm_model.h5`
- Prints predicted next-day price to console
- Plots actual vs. predicted prices

---

## How It Works

1. **Data Loading:**
   - Downloads historical closing prices for a given ticker using `yfinance`.
2. **Preprocessing:**
   - Normalizes data with MinMaxScaler
   - Creates sequences for LSTM input
3. **Model Building:**
   - Constructs an LSTM neural network using Keras
4. **Training:**
   - Trains the model on historical data
   - Saves weights for reuse
5. **Prediction:**
   - Predicts test set prices and next-day price
   - Inverse transforms predictions for readability
6. **Visualization:**
   - Plots actual vs. predicted prices using Matplotlib/Streamlit

---

## Customization

- Change ticker, date range, lookback window, and epochs in the Streamlit sidebar or in `main.py`.
- Modify model architecture in `src/model_builder.py`.
- Adjust training logic in `src/trainer.py`.

---

## Troubleshooting

- **No data downloaded:**
  - Check ticker symbol and date range
  - Ensure internet connection
- **Model training errors:**
  - Verify dependencies are installed
  - Check input data shape
- **Streamlit not launching:**
  - Run `pip install streamlit` if missing

---

## Requirements

- Python 3.8+
- `yfinance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `streamlit`

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Author

Rahul Dhiman ([GitHub](https://github.com/rahuldhiman-dev))
