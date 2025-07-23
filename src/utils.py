import matplotlib.pyplot as plt

def plot_predictions(actual, predicted, ticker="AAPL"):
    plt.figure(figsize=(14, 5))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
