import streamlit as st

st.set_page_config(page_title="About | Stock Price Predictor", page_icon="ℹ️")

st.title("ℹ️ About This App")
st.markdown(
    """
This tool trains a lightweight LSTM to forecast the next daily closing price for any publicly traded ticker.

**How it works**
- **Data**: Downloads historical daily prices from Yahoo Finance with your chosen date range.
- **Scaling & windows**: Normalizes prices with MinMaxScaler and builds sliding windows (lookback you set in the sidebar).
- **Model**: Two-layer LSTM with dropout, optimized with Adam on MSE loss.
- **Evaluation**: Splits 80/20 train-test, plots predicted vs. actual prices.
- **Next-day forecast**: Uses the latest window to infer tomorrow’s close and reports it immediately.

**Tips for better results**
- Use at least two years of data for more stable patterns.
- Commodity or very illiquid tickers may yield noisy signals.
- If you only need quick exploration, reduce epochs to speed up training.
"""
)

# st.subheader("Project Files")
# st.code(
#     """app.py              # Streamlit UI
# src/data_loader.py   # Download + scaling
# src/model_builder.py  # LSTM architecture
# src/trainer.py        # Training loop + checkpoint
# src/predictor.py      # Single-step forecast
# src/utils.py          # Plotting helpers
# model/lstm_model.h5   # Saved weights (created after training)
# """
# )

st.info(
    "All computations run locally in your session. No data is sent to external services "
    "beyond fetching market prices from Yahoo Finance."
)
