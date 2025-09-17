import pandas as pd
import numpy as np
import joblib
import os
import json
from tensorflow.keras.models import load_model
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
TICKERS = ["SLV", "GLD"]
TIME_STEPS = 50
ENSEMBLE_MODELS_DIR = "new_ensemble_models"
FEATURES = [
    "GLD_return", "SLV_return", "spread", "spread_mean", "spread_std",
    "z_score", "sp500_return", "usd_return", "tnx_return",
    "gs_ratio", "sma_gld", "sma_slv"
]

API_KEY = "PKO3LZIDZAM1695TJCYO"
API_SECRET = "MZcO0STMgg84OAqslMe1zFNr7Yf6rUearnV04jQ2"
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL)

STATE_FILE = "positions.json"

# ---------------- LOAD MODELS & SCALER ----------------
model_paths = sorted([os.path.join(ENSEMBLE_MODELS_DIR, f) 
                      for f in os.listdir(ENSEMBLE_MODELS_DIR) if f.endswith(".h5")])
models = [load_model(p) for p in model_paths]
scaler = joblib.load(f"{ENSEMBLE_MODELS_DIR}/scaler.gz")

# ---------------- FETCH HISTORICAL DATA ----------------
import yfinance as yf
hist_data = yf.download(TICKERS + ["^GSPC", "DX-Y.NYB", "^TNX"], period="60d", threads=False)["Close"]

# ---------------- FETCH LATEST PRICES ----------------
latest_prices = {}
for ticker in TICKERS + ["^GSPC", "DX-Y.NYB", "^TNX"]:
    latest_prices[ticker] = api.get_latest_trade(ticker).price

# ---------------- COMPUTE RETURNS ----------------
returns = hist_data.pct_change()
# Replace last row with "today's" returns using latest price
today_returns = [(latest_prices[t] / hist_data[t].iloc[-1]) - 1 for t in TICKERS + ["^GSPC", "DX-Y.NYB", "^TNX"]]

returns.iloc[-1] = today_returns  # update last row to reflect latest "close"

# ---------------- DERIVED FEATURES ----------------
spread = latest_prices[TICKERS[0]] - latest_prices[TICKERS[1]]
spread_mean = hist_data[TICKERS[0]] - hist_data[TICKERS[1]]
spread_mean = spread_mean.rolling(20, min_periods=1).mean().iloc[-1]
spread_std = (hist_data[TICKERS[0]] - hist_data[TICKERS[1]]).rolling(20, min_periods=1).std().iloc[-1]
z_score = (spread - spread_mean) / spread_std
gs_ratio = latest_prices[TICKERS[0]] / latest_prices[TICKERS[1]]
sma_gld = hist_data[TICKERS[0]].rolling(20, min_periods=1).mean().iloc[-1]
sma_slv = hist_data[TICKERS[1]].rolling(20, min_periods=1).mean().iloc[-1]

df_features = pd.DataFrame({
    "GLD_return": [today_returns[1]],   # GLD
    "SLV_return": [today_returns[0]],   # SLV
    "spread": [spread],
    "spread_mean": [spread_mean],
    "spread_std": [spread_std],
    "z_score": [z_score],
    "sp500_return": [today_returns[2]],
    "usd_return": [today_returns[3]],
    "tnx_return": [today_returns[4]],
    "gs_ratio": [gs_ratio],
    "sma_gld": [sma_gld],
    "sma_slv": [sma_slv]
})

# ---------------- SCALE & CREATE SEQUENCE ----------------
X_scaled = scaler.transform(df_features[FEATURES].values)
# replicate last TIME_STEPS rows to feed LSTM if needed
X_seq = np.array([np.repeat(X_scaled, TIME_STEPS, axis=0)])

# ---------------- ENSEMBLE PREDICTIONS ----------------
ensemble_preds = np.mean([m.predict(X_seq, verbose=0) for m in models], axis=0)
signal = int(ensemble_preds[0,1] > 0.5)  # 1 = GLD, 0 = SLV
target_symbol = TICKERS[signal]

print("Predicted signal:", "GLD" if signal == 1 else "SLV")

# ---------------- POSITION TRACKING & EXECUTION ----------------
# Load last position if exists
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
else:
    state = {"holding": None, "qty": 0}

current_holding = state["holding"]

if current_holding != target_symbol:
    print(f"Switching position: {current_holding} -> {target_symbol}")

    # Close previous holding
    if current_holding is not None and state["qty"] > 0:
        api.submit_order(
            symbol=current_holding,
            qty=state["qty"],
            side="sell",
            type="market",
            time_in_force="day"
        )
        print(f"Sold {state['qty']} {current_holding}")

    # Account cash
    account = api.get_account()
    cash = float(account.cash)
    print(f"Cash: {cash}")

    # Use latest price for sizing
    last_price = latest_prices[target_symbol]
    qty = int(cash // last_price)  # max whole shares

    if qty > 0:
        api.submit_order(
            symbol=target_symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"Bought {qty} {target_symbol}")

        # Save new state
        state["holding"] = target_symbol
        state["qty"] = qty
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
else:
    print(f"Holding steady in {current_holding} (no trade).")
