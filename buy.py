import boto3
import json
import pytz
import datetime
import time
import pandas_market_calendars as mcal
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
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
STATE_FILE = "positions.json"

# ---------------- FETCH ALPACA KEYS FROM SECRETS MANAGER ----------------
def get_alpaca_keys(secret_name="alpaca_paper", region_name="us-east-1"):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(response["SecretString"])
        return secret_dict["API_KEY"], secret_dict["API_SECRET"]
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca keys from Secrets Manager: {e}")

API_KEY, API_SECRET = get_alpaca_keys()
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL)

# ---------------- NYSE CALENDAR ----------------
nyse = mcal.get_calendar("XNYS")
ny_tz = pytz.timezone("America/New_York")


def is_market_open_now():
    now = datetime.datetime.now(ny_tz)
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

    if schedule.empty:
        return False  # holiday / weekend

    market_open = schedule.iloc[0]["market_open"].tz_convert(ny_tz)
    market_close = schedule.iloc[0]["market_close"].tz_convert(ny_tz)

    return market_open <= now <= market_close


def run_trading():
    now = datetime.datetime.now(ny_tz)
    print(f"\n[{now}] Checking trading job...")

    if not is_market_open_now():
        print("Market is closed today. Skipping trade.")
        return

    print("Market is OPEN. Running strategy...")

    # ---------------- LOAD MODELS & SCALER ----------------
    model_paths = sorted([os.path.join(ENSEMBLE_MODELS_DIR, f) 
                          for f in os.listdir(ENSEMBLE_MODELS_DIR) if f.endswith(".h5")])
    models = [load_model(p) for p in model_paths]
    scaler = joblib.load(f"{ENSEMBLE_MODELS_DIR}/scaler.gz")

    # ---------------- FETCH LATEST DATA ----------------
    data = yf.download(TICKERS + ["^GSPC", "DX-Y.NYB", "^TNX"], period="60d", threads=False)["Close"]
    returns = data.pct_change()

    spread = data[TICKERS[0]] - data[TICKERS[1]]
    spread_mean = spread.rolling(20, min_periods=1).mean()
    spread_std = spread.rolling(20, min_periods=1).std()
    z_score = (spread - spread_mean) / spread_std
    gs_ratio = data[TICKERS[0]] / data[TICKERS[1]]
    sma_gld = data[TICKERS[0]].rolling(20, min_periods=1).mean()
    sma_slv = data[TICKERS[1]].rolling(20, min_periods=1).mean()

    df_features = pd.DataFrame({
        "GLD_return": returns[TICKERS[0]],
        "SLV_return": returns[TICKERS[1]],
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "z_score": z_score,
        "sp500_return": returns["^GSPC"],
        "usd_return": returns["DX-Y.NYB"],
        "tnx_return": returns["^TNX"],
        "gs_ratio": gs_ratio,
        "sma_gld": sma_gld,
        "sma_slv": sma_slv
    })
    df_features.dropna(inplace=True)

    # ---------------- SCALE & CREATE SEQUENCES ----------------
    X_scaled = scaler.transform(df_features[FEATURES].values)
    X_seq = np.array([X_scaled[-TIME_STEPS:]])  # last sequence for prediction

    # ---------------- ENSEMBLE PREDICTIONS ----------------
    ensemble_preds = np.mean([m.predict(X_seq, verbose=0) for m in models], axis=0)
    signal = int(ensemble_preds[0,1] > 0.5)  # 1 = GLD, 0 = SLV
    target_symbol = TICKERS[signal]

    print("Predicted signal:", "GLD" if signal == 1 else "SLV")

    # ---------------- POSITION TRACKING ----------------
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {"holding": None, "qty": 0}

    current_holding = state["holding"]

    if current_holding != target_symbol:
        print(f"Switching position: {current_holding} -> {target_symbol}")

        if current_holding is not None and state["qty"] > 0:
            api.submit_order(
                symbol=current_holding,
                qty=state["qty"],
                side="sell",
                type="market",
                time_in_force="day"
            )
            print(f"Sold {state['qty']} {current_holding}")

        account = api.get_account()
        cash = float(account.cash)
        print(f"Cash: {cash}")

        last_price = yf.download(target_symbol, period="1d", interval="1m")["Close"].iloc[-1]
        qty = int(cash // last_price)

        if qty > 0:
            api.submit_order(
                symbol=target_symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day"
            )
            print(f"Bought {qty} {target_symbol}")

            state["holding"] = target_symbol
            state["qty"] = qty
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
    else:
        print(f"Holding steady in {current_holding} (no trade).")


if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone=ny_tz)
    scheduler.add_job(run_trading, "cron", day_of_week="mon-fri", hour=15, minute=45)
    scheduler.start()
    print("Scheduler started. Waiting for next trading time...")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
