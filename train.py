import os
import json
import time
from pathlib import Path
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------- CONFIG ----------------
TICKERS = ["GLD", "SLV"]
SP500_TICKER = "^GSPC"
START_DATE = "2013-01-01"
END_DATE = "2025-09-16"
TIME_STEPS = 50
BATCH_SIZE = 16
EPOCHS = 15
LSTM_UNITS = [64, 32]
DROPOUT_RATE = 0.3
LR = 0.0005
NUM_MODELS = 15
TRANSACTION_COST = 0
SEED = 42
TEST_YEARS = 1
DAYS_PER_YEAR = 252
OUT_DIR = Path("models")  # output folder for models & results

# ---------------- UTILITIES ----------------
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

def sharpe_ratio(returns, days_per_year=252):
    if len(returns) == 0:
        return np.nan
    mean = np.nanmean(returns)
    std = np.nanstd(returns)
    if std == 0:
        return np.nan
    return mean / std * np.sqrt(days_per_year)

def max_drawdown(cum_returns):
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peaks) / peaks
    return drawdowns.min()

def strategy_returns(labels, rets0, rets1, tx_cost=TRANSACTION_COST):
    gross = labels * rets0 + (1 - labels) * rets1
    prev = np.concatenate(([labels[0]], labels[:-1]))
    changes = (labels != prev).astype(float)
    costs = changes * tx_cost
    return gross - costs

def walk_forward_split(X, y, seq_idx, train_window=252*3, val_window=252):
    splits = []
    start = 0
    while start + train_window + val_window <= len(X):
        X_train = X[start:start+train_window]
        y_train = y[start:start+train_window]
        X_val = X[start+train_window:start+train_window+val_window]
        y_val = y[start+train_window:start+train_window+val_window]
        idx_val = seq_idx[start+train_window:start+train_window+val_window]
        splits.append((X_train, y_train, X_val, y_val, idx_val))
        start += val_window
    return splits

# ---------------- LOAD DATA ----------------
tickers_all = TICKERS + [SP500_TICKER, "DX-Y.NYB", "^TNX"]  # USD index and 10yr yield
data = yf.download(tickers_all, start=START_DATE, end=END_DATE)["Close"]
returns = data.pct_change()

# ---------------- ENGINEER FEATURES ----------------
spread = data[TICKERS[0]] - data[TICKERS[1]]
spread_mean = spread.rolling(20, min_periods=1).mean()
spread_std = spread.rolling(20, min_periods=1).std()
z_score = (spread - spread_mean) / spread_std
gs_ratio = data[TICKERS[0]] / data[TICKERS[1]]
sma_gld = data[TICKERS[0]].rolling(20, min_periods=1).mean()
sma_slv = data[TICKERS[1]].rolling(20, min_periods=1).mean()

df_features = pd.DataFrame({
    f"{TICKERS[0]}_return": returns[TICKERS[0]],
    f"{TICKERS[1]}_return": returns[TICKERS[1]],
    "spread": spread,
    "spread_mean": spread_mean,
    "spread_std": spread_std,
    "z_score": z_score,
    "sp500_return": returns[SP500_TICKER],
    "usd_return": returns["DX-Y.NYB"],
    "tnx_return": returns["^TNX"],
    "gs_ratio": gs_ratio,
    "sma_gld": sma_gld,
    "sma_slv": sma_slv
})
df_features.dropna(inplace=True)

# ---------------- CREATE LABELS ----------------
df_features["Target"] = (returns[TICKERS[0]].shift(-1) > returns[TICKERS[1]].shift(-1)).astype(int)
df_features.dropna(inplace=True)

# ---------------- FEATURES LIST ----------------
FEATURES = [f"{TICKERS[0]}_return", f"{TICKERS[1]}_return", "spread",
            "spread_mean", "spread_std", "z_score", "sp500_return",
            "usd_return", "tnx_return", "gs_ratio", "sma_gld", "sma_slv"]

# ---------------- SCALE FEATURES ----------------
scaler = StandardScaler()
X_all = scaler.fit_transform(df_features[FEATURES].values)

# ---------------- CREATE SEQUENCES ----------------
X_seq, y_seq = create_sequences(X_all, df_features["Target"].values, TIME_STEPS)
seq_idx = df_features.index[TIME_STEPS:TIME_STEPS + len(y_seq)]
n_features = X_seq.shape[2]

# ---------------- SPLIT INTO TRAIN/VAL AND TEST ----------------
test_window = TEST_YEARS * DAYS_PER_YEAR
X_train_val = X_seq[:-test_window]
y_train_val = y_seq[:-test_window]
idx_train_val = seq_idx[:-test_window]

X_test = X_seq[-test_window:]
y_test = y_seq[-test_window:]
idx_test = seq_idx[-test_window:]

# ---------------- WALK-FORWARD SPLITS ----------------
splits = walk_forward_split(X_train_val, y_train_val, idx_train_val,
                            train_window=252*3, val_window=252)

# ---------------- TRAIN MULTIPLE MODELS ----------------
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

ensemble_preds_val = []
for X_train, y_train, X_val, y_val, idx_val in splits:
    split_preds = []
    for seed_model in range(NUM_MODELS):
        tf.random.set_seed(seed_model + SEED)
        np.random.seed(seed_model + SEED)

        model = Sequential([
            LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=(TIME_STEPS, n_features), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            LSTM(LSTM_UNITS[1], kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(2, activation="softmax", kernel_regularizer=l2(0.001))
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=[es, rlrop],
                  shuffle=False,
                  verbose=2)

        preds_val = model.predict(X_val, verbose=0)
        split_preds.append(preds_val)

    ensemble_preds_val.append(np.mean(split_preds, axis=0))

# ---------------- FINAL ENSEMBLE ON TRAIN+VAL ----------------
final_models = []
for seed_model in range(NUM_MODELS):
    tf.random.set_seed(seed_model + SEED)
    np.random.seed(seed_model + SEED)

    model = Sequential([
        LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=(TIME_STEPS, n_features), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS[1], kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(2, activation="softmax", kernel_regularizer=l2(0.001))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3)
    model.fit(X_train_val, y_train_val, epochs=max(1, EPOCHS // 3), batch_size=BATCH_SIZE,
              callbacks=[es, rlrop], shuffle=False, verbose=2)
    final_models.append(model)

# ---------------- SAVE MODELS & SCALER ----------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
version_dir = OUT_DIR / f"ensemble_{timestamp}"
version_dir.mkdir(parents=True, exist_ok=True)

# Save scaler
joblib.dump(scaler, version_dir / "scaler.gz")

# Save each model
model_paths = []
for i, model in enumerate(final_models):
    path = version_dir / f"model_{i}.h5"
    model.save(path)
    model_paths.append(str(path))

# Save metadata
meta = {
    "tickers": TICKERS,
    "features": FEATURES,
    "time_steps": TIME_STEPS,
    "test_len": len(X_test),
    "timestamp": timestamp,
    "models": model_paths
}
with open(version_dir / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# ---------------- PREDICT & BACKTEST ON TEST SET ----------------
ensemble_preds_test = np.mean([m.predict(X_test) for m in final_models], axis=0)
ensemble_labels_test = (ensemble_preds_test[:,1] > 0.5).astype(int)

rets0_test = returns[TICKERS[0]].loc[idx_test].values
rets1_test = returns[TICKERS[1]].loc[idx_test].values
strategy_rets_test = strategy_returns(ensemble_labels_test, rets0_test, rets1_test)
cum_test = np.cumprod(1 + strategy_rets_test)

# ---------------- METRICS ----------------
count1 = np.sum(ensemble_labels_test==1)
count0 = np.sum(ensemble_labels_test==0)
total = len(ensemble_labels_test)

results = {
    "ensemble_picks": {
        TICKERS[0]: int(count1),
        TICKERS[1]: int(count0),
        "percent": {TICKERS[0]: float(count1/total), TICKERS[1]: float(count0/total)}
    },
    "total_return": float(cum_test[-1]-1),
    "sharpe": float(sharpe_ratio(strategy_rets_test)),
    "max_drawdown": float(max_drawdown(cum_test))
}

# Save test results
with open(version_dir / "test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved ensemble to:", version_dir)
print("Test results:", results)

# ---------------- PLOT ----------------
plt.figure(figsize=(14,7))
plt.plot(idx_test, cum_test, label="Ensemble Strategy", linewidth=2)
plt.plot(idx_test, np.cumprod(1+rets0_test), label=f"{TICKERS[0]} Buy & Hold", linewidth=2)
plt.plot(idx_test, np.cumprod(1+rets1_test), label=f"{TICKERS[1]} Buy & Hold", linewidth=2)
plt.title("LSTM Pairs Trading: Test Set Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
