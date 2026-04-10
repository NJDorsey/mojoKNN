#!/usr/bin/env python3
"""
Stock Data Download & Feature Engineering Pipeline
====================================================
Downloads 1-minute OHLCV data from Tiingo for specified tickers,
engineers 16 causal features, and produces:
  - {TICKER}_LONG_features_causal.csv   (N rows x 16 columns, no header)
  - {TICKER}_LONG_target_causal.csv     (N rows x 1 column, no header)

These match the format of the existing AAPL_LONG_features_causal.csv.

Usage:
    pixi run python3 HFTData/download_and_engineer.py          # downloads WMT, CPRI, JPM
    pixi run python3 HFTData/download_and_engineer.py AAPL     # re-engineer AAPL only
    pixi run python3 HFTData/download_and_engineer.py --all     # all 4 tickers
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIINGO_API_KEY = "cb5f8ef1e686b37a79dd8b3a0878a1572aa63b43"
START_DATE = "2024-02-13"
END_DATE   = "2026-02-13"
DEFAULT_TICKERS = ["WMT", "CPRI", "JPM"]   # AAPL already done
ALL_TICKERS     = ["AAPL", "WMT", "CPRI", "JPM"]

# Output directory (same directory as this script)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_ticker(ticker: str) -> pd.DataFrame:
    """Download 1-minute OHLCV data from Tiingo IEX endpoint."""
    from tiingo import TiingoClient

    config = {"api_key": TIINGO_API_KEY, "session": True}
    client = TiingoClient(config)

    print(f"  Downloading {ticker} 1-min data ({START_DATE} to {END_DATE})...")
    df = client.get_dataframe(
        ticker=ticker,
        startDate=START_DATE,
        endDate=END_DATE,
        frequency="1min",
        fmt="json",
    )

    # Save raw CSV for reference
    raw_path = os.path.join(OUT_DIR, f"{ticker}_LONG.csv")
    df.to_csv(raw_path, index=True)
    print(f"  Raw data saved: {raw_path}  ({len(df)} rows)")
    return df


def load_raw_csv(ticker: str) -> pd.DataFrame:
    """Load an already-downloaded raw CSV if it exists."""
    raw_path = os.path.join(OUT_DIR, f"{ticker}_LONG.csv")
    if not os.path.exists(raw_path):
        return None
    print(f"  Loading cached raw data: {raw_path}")
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering  (16 causal features from OHLCV)
# ---------------------------------------------------------------------------
# "Causal" means each feature at time t uses only data from t and earlier.
# No future leakage.
#
# Features (16 total):
#  0: close_return_1       — 1-bar log return of close
#  1: high_return_1        — 1-bar log return of high
#  2: low_return_1         — 1-bar log return of low
#  3: open_return_1        — 1-bar log return of open
#  4: log_volume           — log(volume + 1)
#  5: volatility_5         — 5-bar rolling std of close returns
#  6: volatility_20        — 20-bar rolling std of close returns
#  7: ma_ratio_5           — close / 5-bar SMA - 1
#  8: ma_ratio_20          — close / 20-bar SMA - 1
#  9: rsi_14               — 14-bar RSI (scaled 0-100)
# 10: high_low_range       — (high - low) / close
# 11: volume_ma_ratio      — volume / 20-bar MA volume
# 12: cumulative_return_10 — 10-bar cumulative log return
# 13: cumulative_return_50 — 50-bar cumulative log return
# 14: momentum_5           — 5-bar price momentum (pct change)
# 15: trade_intensity      — log(notional_value + 1) if available, else log(close * volume + 1)
#
# Target:
#   +1.0 if close[t+1] > close[t], else -1.0
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) without future leakage."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Engineer 16 causal features from OHLCV data.

    Tiingo IEX columns may include:
      close, high, low, open, volume
    or market-prefixed variants:
      marketClose, marketHigh, marketLow, marketOpen, marketVolume, marketNotional

    Returns (features_df, target_series) after dropping NaN rows.
    """
    # Normalize column names — handle both Tiingo formats
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "close" in lc and "change" not in lc:
            col_map["close"] = col
        elif "high" in lc:
            col_map["high"] = col
        elif "low" in lc:
            col_map["low"] = col
        elif "open" in lc:
            col_map["open"] = col
        elif "volume" in lc:
            col_map["volume"] = col
        elif "notional" in lc:
            col_map["notional"] = col

    close  = df[col_map["close"]].astype(float)
    high   = df[col_map["high"]].astype(float)
    low    = df[col_map["low"]].astype(float)
    opn    = df[col_map["open"]].astype(float)
    volume = df[col_map["volume"]].astype(float)

    has_notional = "notional" in col_map
    if has_notional:
        notional = df[col_map["notional"]].astype(float)

    # --- Log returns (1-bar) ---
    close_ret = np.log(close / close.shift(1))
    high_ret  = np.log(high / high.shift(1))
    low_ret   = np.log(low / low.shift(1))
    open_ret  = np.log(opn / opn.shift(1))

    features = pd.DataFrame(index=df.index)

    # 0-3: 1-bar log returns
    features["close_return_1"]  = close_ret
    features["high_return_1"]   = high_ret
    features["low_return_1"]    = low_ret
    features["open_return_1"]   = open_ret

    # 4: log volume
    features["log_volume"] = np.log(volume + 1)

    # 5-6: rolling volatility
    features["volatility_5"]  = close_ret.rolling(5, min_periods=5).std()
    features["volatility_20"] = close_ret.rolling(20, min_periods=20).std()

    # 7-8: MA ratios
    ma5  = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    features["ma_ratio_5"]  = close / ma5 - 1.0
    features["ma_ratio_20"] = close / ma20 - 1.0

    # 9: RSI
    features["rsi_14"] = compute_rsi(close, period=14)

    # 10: high-low range normalized by close
    features["high_low_range"] = (high - low) / (close + 1e-10)

    # 11: volume / 20-bar MA volume
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    features["volume_ma_ratio"] = volume / (vol_ma20 + 1e-10)

    # 12-13: cumulative returns
    features["cumulative_return_10"] = close_ret.rolling(10, min_periods=10).sum()
    features["cumulative_return_50"] = close_ret.rolling(50, min_periods=50).sum()

    # 14: 5-bar momentum
    features["momentum_5"] = close.pct_change(periods=5)

    # 15: trade intensity
    if has_notional:
        features["trade_intensity"] = np.log(notional.abs() + 1)
    else:
        features["trade_intensity"] = np.log(close * volume + 1)

    # --- Target: next-bar direction ---
    target = np.sign(close.shift(-1) - close)
    target = target.replace(0, 1.0)   # ties → positive

    # Drop rows with NaN (from rolling windows + the last row with no target)
    valid = features.notna().all(axis=1) & target.notna()
    features = features[valid]
    target   = target[valid]

    print(f"  Features: {features.shape[0]} rows x {features.shape[1]} columns")
    return features, target


# ---------------------------------------------------------------------------
# Standardize features (z-score, fit on full dataset to match AAPL pipeline)
# ---------------------------------------------------------------------------

def standardize(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardization per feature column."""
    return (features - features.mean()) / (features.std() + 1e-10)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_features(ticker: str, features: pd.DataFrame, target: pd.Series):
    """Save feature matrix and target vector as headerless CSVs."""
    feat_path   = os.path.join(OUT_DIR, f"{ticker}_LONG_features_causal.csv")
    target_path = os.path.join(OUT_DIR, f"{ticker}_LONG_target_causal.csv")

    features.to_csv(feat_path,   index=False, header=False, float_format="%.8g")
    target.to_csv(target_path,   index=False, header=False, float_format="%.0f")

    print(f"  Saved: {feat_path}  ({features.shape[0]} x {features.shape[1]})")
    print(f"  Saved: {target_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_ticker(ticker: str, force_download: bool = False):
    """Download (if needed) and engineer features for a single ticker."""
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")

    # Try loading cached raw data first
    df = None if force_download else load_raw_csv(ticker)
    if df is None:
        df = download_ticker(ticker)

    # Engineer features
    features, target = engineer_features(df)

    # Standardize
    features = standardize(features)

    # Save
    save_features(ticker, features, target)

    return len(features)


def main():
    # Parse CLI args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            tickers = ALL_TICKERS
        else:
            tickers = sys.argv[1:]
    else:
        tickers = DEFAULT_TICKERS

    force_download = "--force" in sys.argv
    tickers = [t for t in tickers if t != "--force"]

    print("Stock Data Download & Feature Engineering Pipeline")
    print(f"Tickers: {tickers}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output dir: {OUT_DIR}")

    summary = {}
    for ticker in tickers:
        try:
            n_rows = process_ticker(ticker, force_download=force_download)
            summary[ticker] = n_rows
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            import traceback; traceback.print_exc()
            summary[ticker] = "FAILED"

    print(f"\n{'='*60}")
    print("Summary:")
    for ticker, n in summary.items():
        print(f"  {ticker}: {n} rows" if isinstance(n, int) else f"  {ticker}: {n}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
