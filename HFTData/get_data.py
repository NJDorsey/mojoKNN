from tiingo import TiingoClient
import pandas as pd

# Initialize client (use your real key – get one free at tiingo.com)
config = {
    'api_key': 'cb5f8ef1e686b37a79dd8b3a0878a1572aa63b43',
    'session': True  # recommended for connection pooling
}
client = TiingoClient(config)

# Fetch 2 years of 1-minute OHLCV data
# Adjust dates as needed; current date is ~Feb 2026, so back ~2 years
df = client.get_dataframe(
    ticker='AAPL',
    startDate='2024-02-13',   # or whatever exact start you want
    endDate='2026-02-13',
    frequency='1min',          # 1-minute bars
    fmt='json'                 # default; use 'csv' if you want raw faster response (but json is fine)
    # Optional: columns='open,high,low,close,volume' to subset if needed
)

# Quick checks (optional but useful)
print(df.head())               # preview first few rows
print(df.tail())               # preview last few
print(f"Shape: {df.shape}")    # should be ~195,000–200,000 rows for 2 full years

# Save to CSV – this is the key step you asked for
df.to_csv('AAPL_LONG.csv', index=True)  # index=True keeps the datetime as first column

print("Data saved to AAPL_LONG.csv")