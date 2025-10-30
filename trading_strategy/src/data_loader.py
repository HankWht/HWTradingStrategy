import yfinance as yf
import pandas as pd

def download_intraday(ticker: str, lookback_days: int, interval: str = "1m"):
    df = yf.download(
        tickers=ticker,
        period=f"{lookback_days}d",
        interval=interval,
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=True,
    )

    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low",
        "Close":"close","Volume":"volume"
    })

    df = df.dropna().copy()

    # quitar zona horaria para evitar choques en merges / backtest
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df

def load_universe_data(tickers, lookback_days, interval="1m"):
    data_map = {}
    for t in tickers:
        df = download_intraday(t, lookback_days, interval)
        data_map[t] = df
    return data_map
