# VWAP y distancia a VWAP
# above_open / dist_to_open / day_return
# volumen relativo y cumvol_ratio
# breakout strength
# volatilidad y ATR
# pullback_ratio
# tick_momentum
# body_ratio
# RSI
# slope de precio y EMA slope
# gap_open
# vwap_delta
# reversion_score

import numpy as np
import pandas as pd

def _rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.to_frame("a").join(high_close.to_frame("b")).join(low_close.to_frame("c")).max(axis=1)
    return tr.rolling(period).mean()

def _slope(series):
    # pendiente lineal normalizada
    y = np.array(series)
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0
    m = np.polyfit(x, y, 1)[0]
    # normalizamos por último valor para que sea comparable
    return m / (y[-1] + 1e-9)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out.index.date

    # VWAP intradía acumulado
    out["pv"] = out["close"] * out["volume"]
    out["cum_pv"] = out.groupby("date")["pv"].cumsum()
    out["cum_vol"] = out.groupby("date")["volume"].cumsum()
    out["vwap"] = out["cum_pv"] / out["cum_vol"]

    # Distancia a vwap
    out["distance_to_vwap"] = (out["close"] - out["vwap"]) / out["vwap"]

    # Apertura diaria / sesgo direccional del día
    out["open_day"] = out.groupby("date")["open"].transform("first")
    out["above_open"] = (out["close"] > out["open_day"]).astype(int)
    out["dist_to_open"] = (out["close"] - out["open_day"]) / out["open_day"]
    out["day_return"] = out["dist_to_open"]

    # Volumen relativo
    out["volume_rel"] = out["volume"] / out["volume"].rolling(20).mean()

    # Volumen acumulado del día vs promedio largo (proxy de flujo anormal)
    out["cumvol_day"] = out.groupby("date")["volume"].cumsum()
    out["cumvol_ratio"] = out["cumvol_day"] / out["volume"].rolling(390).mean()

    # Breakout strength
    out["high_10"] = out["high"].rolling(10).max()
    out["high_breakout_strength"] = out["close"] / out["high_10"]

    # Volatilidad relativa instantánea
    out["range_now"] = out["high"] - out["low"]
    out["range_std20"] = out["close"].rolling(20).std()
    out["volatility_ratio"] = out["range_now"] / (out["range_std20"] + 1e-9)

    # Pullback ratio: dónde está el close dentro del mini rango reciente
    win_high5 = out["high"].rolling(5).max()
    win_low5  = out["low"].rolling(5).min()
    out["pullback_ratio"] = (out["close"] - win_low5) / (win_high5 - win_low5 + 1e-9)

    # Tick momentum: cuántas velas verdes vs rojas en 10 pasos
    out["tick_dir"] = np.where(out["close"].diff() > 0, 1, -1)
    out["tick_momentum"] = out["tick_dir"].rolling(10).sum()

    # Candle body / range conviction
    out["body_ratio"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"] + 1e-9)

    # RSI y ATR
    out["rsi_14"] = _rsi(out["close"], 14)
    out["atr_14"] = _atr(out, 14)

    # EMA y su pendiente
    out["ema_20"] = out["close"].ewm(span=20).mean()
    out["ema_slope"] = out["ema_20"].diff()
    out["ema_above_vwap"] = (out["ema_20"] > out["vwap"]).astype(int)

    # Gap de apertura vs cierre previo
    prev_close = out["close"].shift(1)
    out["gap_open"] = (out["open"] - prev_close) / (prev_close + 1e-9)

    # VWAP delta (cambio en vwap)
    out["vwap_delta"] = out["vwap"] - out["vwap"].shift(10)

    # Reversion score (z-score del precio vs su media 10)
    out["reversion_score"] = (
        (out["close"] - out["close"].rolling(10).mean()) /
        (out["close"].rolling(10).std() + 1e-9)
    )

    # price_slope sobre ventana móvil
    out["price_slope"] = out["close"].rolling(10).apply(_slope, raw=False)

    # limpieza
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out
