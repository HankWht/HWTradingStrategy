import numpy as np
import pandas as pd

def make_labels(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> pd.DataFrame:
    out = df.copy()
    closes = out["close"].values
    labels = []

    for i in range(len(out)):
        future = closes[i : i + horizon]
        if len(future) < horizon:
            labels.append(np.nan)
            continue

        entry_price = closes[i]
        hit_tp = False
        hit_sl = False

        for p in future:
            move_pct = (p - entry_price) / entry_price
            if move_pct >= tp:
                hit_tp = True
                break
            if move_pct <= -sl:
                hit_sl = True
                break

        if hit_tp and not hit_sl:
            labels.append(1)
        else:
            labels.append(0)

    out["label"] = labels
    out = out.dropna().copy()
    return out