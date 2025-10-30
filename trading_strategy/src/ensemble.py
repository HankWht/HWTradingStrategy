import joblib
import pandas as pd

def get_latest_signal_for_ticker(df, model, feature_cols):
    last_row = df.iloc[[-1]]  # mantiene DataFrame
    prob = model.predict_proba(last_row[feature_cols])[:,1][0]
    entry_price = last_row["close"].iloc[0]
    day_ret = last_row["day_return"].iloc[0]
    return {
        "ticker": df.get("ticker_name", "UNKNOWN"),
        "prob": float(prob),
        "price": float(entry_price),
        "day_return": float(day_ret)
    }

def generate_ensemble_signals(
    data_map,            # dict: { "NVDA": df_nvda_con_features_y_label, ... }
    feature_cols,
    model_dir="models",
    base_threshold=0.6
):
    rows = []
    for ticker, df in data_map.items():
        model_path = f"{model_dir}/{ticker}_model.pkl"
        try:
            model = joblib.load(model_path)
        except:
            continue

        df_local = df.copy()
        df_local["ticker_name"] = ticker
        sig = get_latest_signal_for_ticker(df_local, model, feature_cols)

        # Clasificación manual de señal:
        if sig["prob"] > base_threshold + 0.1:
            signal = "BUY"
            conf = "Alta"
        elif sig["prob"] > base_threshold:
            signal = "WATCH"
            conf = "Media"
        else:
            signal = "HOLD"
            conf = "Baja"

        rows.append({
            "Ticker": ticker,
            "Probabilidad": round(sig["prob"], 3),
            "Señal": signal,
            "Confianza": conf,
            "Precio actual": round(sig["price"], 2),
            "day_return": round(sig["day_return"], 4)
        })

    # rankear por probabilidad
    out_df = pd.DataFrame(rows).sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    return out_df
