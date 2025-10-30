# ==========================================================
# HWTradingStrategy - Pipeline Principal
# ==========================================================
# Este script coordina todo el flujo:
# 1. Descarga datos y genera features
# 2. Crea etiquetas (TP/SL)
# 3. Entrena modelos por ticker
# 4. Genera señales actuales
# ==========================================================

import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils import load_config, ensure_dirs
from src.data_loader import load_universe_data
from src.features import build_features
from src.labeler import make_labels

# ==========================================================
# Inicialización
# ==========================================================
cfg = load_config()
ensure_dirs()

tickers = cfg.get("tickers", ["NVDA", "AAPL", "TSLA"])
lookback_days = cfg["data"]["lookback_days"]
interval = cfg["data"]["interval"]

feature_cols = cfg["features"]["include"]
model_params = cfg["model"]
strategy = cfg["strategy"]

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=== HWTradingStrategy: Pipeline Iniciado ===")
print(f"Tickers a procesar: {tickers}")
print(f"Intervalo: {interval} | Lookback: {lookback_days} días\n")

# ==========================================================
# Descarga y procesamiento de datos
# ==========================================================
print("Descargando datos...")
raw_data = load_universe_data(
    tickers=tickers,
    lookback_days=lookback_days,
    interval=interval
)

signals = []

for ticker, df_raw in raw_data.items():
    print(f"\nProcesando {ticker} ...")

    # Construir features
    df_feat = build_features(df_raw)

    # Generar etiquetas (TP/SL)
    df_labeled = make_labels(
        df_feat,
        horizon=strategy["horizon"],
        tp=strategy["tp"],
        sl=strategy["sl"]
    )

    # Limpieza de NaN
    df_labeled.dropna(subset=feature_cols + ["label"], inplace=True)
    if df_labeled.empty:
        print(f"⚠️ No hay suficientes datos procesados para {ticker}.")
        continue

    # Separación train/test
    X = df_labeled[feature_cols].values
    y = df_labeled["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # Entrenar modelo
    model = XGBClassifier(
        n_estimators=model_params["n_estimators"],
        learning_rate=model_params["learning_rate"],
        max_depth=model_params["max_depth"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        random_state=model_params["random_state"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Guardar modelo
    model_path = f"models/{ticker}_model.pkl"
    joblib.dump(model, model_path)

    # Predicciones sobre el último punto
    latest_features = df_labeled[feature_cols].iloc[-1:].values
    proba = model.predict_proba(latest_features)[0][1]
    price_now = df_raw["Close"].iloc[-1]

    # Definir señal
    signal = "BUY" if proba > strategy["threshold"] else "HOLD"
    confidence = (
        "Alta" if proba >= 0.7 else
        "Media" if proba >= 0.55 else
        "Baja"
    )

    signals.append({
        "ticker": ticker,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "probabilidad": round(proba, 4),
        "señal": signal,
        "confianza": confidence,
        "precio_actual": round(price_now, 2),
        "TP_target_%": strategy["tp"],
        "SL_stop_%": strategy["sl"]
    })

# ==========================================================
# Guardar señales generadas
# ==========================================================
if signals:
    df_signals = pd.DataFrame(signals)
    df_signals.to_csv("reports/signals_latest.csv", index=False)
    print("\n✅ Señales generadas correctamente en: reports/signals_latest.csv\n")
else:
    print("\n⚠️ No se generaron señales. Revisa datos o configuración.\n")

print("=== Pipeline completado con éxito ===")
