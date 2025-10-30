import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from src.features import build_features
from src.labeler import make_labels
from src.data_loader import load_universe_data
from src.utils import load_config, ensure_dirs

def _train_single_model(df_train, feature_cols, model_params):
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    model = XGBClassifier(**model_params, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def _evaluate_model(model, df_test, feature_cols):
    if len(df_test) == 0:
        return {"acc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
    X_test = df_test[feature_cols].values
    y_true = df_test["label"].values
    y_pred = model.predict(X_test)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def _split_walkforward(df, train_size, test_size):
    windows = []
    total = len(df)
    start = 0
    while start + train_size + test_size <= total:
        df_train = df.iloc[start:start + train_size].copy()
        df_test = df.iloc[start + train_size:start + train_size + test_size].copy()
        windows.append((df_train, df_test, df_train.index[0], df_test.index[-1]))
        start += test_size
    return windows

def run_walkforward_validation(tickers, lookback_days, interval, cfg, train_size=1200, test_size=200):
    ensure_dirs()
    os.makedirs("reports", exist_ok=True)
    all_results = []
    raw_map = load_universe_data(tickers, lookback_days, interval)
    features = cfg["features"]["include"]
    params = cfg["model"]

    for ticker, df_raw in raw_map.items():
        df_feat = build_features(df_raw)
        df_labeled = make_labels(df_feat, horizon=cfg["strategy"]["horizon"], tp=cfg["strategy"]["tp"], sl=cfg["strategy"]["sl"])
        df_labeled.dropna(subset=["label"], inplace=True)
        df_labeled = df_labeled.sort_index()

        windows = _split_walkforward(df_labeled, train_size, test_size)
        for df_train, df_test, start, end in windows:
            model = _train_single_model(df_train, features, params)
            metrics = _evaluate_model(model, df_test, features)
            all_results.append({
                "ticker": ticker,
                "train_start": str(start),
                "test_end": str(end),
                "train_size": len(df_train),
                "test_size": len(df_test),
                **metrics
            })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("reports/walkforward_results.csv", index=False)
    summary = df_results.groupby("ticker")[["acc", "precision", "recall", "f1"]].mean().round(4)
    summary.to_csv("reports/walkforward_summary.csv")
    print("✅ Walk-forward completado. Resultados guardados en /reports/")
    return df_results
