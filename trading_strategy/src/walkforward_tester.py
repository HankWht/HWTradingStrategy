import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.features import build_features
from src.labeler import make_labels
from src.data_loader import load_universe_data
from src.utils import load_config, ensure_dirs


def _train_single_model(df_train, feature_cols, model_params):
    """
    Entrena un modelo XGBoost en df_train usando feature_cols y devuelve el modelo.
    df_train debe contener 'label'.
    """
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

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
    return model


def _evaluate_model(model, df_test, feature_cols):
    """
    Evalúa el modelo en df_test y devuelve métricas.
    df_test debe contener 'label'.
    """
    if len(df_test) == 0:
        return {
            "acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    X_test = df_test[feature_cols].values
    y_true = df_test["label"].values

    y_pred = model.predict(X_test)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def _split_walkforward(df, train_size_bars, test_size_bars):
    """
    Genera ventanas walk-forward sobre df ordenado por tiempo.
    Retorna lista de tuplas: (df_train, df_test, start_time, end_time)
    """
    windows = []
    total = len(df)

    start_idx = 0
    while True:
        train_start = start_idx
        train_end = start_idx + train_size_bars
        test_end = train_end + test_size_bars

        if test_end > total:
            break

        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[train_end:test_end].copy()

        windows.append((
            df_train,
            df_test,
            df_train.index[0],
            df_test.index[-1],
        ))

        # avanzar ventana
        start_idx += test_size_bars

    return windows


def _prepare_featured_labeled(df_raw, cfg):
    """
    Toma el dataframe crudo (OHLCV intradía para un ticker),
    construye features y etiquetas iguales al flujo normal.
    Devuelve df con features + label + índice temporal.
    """
    feat_df = build_features(df_raw)

    labeled_df = make_labels(
        feat_df,
        horizon=cfg["strategy"]["horizon"],
        tp=cfg["strategy"]["tp"],
        sl=cfg["strategy"]["sl"]
    )

    # limpiamos filas con NaN en features o label
    labeled_df = labeled_df.dropna(subset=["label"])
    return labeled_df


def run_walkforward_validation(
    tickers,
    lookback_days,
    interval,
    cfg,
    train_size_bars=1200,
    test_size_bars=200,
    save_path="reports/walkforward_results.csv"
):
    """
    Entrena/evalúa en ventanas deslizantes cronológicas.
    - train_size_bars: cuántas velas 1m usas para entrenar cada bloque
    - test_size_bars: cuántas velas 1m pruebas justo después
    Devuelve dataframe con métricas por ticker y por ventana temporal.
    También guarda CSV en reports/.
    """

    ensure_dirs()
    os.makedirs("reports", exist_ok=True)

    all_results = []

    # 1. Descargamos datos crudos para todos los tickers
    raw_map = load_universe_data(
        tickers=tickers,
        lookback_days=lookback_days,
        interval=interval
    )

    feature_cols = cfg["features"]["include"]
    model_params = cfg["model"]

    for ticker, df_raw in raw_map.items():
        # 2. Construir features + etiquetas igual que en pipeline normal
        df_full = _prepare_featured_labeled(df_raw, cfg)

        # aseguramos que esté ordenado temporalmente
        df_full = df_full.sort_index()

        # 3. Generar las ventanas walk-forward
        windows = _split_walkforward(df_full, train_size_bars, test_size_bars)

        for (df_train, df_test, start_time, end_time) in windows:
            # 4. Entrenar un modelo SOLO en df_train
            model = _train_single_model(df_train, feature_cols, model_params)

            # 5. Evaluar en df_test
            metrics = _evaluate_model(model, df_test, feature_cols)

            result_row = {
                "ticker": ticker,
                "train_start": str(start_time),
                "test_end": str(end_time),
                "train_size": len(df_train),
                "test_size": len(df_test),
                "acc": metrics["acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            all_results.append(result_row)

    results_df = pd.DataFrame(all_results)

    # 6. Guardar resultados en CSV
    results_df.to_csv(save_path, index=False)

    print(f"[Walk-Forward] Resultados guardados en {save_path}")
    return results_df
