import logging
import pprint
import os

from src.utils import load_config, ensure_dirs
from src.data_loader import load_universe_data
from src.features import build_features
from src.labeler import make_labels
from src.backtester_multi import run_full_backtest

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = load_config()
    ensure_dirs()

    # Leer tickers del universo
    with open("data/tickers.txt") as f:
        tickers = [line.strip() for line in f if line.strip()]

    logging.info(f"Backtest para tickers: {tickers}")

    # 1. Descargar datos crudos
    raw_map = load_universe_data(
        tickers,
        cfg["data"]["lookback_days"],
        cfg["data"]["interval"]
    )

    # 2. Generar features y etiquetas igual que el pipeline
    data_map = {}
    for t, df in raw_map.items():
        feat_df = build_features(df)
        labeled_df = make_labels(
            feat_df,
            cfg["strategy"]["horizon"],
            cfg["strategy"]["tp"],
            cfg["strategy"]["sl"]
        )
        data_map[t] = labeled_df

    feature_cols = cfg["features"]["include"]

    # 3. Correr el motor de backtest multi-asset
    equity_result, summary = run_full_backtest(
        data_map=data_map,
        feature_cols=feature_cols,
        model_dir="models",
        horizon=cfg["strategy"]["horizon"],
        tp=cfg["strategy"]["tp"],
        sl=cfg["strategy"]["sl"],
        base_prob_threshold=cfg["strategy"]["base_prob_threshold"],
        initial_capital=cfg["strategy"]["capital"],
        risk_fraction=cfg["strategy"]["risk_fraction"]
    )

    # 4. Mostrar KPIs en consola
    logging.info("===== RESUMEN BACKTEST =====")
    pprint.pprint(summary)

    # 5. Guardar resultados para el dashboard
    os.makedirs("reports", exist_ok=True)
    equity_result["equity_curve"].to_csv("reports/equity_curve.csv", index=False)
    equity_result["trades_enriched"].to_csv("reports/trades_log.csv", index=False)

    logging.info("Equity curve guardada en reports/equity_curve.csv")
    logging.info("Log de trades guardado en reports/trades_log.csv")

if __name__ == "__main__":
    main()
