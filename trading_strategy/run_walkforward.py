from src.walkforward_tester import run_walkforward_validation
from src.utils import load_config
import pandas as pd
import os
import numpy as np
from pprint import pprint

if __name__ == "__main__":
    print("=== HWTradingStrategy: Walk-Forward Validation ===")

    cfg = load_config()

    # tickers a validar
    # si en config.yaml no tienes cfg["tickers"], puedes pasarlos manualmente
    tickers = cfg.get("tickers", ["NVDA", "AAPL", "TSLA"])

    lookback_days = cfg["data"]["lookback_days"]
    interval = cfg["data"]["interval"]

    # puedes ajustar estas ventanas según la granularidad que quieras
    results_df = run_walkforward_validation(
        tickers=tickers,
        lookback_days=lookback_days,
        interval=interval,
        cfg=cfg,
        train_size_bars=1200,  # tamaño bloque train
        test_size_bars=200,    # tamaño bloque test
        save_path="reports/walkforward_results.csv"
    )

    print("\n--- Métricas promedio por ticker ---")
    summary = (
        results_df
        .groupby("ticker")[["acc", "precision", "recall", "f1"]]
        .mean()
        .round(4)
    )

    print(summary)

    os.makedirs("reports", exist_ok=True)
    summary_path = "reports/walkforward_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nResumen guardado en {summary_path}")

    print("\nSi ves que un ticker tiene f1 o precision muy baja de manera consistente, ese ticker es inestable o está sobreajustado.")
