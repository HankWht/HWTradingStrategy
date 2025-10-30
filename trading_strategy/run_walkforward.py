from src.walkforward_tester import run_walkforward_validation
from src.utils import load_config
import os
import pandas as pd

if __name__ == "__main__":
    print("=== HWTradingStrategy: Validación Walk-Forward ===")

    cfg = load_config()
    tickers = cfg.get("tickers", ["NVDA", "AAPL", "TSLA"])
    lookback_days = cfg["data"]["lookback_days"]
    interval = cfg["data"]["interval"]

    df = run_walkforward_validation(
        tickers=tickers,
        lookback_days=lookback_days,
        interval=interval,
        cfg=cfg,
        train_size=1200,
        test_size=200
    )

    summary = df.groupby("ticker")[["acc", "precision", "recall", "f1"]].mean().round(4)
    os.makedirs("reports", exist_ok=True)
    summary.to_csv("reports/walkforward_summary.csv")
    print("\n✅ Resumen guardado en reports/walkforward_summary.csv")
    print(summary)
    print("=== Validación Walk-Forward Completada ===")