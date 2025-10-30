from src.optimizer import optimize_strategy
from src.utils import load_config
from datetime import datetime
import time

if __name__ == "__main__":
    print("=== HWTradingStrategy: Optimización Automática ===")
    start_time = time.time()

    cfg = load_config()
    tickers = cfg.get("tickers", ["NVDA", "AAPL", "TSLA"])

    print(f"\nOptimizando {len(tickers)} activos...")
    results = optimize_strategy(ticker_list=tickers, n_trials=30, reuse_study=True)

    print("\n--- Mejores configuraciones encontradas ---")
    for t, res in results.items():
        print(f"{t}: {res['best_params']} -> score={res['score']} (archivo: {res['results_file']})")

    print(f"\nDuración total: {round((time.time() - start_time)/60, 2)} minutos")
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
