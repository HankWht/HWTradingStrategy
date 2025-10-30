import optuna
import yaml
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.backtester_multi import run_full_backtest
from src.utils import load_config, save_yaml


def evaluate_score(stats):
    """
    Calcula un score compuesto basado en múltiples métricas.
    """
    total_return = stats.get("total_return_pct", 0)
    win_rate = stats.get("win_rate_pct", 0)
    drawdown = abs(stats.get("max_drawdown_pct", 0))
    profit_factor = stats.get("profit_factor", 1)

    # Fórmula ajustada de score:
    # Alta ponderación a retorno y winrate, penalización por drawdown
    score = (0.5 * total_return) + (0.3 * win_rate) + (0.2 * profit_factor * 10) - (0.4 * drawdown)
    return round(score, 3)


def optimize_strategy(ticker_list, n_trials=25, reuse_study=True):
    """
    Optimiza automáticamente los parámetros TP, SL y threshold
    para varios tickers usando Optuna y el motor de backtesting.
    """

    cfg = load_config()
    os.makedirs("reports", exist_ok=True)
    results = {}

    for ticker in ticker_list:
        print(f"\n=== Optimizando estrategia para {ticker} ===")

        study_name = f"optuna_{ticker}"
        storage_url = f"sqlite:///reports/{study_name}.db"

        # Reutiliza estudio anterior si existe
        if reuse_study and os.path.exists(f"reports/{study_name}.db"):
            print(f"Reanudando estudio previo: {study_name}.db")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
        else:
            study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_url)

        def objective(trial):
            tp = trial.suggest_float("tp", 0.3, 1.5)
            sl = trial.suggest_float("sl", 0.1, 0.8)
            threshold = trial.suggest_float("threshold", 0.5, 0.8)

            cfg["strategy"]["tp"] = tp
            cfg["strategy"]["sl"] = sl
            cfg["strategy"]["prob_threshold"] = threshold

            stats = run_full_backtest([ticker], cfg=cfg, silent=True)
            return evaluate_score(stats)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        # Guardar datos en CSV
        csv_path = f"reports/optuna_results_{ticker}.csv"
        trials_df.to_csv(csv_path, index=False)

        results[ticker] = {
            "best_params": best_params,
            "score": round(best_value, 2),
            "last_optimized": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results_file": csv_path
        }

        print(f"✅ {ticker}: mejor configuración {best_params} (score={best_value:.2f})")
        print(f"Resultados completos guardados en {csv_path}")

    # Guardar YAML con los mejores parámetros globales
    save_yaml(results, "reports/strategy_optimization.yaml")
    print("\nResultados globales guardados en reports/strategy_optimization.yaml")
    return results
