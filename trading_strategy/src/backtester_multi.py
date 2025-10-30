import numpy as np
import pandas as pd
import joblib
from typing import Dict, List


# =========================
# Helpers de estrategia
# =========================

def dynamic_threshold(day_return: float, base_threshold: float = 0.6) -> float:
    """
    Ajusta el umbral de activación de señal según el sesgo del día.
    - Día extremadamente bullish -> pedimos más convicción.
    - Día muerto -> aceptamos señales más débiles.
    """
    if day_return > 0.02:      # +2% intradía es euforia
        return max(base_threshold, 0.7)
    elif day_return < 0.005:   # <0.5% es día lento
        return min(base_threshold, 0.55)
    return base_threshold


def _simulate_trade_window(
    future_prices: np.ndarray,
    entry_price: float,
    tp: float,
    sl: float
):
    """
    Simula qué pasa con UNA posición abierta:
    - TP (+0.7%) o SL (-0.25%) primero
    - si no toca ninguno, salida al final del horizonte
    Devuelve (exit_price, exit_reason)
    """
    hit_tp = False
    hit_sl = False
    exit_price = future_prices[-1]
    exit_reason = "timeout"

    for p in future_prices:
        move_pct = (p - entry_price) / entry_price
        if move_pct >= tp:
            hit_tp = True
            exit_price = p
            exit_reason = "tp"
            break
        if move_pct <= -sl:
            hit_sl = True
            exit_price = p
            exit_reason = "sl"
            break

    return exit_price, exit_reason


# =========================
# Simulación por ticker
# =========================

def simulate_trades_for_ticker(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    horizon: int,
    tp: float,
    sl: float,
    base_prob_threshold: float,
) -> pd.DataFrame:
    """
    Genera TODAS las operaciones simuladas para un solo ticker.
    - df: dataframe ya con features y label (output de build_features + make_labels)
    - model: modelo entrenado para ese ticker (XGBoost cargado de /models)
    - feature_cols: lista de columnas que el modelo espera como input
    - horizon: minutos a mantener máx la posición
    - tp/sl: niveles de take profit / stop loss (% como decimal)
    - base_prob_threshold: umbral inicial (ej. 0.6)

    Retorna DataFrame de trades con pnl_pct y timestamps.
    """

    df_local = df.copy()

    # sanity: necesitamos estas columnas
    required_cols = ["close", "day_return"]
    for col in required_cols:
        if col not in df_local.columns:
            raise ValueError(f"simulate_trades_for_ticker: falta la columna '{col}' en df")

    # Paso 1: probabilidad histórica por minuto
    probs = model.predict_proba(df_local[feature_cols])[:, 1]
    df_local["pred_prob"] = probs

    trades = []

    # Paso 2: recorrer cada punto temporal como posible entrada
    last_valid_index = len(df_local) - horizon
    for i in range(last_valid_index):
        row = df_local.iloc[i]
        prob = row["pred_prob"]
        entry_price = row["close"]
        day_bias = row["day_return"]
        ts_entry = df_local.index[i]

        # threshold dinámico basado en sesgo del día
        th = dynamic_threshold(day_bias, base_prob_threshold)

        # no hay trade si la confianza no rompe el umbral
        if prob < th:
            continue

        # ventana futura de horizonte 'horizon' minutos
        fut_slice = df_local.iloc[i : i + horizon]
        future_prices = fut_slice["close"].values
        ts_exit = fut_slice.index[-1]

        exit_price, exit_reason = _simulate_trade_window(
            future_prices=future_prices,
            entry_price=entry_price,
            tp=tp,
            sl=sl
        )

        pnl_pct = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": df_local.get("ticker_name", "UNKNOWN"),
            "entry_time": ts_entry,
            "exit_time": ts_exit,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pred_prob": prob,
            "threshold_used": th,
            "exit_reason": exit_reason,
            "pnl_pct": pnl_pct,
        })

    if not trades:
        return pd.DataFrame(columns=[
            "ticker","entry_time","exit_time","entry_price","exit_price",
            "pred_prob","threshold_used","exit_reason","pnl_pct"
        ])

    return pd.DataFrame(trades)


# =========================
# Curva de capital / métrica global
# =========================

def build_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    risk_fraction: float
):
    """
    Construye la curva de equity aplicando sizing y riesgo.
    Asume ejecución secuencial (no simultánea) en orden temporal.
    Cada trade arriesga ~risk_fraction del capital actual.
    Retorna:
      - equity_curve (capital vs tiempo)
      - trades_realizados (con PnL en USD)
      - métricas de riesgo básicas
    """
    if trades_df.empty:
        return {
            "equity_curve": pd.DataFrame(columns=["time","capital"]),
            "trades_enriched": pd.DataFrame(columns=list(trades_df.columns)+["pnl_usd"]),
            "final_capital": initial_capital,
            "max_drawdown_pct": 0.0
        }

    # Ordenar cronológicamente
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    capital = initial_capital
    peak_capital = initial_capital
    equity_points = []
    realized_rows = []
    drawdown_list = []

    for _, trade in trades_df.iterrows():
        # capital asignado/riesgado en este trade
        stake = capital * risk_fraction

        pnl_usd = stake * trade["pnl_pct"]
        capital = capital + pnl_usd

        # track curva
        equity_points.append({
            "time": trade["exit_time"],
            "capital": capital
        })

        # track drawdown
        if capital > peak_capital:
            peak_capital = capital
        dd = (capital - peak_capital) / peak_capital
        drawdown_list.append(dd)

        # enriquecer fila con PnL en USD y capital post-trade
        row_copy = trade.to_dict()
        row_copy["pnl_usd"] = pnl_usd
        row_copy["capital_after"] = capital
        realized_rows.append(row_copy)

    equity_curve = pd.DataFrame(equity_points)
    trades_enriched = pd.DataFrame(realized_rows)

    max_drawdown_pct = (min(drawdown_list) * 100.0) if drawdown_list else 0.0

    return {
        "equity_curve": equity_curve,
        "trades_enriched": trades_enriched,
        "final_capital": capital,
        "max_drawdown_pct": max_drawdown_pct
    }


def summarize_performance(
    equity_result: dict,
    initial_capital: float
):
    """
    Genera KPIs globales:
    - win rate
    - profit factor
    - sharpe-like
    - retorno total
    - drawdown máximo
    - PnL por ticker
    """
    trades = equity_result["trades_enriched"]
    final_capital = equity_result["final_capital"]
    max_dd = equity_result["max_drawdown_pct"]

    if trades.empty:
        return {
            "final_capital": final_capital,
            "total_return_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "avg_trade_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown_pct": max_dd,
            "by_ticker": {}
        }

    # win rate
    wins = trades[trades["pnl_pct"] > 0]
    win_rate_pct = 100.0 * (len(wins) / len(trades))

    # promedio por trade (%)
    avg_trade_pct = trades["pnl_pct"].mean() * 100.0

    # profit factor
    total_win = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"].sum()
    total_loss = trades.loc[trades["pnl_pct"] <= 0, "pnl_pct"].sum()
    profit_factor = (total_win / abs(total_loss)) if total_loss < 0 else np.inf

    # sharpe-like (por trade)
    ret_series = trades["pnl_pct"]
    mean_ret = ret_series.mean()
    std_ret = ret_series.std() + 1e-9
    sharpe_like = (mean_ret / std_ret) * np.sqrt(len(trades))

    # retorno total %
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100.0

    # ranking por ticker
    by_ticker = (
        trades.groupby("ticker")["pnl_usd"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    return {
        "final_capital": round(final_capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "num_trades": int(len(trades)),
        "win_rate_pct": round(win_rate_pct, 2),
        "avg_trade_pct": round(avg_trade_pct, 3),
        "profit_factor": round(profit_factor, 3),
        "sharpe_like": round(sharpe_like, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "by_ticker": by_ticker
    }


# =========================
# Orquestador principal
# =========================

def run_full_backtest(
    data_map: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    model_dir: str,
    horizon: int,
    tp: float,
    sl: float,
    base_prob_threshold: float,
    initial_capital: float,
    risk_fraction: float
):
    """
    data_map:
        {
           "NVDA": df_nvda,   # df con features, labels, etc
           "AAPL": df_aapl,
           ...
        }

    Cada df debe ya contener las columnas de features +:
      - 'close'
      - 'day_return'
    Y su index debe ser timestamps cronológicos.

    Devuelve:
      - results_dict (equity_curve, trades_enriched, final_capital,...)
      - summary_dict (métricas resumidas)
    """

    all_trades = []

    # 1. simular trades por cada ticker usando SU modelo entrenado
    for ticker, df in data_map.items():
        model_path = f"{model_dir}/{ticker}_model.pkl"
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            # si no hay modelo guardado para este ticker, lo saltamos
            continue

        df_local = df.copy()
        df_local["ticker_name"] = ticker

        ticker_trades = simulate_trades_for_ticker(
            df=df_local,
            model=model,
            feature_cols=feature_cols,
            horizon=horizon,
            tp=tp,
            sl=sl,
            base_prob_threshold=base_prob_threshold
        )

        if not ticker_trades.empty:
            all_trades.append(ticker_trades)

    if not all_trades:
        # no hubo trades en ningún ticker
        empty_equity = {
            "equity_curve": pd.DataFrame(columns=["time","capital"]),
            "trades_enriched": pd.DataFrame(columns=[
                "ticker","entry_time","exit_time","entry_price","exit_price",
                "pred_prob","threshold_used","exit_reason","pnl_pct",
                "pnl_usd","capital_after"
            ]),
            "final_capital": initial_capital,
            "max_drawdown_pct": 0.0
        }
        summary = summarize_performance(empty_equity, initial_capital)
        return empty_equity, summary

    # 2. juntar todas las operaciones de todos los tickers
    merged_trades = pd.concat(all_trades, ignore_index=True)

    # 3. construir curva de equity global con sizing y riesgo
    equity_result = build_equity_curve(
        trades_df=merged_trades,
        initial_capital=initial_capital,
        risk_fraction=risk_fraction
    )

    # 4. KPIs finales
    summary = summarize_performance(
        equity_result,
        initial_capital=initial_capital
    )

    return equity_result, summary
