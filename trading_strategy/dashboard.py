# ==========================================================
# HWTradingStrategy Unified Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import yaml
from datetime import datetime

# Optuna visualization
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour
)

# --- Configuración general ---
st.set_page_config(
    page_title="HWTradingStrategy Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("HWTradingStrategy Dashboard")
st.caption("Plataforma unificada de análisis, backtesting, optimización y validación walk-forward")

# ==========================================================
# Funciones auxiliares
# ==========================================================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_yaml(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def metric_card(title, value, description=""):
    st.metric(label=title, value=value, delta=description)

# ==========================================================
# Cargar datos
# ==========================================================
signals_df = load_csv("reports/signals_latest.csv")
equity_df = load_csv("reports/equity_curve.csv")
trades_df = load_csv("reports/trades_log.csv")
optimization_yaml = load_yaml("reports/strategy_optimization.yaml")
walkforward_df = load_csv("reports/walkforward_results.csv")
walkforward_summary = load_csv("reports/walkforward_summary.csv")

# ==========================================================
# Interfaz: Pestañas principales
# ==========================================================
tabs = st.tabs([
    "📊 Resumen General",
    "💹 Trades & Backtest",
    "⚡ Señales Actuales",
    "🔍 Optimización (Optuna)",
    "🧪 Validación Walk-Forward"
])

# ==========================================================
# TAB 1: Resumen General
# ==========================================================
with tabs[0]:
    st.header("Resumen de Desempeño")

    if not equity_df.empty:
        total_return = (equity_df["capital"].iloc[-1] / equity_df["capital"].iloc[0] - 1) * 100
        num_trades = len(trades_df)
        win_rate = (trades_df["pnl_usd"] > 0).mean() * 100 if not trades_df.empty else 0
        avg_trade = trades_df["pnl_pct"].mean() * 100 if not trades_df.empty else 0
        profit_factor = (
            trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"].sum()
            / abs(trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"].sum())
            if not trades_df.empty and (trades_df["pnl_usd"] < 0).any()
            else 1
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        metric_card("Retorno Total", f"{total_return:.2f}%")
        metric_card("Número de Trades", f"{num_trades}")
        metric_card("Win Rate", f"{win_rate:.1f}%")
        metric_card("Promedio/Trade", f"{avg_trade:.2f}%")
        metric_card("Profit Factor", f"{profit_factor:.2f}")

        st.markdown("### Curva de Capital")
        fig_equity = px.line(equity_df, x="time", y="capital", title="Evolución del Capital")
        st.plotly_chart(fig_equity, use_container_width=True)

    else:
        st.warning("Ejecuta `python run_backtest.py` para generar resultados de backtesting.")

# ==========================================================
# TAB 2: Trades & Backtest
# ==========================================================
with tabs[1]:
    st.header("Historial de Trades")

    if not trades_df.empty:
        tickers = sorted(trades_df["ticker"].unique())
        selected_ticker = st.selectbox("Filtrar por Ticker", ["Todos"] + tickers)

        if selected_ticker != "Todos":
            df = trades_df[trades_df["ticker"] == selected_ticker]
        else:
            df = trades_df.copy()

        st.dataframe(df.sort_values("entry_time", ascending=False), use_container_width=True, height=400)

        st.markdown("### Distribución de PnL")
        fig = px.bar(
            df, x="entry_time", y="pnl_usd",
            color="pnl_usd", color_continuous_scale=["red", "green"],
            title=f"PnL por Trade ({selected_ticker})"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Rendimiento por Ticker")
        perf_by_ticker = trades_df.groupby("ticker")["pnl_usd"].sum().sort_values(ascending=False)
        fig_perf = px.bar(
            perf_by_ticker, x=perf_by_ticker.index, y=perf_by_ticker.values,
            title="Rendimiento total por ticker", color=perf_by_ticker.values,
            color_continuous_scale="Bluered"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("Aún no hay datos de trades registrados.")

# ==========================================================
# TAB 3: Señales Actuales
# ==========================================================
with tabs[2]:
    st.header("Señales Generadas por el Modelo")

    if not signals_df.empty:
        st.dataframe(signals_df, use_container_width=True, height=400)
        st.markdown("Filtra señales o monitorea oportunidades recientes.")

    else:
        st.warning("Ejecuta `python run_pipeline.py` para generar señales actualizadas.")

# ==========================================================
# TAB 4: Análisis de Optimización (Optuna)
# ==========================================================
with tabs[3]:
    st.header("Análisis de Optimización (Optuna)")

    if optimization_yaml:
        tickers = list(optimization_yaml.keys())
        selected_ticker = st.selectbox("Selecciona un ticker:", tickers)

        study_db = f"reports/optuna_{selected_ticker}.db"
        if os.path.exists(study_db):
            try:
                study = optuna.load_study(
                    study_name=f"optuna_{selected_ticker}",
                    storage=f"sqlite:///{study_db}"
                )

                best = optimization_yaml[selected_ticker]["best_params"]
                st.markdown(f"**Mejores Parámetros:** `{best}`")
                st.markdown(f"**Última Optimización:** {optimization_yaml[selected_ticker]['last_optimized']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_optimization_history(study), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_param_importances(study), use_container_width=True)

                st.markdown("### Relación entre Parámetros (3D)")
                st.plotly_chart(plot_parallel_coordinate(study), use_container_width=True)
                st.plotly_chart(plot_contour(study), use_container_width=True)

            except Exception as e:
                st.error(f"Error al cargar el estudio: {e}")
        else:
            st.warning(f"No se encontró {study_db}. Ejecuta `python run_optimize.py` primero.")
    else:
        st.info("No se encontró 'strategy_optimization.yaml'. Ejecuta la optimización antes de visualizar.")

# ==========================================================
# TAB 5: Validación Walk-Forward / Robustez
# ==========================================================
with tabs[4]:
    st.header("Validación Walk-Forward y Robustez del Modelo")

    wf_path = "reports/walkforward_results.csv"
    wf_summary_path = "reports/walkforward_summary.csv"

    if os.path.exists(wf_path):
        wf_df = pd.read_csv(wf_path)

        st.markdown("### Resultados por ventana temporal")
        st.dataframe(
            wf_df.sort_values(["ticker", "test_end"]),
            use_container_width=True,
            height=300
        )

        # gráfico de F1 score a lo largo del tiempo
        if "f1" in wf_df.columns:
            st.markdown("### Estabilidad del F1 Score por ticker")
            ticker_sel = st.selectbox(
                "Selecciona ticker para análisis temporal",
                sorted(wf_df["ticker"].unique())
            )

            df_t = wf_df[wf_df["ticker"] == ticker_sel].copy()
            df_t["test_end"] = pd.to_datetime(df_t["test_end"])

            fig_f1 = px.line(
                df_t,
                x="test_end",
                y="f1",
                title=f"F1 Score en el tiempo ({ticker_sel})",
                markers=True
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # resumen agregado
        if os.path.exists(wf_summary_path):
            st.markdown("### Métricas promedio por ticker")
            wf_sum = pd.read_csv(wf_summary_path, index_col=0)
            st.dataframe(wf_sum, use_container_width=True)

            # barplot de F1 promedio
            if "f1" in wf_sum.columns:
                fig_bar = px.bar(
                    wf_sum.reset_index(),
                    x="ticker",
                    y="f1",
                    title="F1 promedio por ticker (walk-forward)",
                    color="f1",
                    color_continuous_scale="Bluered"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.info("Interpretación: un ticker con F1 estable y alto en ventanas distintas es más robusto para trading real.")

    else:
        st.warning("Aún no se ha generado walkforward_results.csv. Ejecuta: `python run_walkforward.py`")
