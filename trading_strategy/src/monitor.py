import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.utils import load_config

class PerformanceMonitor:
    def __init__(self):
        self.cfg = load_config()
        self.report_file = "reports/live_trades.csv"
        self.log_file = "reports/model_performance_log.csv"

    def analyze_performance(self):
        if not os.path.exists(self.report_file):
            print("No hay operaciones registradas aún.")
            return None

        df = pd.read_csv(self.report_file)
        if df.empty:
            print("Archivo de trades vacío.")
            return None

        df["pnl_pct"] = ((df["take_profit"] - df["price"]) / df["price"]) * 100 * (df["side"].map({"BUY": 1, "SELL": -1}))
        total_trades = len(df)
        win_rate = (df["pnl_pct"] > 0).mean() * 100
        avg_pnl = df["pnl_pct"].mean()
        drawdown = df["pnl_pct"].cumsum().min()
        profit_factor = (
            df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
            / abs(df.loc[df["pnl_pct"] < 0, "pnl_pct"].sum())
            if (df["pnl_pct"] < 0).any()
            else 1
        )

        stats = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 3),
            "drawdown": round(drawdown, 3),
            "profit_factor": round(profit_factor, 3),
        }

        os.makedirs("reports", exist_ok=True)
        pd.DataFrame([stats]).to_csv(
            self.log_file, mode="a", header=not os.path.exists(self.log_file), index=False
        )

        print(f"📈 Monitor actualizado: {stats}")
        return stats

    def detect_degradation(self, threshold_drawdown=-5, threshold_profit_factor=1.2):
        stats = self.analyze_performance()
        if not stats:
            return None

        degraded = stats["drawdown"] <= threshold_drawdown or stats["profit_factor"] < threshold_profit_factor
        if degraded:
            print("⚠️ Modelo en degradación detectado.")
            return True
        else:
            print("✅ Rendimiento dentro de parámetros normales.")
            return False
