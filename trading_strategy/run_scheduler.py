import os
import time
import schedule
from datetime import datetime
import subprocess

# ==========================================================
# Funciones del orquestador
# ==========================================================
def run_pipeline():
    print(f"[{datetime.now()}] Ejecutando pipeline completo...")
    subprocess.run(["python", "run_pipeline.py"])
    print("Pipeline completado.\n")

def run_backtest():
    print(f"[{datetime.now()}] Ejecutando backtest...")
    subprocess.run(["python", "run_backtest.py"])
    print("Backtest completado.\n")

def run_optimize():
    print(f"[{datetime.now()}] Ejecutando optimización semanal (Optuna)...")
    subprocess.run(["python", "run_optimize.py"])
    print("Optimización completada.\n")

def run_walkforward():
    print(f"[{datetime.now()}] Ejecutando validación walk-forward...")
    subprocess.run(["python", "run_walkforward.py"])
    print("Validación completada.\n")

def run_monitor():
    print(f"[{datetime.now()}] Ejecutando monitor inteligente...")
    subprocess.run(["python", "run_monitor.py"])
    print("Monitor actualizado.\n")

# ==========================================================
# Programación de tareas
# ==========================================================
# Pipeline diario (descarga + entrenamiento + señales)
schedule.every().day.at("09:00").do(run_pipeline)

# Backtest diario (evalúa rendimiento reciente)
schedule.every().day.at("10:30").do(run_backtest)

# Walk-forward semanal (valida robustez)
schedule.every().monday.at("13:00").do(run_walkforward)

# Optimización semanal (reentrena parámetros)
schedule.every().sunday.at("14:00").do(run_optimize)

# Monitor inteligente (cada 2 horas)
schedule.every(2).hours.do(run_monitor)

# ==========================================================
# Bucle principal
# ==========================================================
print("HWTradingStrategy Orchestrator iniciado ✅")
print("Tareas programadas:")
print("- Pipeline diario: 09:00")
print("- Backtest diario: 10:30")
print("- Walk-forward: lunes 13:00")
print("- Optimización: domingo 14:00")
print("- Monitor: cada 2 horas\n")

while True:
    schedule.run_pending()
    time.sleep(60)
