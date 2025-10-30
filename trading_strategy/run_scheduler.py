import schedule
import subprocess
import time
from datetime import datetime
from src.log_manager import get_logger

logger = get_logger("scheduler")

def run_task(name, command):
    logger.info(f"Ejecutando tarea: {name}")
    try:
        subprocess.run(["python", command], check=True)
        logger.info(f"✅ {name} completado correctamente.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error al ejecutar {name}: {e}")

# ==========================================================
# Tareas programadas
# ==========================================================
schedule.every().day.at("09:00").do(run_task, "Pipeline diario", "run_pipeline.py")
schedule.every().day.at("10:30").do(run_task, "Backtest diario", "run_backtest.py")
schedule.every().monday.at("13:00").do(run_task, "Walk-Forward semanal", "run_walkforward.py")
schedule.every().sunday.at("14:00").do(run_task, "Optimización semanal", "run_optimize.py")
schedule.every(2).hours.do(run_task, "Monitor inteligente", "run_monitor.py")

logger.info("HWTradingStrategy Scheduler iniciado ✅")
logger.info("Tareas programadas:")
logger.info("- Pipeline diario: 09:00")
logger.info("- Backtest diario: 10:30")
logger.info("- Walk-Forward semanal: lunes 13:00")
logger.info("- Optimización semanal: domingo 14:00")
logger.info("- Monitor inteligente: cada 2 horas\n")

while True:
    schedule.run_pending()
    time.sleep(60)
