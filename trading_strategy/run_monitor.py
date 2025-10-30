import schedule
import time
from src.monitor import PerformanceMonitor
from src.alert_manager import AlertManager
from src.model_manager import ModelManager
from src.log_manager import get_logger

logger = get_logger("monitor")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    model_manager = ModelManager()

    alerts = AlertManager(
        email_conf={
            "server": "smtp.gmail.com",
            "port": 587,
            "sender": "tu_correo@gmail.com",
            "password": "tu_contraseña_app",
            "recipient": "destinatario@gmail.com",
        },
        telegram_token=None,
        telegram_chat_id=None
    )

    def run_monitor_cycle():
        logger.info("Ejecutando ciclo de monitoreo...")
        stats = monitor.analyze_performance()
        if monitor.detect_degradation():
            msg = f"⚠️ Modelo degradado detectado.\nStats: {stats}"
            logger.warning(msg)
            alerts.send_email("⚠️ Alerta de Degradación", msg)
            alerts.send_telegram(msg)
            model_manager.check_and_retrain()
        else:
            logger.info("✅ Rendimiento dentro de parámetros normales.")

    schedule.every(30).minutes.do(run_monitor_cycle)

    logger.info("Monitor inteligente iniciado (revisión cada 30 minutos)...")
    while True:
        schedule.run_pending()
        time.sleep(60)
