import schedule
import time
from src.monitor import PerformanceMonitor
from src.alert_manager import AlertManager
from src.model_manager import ModelManager

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    model_manager = ModelManager()

    # Configuración opcional de alertas
    alerts = AlertManager(
        email_conf={
            "server": "smtp.gmail.com",
            "port": 587,
            "sender": "tu_correo@gmail.com",
            "password": "tu_contraseña_app",
            "recipient": "destinatario@gmail.com",
        },
        telegram_token="TOKEN_TELEGRAM_OPCIONAL",
        telegram_chat_id="CHAT_ID_TELEGRAM"
    )

    def run_monitor_cycle():
        stats = monitor.analyze_performance()
        if monitor.detect_degradation():
            alerts.send_email("⚠️ Alerta de Degradación", f"Modelo con drawdown o profit factor anormal:\n{stats}")
            alerts.send_telegram(f"🚨 Estrategia degradada: {stats}")
            model_manager.check_and_retrain()

    # Ejecutar cada 30 minutos
    schedule.every(30).minutes.do(run_monitor_cycle)

    print("🧠 Monitor inteligente iniciado (verifica cada 30 minutos)...")
    while True:
        schedule.run_pending()
        time.sleep(60)
        print("✅ Modelo funcionando dentro de parámetros normales.")