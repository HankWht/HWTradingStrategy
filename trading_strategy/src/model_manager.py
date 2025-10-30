import os
from src.monitor import PerformanceMonitor
from src.utils import load_config
import subprocess

class ModelManager:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.cfg = load_config()

    def check_and_retrain(self):
        degraded = self.monitor.detect_degradation()
        if degraded:
            print("🚨 Reentrenamiento automático iniciado...")
            subprocess.run(["python", "run_pipeline.py"])
            print("✅ Reentrenamiento completado y señales actualizadas.")
