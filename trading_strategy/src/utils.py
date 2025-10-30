import yaml
import os
from datetime import datetime, timezone


def load_config(path: str = "config.yaml") -> dict:
    """
    Carga el archivo de configuración YAML de manera segura.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """
    Guarda un diccionario de Python como archivo YAML.
    Crea automáticamente las carpetas necesarias si no existen.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_dirs():
    """
    Crea las carpetas base necesarias para el proyecto si no existen.
    """
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "src"  # opcional, asegura estructura completa
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)


def timestamp() -> str:
    """
    Devuelve una marca de tiempo ISO 8601 con zona horaria UTC.
    Ejemplo: '2025-10-30T22:15:30.123456+00:00'
    """
    return datetime.now(timezone.utc).isoformat()
