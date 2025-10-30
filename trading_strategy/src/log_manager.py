import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def get_logger(name: str, max_mb: int = 5, backups: int = 5):
    """
    Devuelve un logger con rotación automática.
    - name: nombre del log (e.g. 'scheduler', 'monitor')
    - max_mb: tamaño máximo por archivo en MB antes de rotar
    - backups: número de archivos antiguos que se conservan
    """
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita duplicados si ya fue configurado
    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        log_file, maxBytes=max_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
    )

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # También muestra en consola
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
