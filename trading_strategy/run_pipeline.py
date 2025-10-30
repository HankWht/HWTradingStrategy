import logging
from src.utils import load_config, ensure_dirs
from src.data_loader import load_universe_data
from src.features import build_features
from src.labeler import make_labels
from src.model_train import train_model_for_ticker
from src.signal_report import save_signal_report

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = load_config()
    ensure_dirs()

    # IMPORTANTE: tickers.txt debe estar en data/tickers.txt (no en processed)
    with open("data/tickers.txt") as f:
        tickers = [line.strip() for line in f if line.strip()]

    lookback_days = cfg["data"]["lookback_days"]
    interval = cfg["data"]["interval"]

    logging.info(f"Iniciando pipeline para tickers: {', '.join(tickers)}")

    # 1. Descargar datos crudos
    raw_map = load_universe_data(tickers, lookback_days, interval)

    # 2. Generar features + etiquetas
    data_map = {}
    for t, df in raw_map.items():
        logging.info(f"Procesando {t}...")
        feat_df = build_features(df)
        labeled_df = make_labels(
            feat_df,
            cfg["strategy"]["horizon"],
            cfg["strategy"]["tp"],
            cfg["strategy"]["sl"]
        )
        data_map[t] = labeled_df

    # 3. Entrenar y guardar modelos por ticker
    feature_cols = cfg["features"]["include"]

    for t, df in data_map.items():
        logging.info(f"Entrenando modelo para {t}...")
        train_model_for_ticker(
            df=df,
            feature_cols=feature_cols,
            model_params=cfg["model"],
            test_ratio=cfg["model"]["test_split_ratio"],
            save_path="models",
            ticker_name=t
        )

    # 4. Generar señales ensemble y guardarlas
    logging.info("Generando señales ensemble...")
    report_df = save_signal_report(data_map, feature_cols)
    logging.info("Listo. Señales guardadas en reports/signals_latest.csv")
    logging.info(report_df)

if __name__ == "__main__":
    main()
