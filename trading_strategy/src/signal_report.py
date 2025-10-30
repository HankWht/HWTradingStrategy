import os
import pandas as pd
from .ensemble import generate_ensemble_signals
from .utils import ensure_dirs, load_config, timestamp

def save_signal_report(data_map, feature_cols):
    cfg = load_config()
    ensure_dirs()

    report_df = generate_ensemble_signals(
        data_map=data_map,
        feature_cols=feature_cols,
        model_dir="models",
        base_threshold=cfg["strategy"]["base_prob_threshold"]
    )

    # calcular TP/SL sugeridos para que tú lo pongas manualmente en el broker
    TP = cfg["strategy"]["tp"]
    SL = cfg["strategy"]["sl"]

    report_df["TP_target_%"] = TP * 100
    report_df["SL_stop_%"] = SL * 100

    # guardamos en reports
    fname = f"reports/signals_{timestamp()}.csv".replace(":", "-")
    report_df.to_csv(fname, index=False)
    report_df.to_csv("reports/signals_latest.csv", index=False)

    return report_df
