import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import os

def train_model_for_ticker(
    df,
    feature_cols,
    model_params,
    test_ratio=0.2,
    save_path="models",
    ticker_name="TICKER"
):
    # Split temporal forward (no shuffle)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df["label"].iloc[:split_idx]
    X_test  = df[feature_cols].iloc[split_idx:]
    y_test  = df["label"].iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=model_params["n_estimators"],
        learning_rate=model_params["learning_rate"],
        max_depth=model_params["max_depth"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        eval_metric="logloss",
        n_jobs=-1,
        random_state=model_params.get("random_state", 42),
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, digits=4)
    print(f"=== {ticker_name} ===")
    print(report)

    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, f"{ticker_name}_model.pkl"))

    return model, report
