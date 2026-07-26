import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/paper_grade_data.csv"
MODEL_DIR = Path("models")

FEATURES = [
    "Stock_Flow",
    "Filler_Flow",
    "Steam_Pressure",
    "Machine_Speed",
    "Moisture",
    "Ash",
    "Caliper",
    "Wire_Vacuum",
    "Refiner_Load",
    "Draw_Tension",
    "Target_BasisWeight",
    "Time",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. Re-run generate_data.py "
            "to produce a dataset compatible with this training script."
        )
    return df


def train_classifier(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["Off_Spec"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    acc = accuracy_score(y_test, pred)

    print("\n=== Off-Spec Classifier ===")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred))

    importances = dict(
        sorted(
            zip(FEATURES, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return model, {
        "accuracy": acc,
        "classification_report": report,
        "feature_importance": importances,
    }


def train_stabilization_regressor(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["Time_To_Stable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    print("\n=== Stabilization-Time Regressor ===")
    print("MAE (time steps):", round(mae, 3))

    importances = dict(
        sorted(
            zip(FEATURES, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return model, {"mae_time_steps": mae, "feature_importance": importances}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()

    clf, clf_metrics = train_classifier(df)
    reg, reg_metrics = train_stabilization_regressor(df)

    joblib.dump(clf, MODEL_DIR / "offspec_model.pkl")
    joblib.dump(reg, MODEL_DIR / "stabilization_model.pkl")

    metrics = {
        "features": FEATURES,
        "offspec_classifier": clf_metrics,
        "stabilization_regressor": reg_metrics,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModels and metrics saved to models/")


if __name__ == "__main__":
    main()