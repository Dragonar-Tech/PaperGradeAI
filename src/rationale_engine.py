from pathlib import Path

import joblib
import pandas as pd

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

MODEL_DIR = Path("models")


def load_reference(df: pd.DataFrame, target_bw: float) -> pd.Series:
    """Median profile of historically successful (in-spec) transitions to
    the same grade, used as the 'healthy' baseline for comparison."""
    good = df[(df["Off_Spec"] == 0) & (df["Target_BasisWeight"] == target_bw)]
    if good.empty:
        good = df[df["Off_Spec"] == 0]
    return good[FEATURES].median()


def explain_row(model, df: pd.DataFrame, row: pd.Series, top_n: int = 4) -> dict:
    """Return risk probability, ranked drivers, and plain-English rationale
    for a single live process row."""
    X = pd.DataFrame([row[FEATURES]])
    risk = float(model.predict_proba(X)[0][1])

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    baseline = load_reference(df, row["Target_BasisWeight"])

    deltas = row[FEATURES] - baseline
    # Weight the raw delta by how much the model cares about that feature,
    # normalized by the feature's typical spread, so we rank "important AND
    # unusual" features first.
    spread = df[FEATURES].std().replace(0, 1)
    weighted_signal = (deltas / spread).abs() * importances
    ranked = weighted_signal.sort_values(ascending=False)

    drivers = []
    for feat in ranked.index[:top_n]:
        delta = deltas[feat]
        direction = "above" if delta > 0 else "below"
        drivers.append({
            "feature": feat,
            "current_value": round(float(row[feat]), 3),
            "healthy_reference": round(float(baseline[feat]), 3),
            "delta": round(float(delta), 3),
            "direction": direction,
            "model_importance": round(float(importances[feat]), 3),
            "text": (
                f"{feat.replace('_', ' ')} is {abs(round(delta, 2))} "
                f"{direction} the median of historically successful "
                f"transitions to this grade (importance="
                f"{importances[feat]:.2f})."
            ),
        })

    return {
        "risk_probability": round(risk, 3),
        "top_drivers": drivers,
        "source": f"Compared against {int((df['Off_Spec']==0).sum())} "
                   f"historical in-spec rows for Target_BasisWeight="
                   f"{row['Target_BasisWeight']}",
    }


def main():
    model = joblib.load(MODEL_DIR / "offspec_model.pkl")
    df = pd.read_csv("data/paper_grade_data.csv")

    print("Global feature importance (Off-Spec classifier):\n")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    for f, i in importances.sort_values(ascending=False).items():
        print(f"{f:20} {i:.3f}")

    sample_row = df.iloc[0]
    print("\n--- Example single-row rationale ---")
    result = explain_row(model, df, sample_row)
    print(f"Risk probability: {result['risk_probability']*100:.1f}%")
    print(f"Source: {result['source']}\n")
    for d in result["top_drivers"]:
        print("•", d["text"])


if __name__ == "__main__":
    main()
