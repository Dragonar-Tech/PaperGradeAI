
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/paper_grade_data.csv"
MODEL_DIR = Path("models")

KNOWN_LOOPS = [
    "Stock_Flow",
    "Filler_Flow",
    "Steam_Pressure",
    "Machine_Speed",
    "Moisture",
    "Ash",
    "Caliper",
]

HIDDEN_SENSORS = ["Wire_Vacuum", "Refiner_Load", "Draw_Tension"]

TARGETS = ["Off_Spec", "Deviation_Pct"]


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    corr = df.corr(numeric_only=True)

    # --- heatmap -------------------------------------------------------
    plt.figure(figsize=(12, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix - Paper Grade Change Process")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # --- structured summary for the dashboard ---------------------------
    summary = {"targets": {}}

    for target in TARGETS:
        if target not in corr.columns:
            continue
        target_corr = corr[target].drop(labels=[t for t in TARGETS if t in corr.columns])

        known = {
            k: round(float(target_corr[k]), 3)
            for k in KNOWN_LOOPS
            if k in target_corr.index
        }
        hidden = {
            k: round(float(target_corr[k]), 3)
            for k in HIDDEN_SENSORS
            if k in target_corr.index
        }

        summary["targets"][target] = {
            "known_loop_correlations": dict(
                sorted(known.items(), key=lambda x: abs(x[1]), reverse=True)
            ),
            "new_correlations": dict(
                sorted(hidden.items(), key=lambda x: abs(x[1]), reverse=True)
            ),
        }

    with open(MODEL_DIR / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Correlation Matrix:\n")
    print(corr.round(2))
    print("\nHeatmap saved to models/correlation_heatmap.png")
    print("Structured summary saved to models/correlation_summary.json\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
