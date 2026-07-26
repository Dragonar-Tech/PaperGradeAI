
from dataclasses import dataclass, field

import numpy as np
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
]

# Hard recipe / actuator limits (independent of history). Adjust to match
# real mill recipe limits.
RECIPE_LIMITS = {
    "Steam_Pressure": (3.0, 6.5),
    "Machine_Speed": (350, 480),
    "Draw_Tension": (1.5, 3.2),
}

K_NEIGHBORS = 25
TIME_WINDOW = 5          # +/- steps considered "similar point in transition"
MIN_MEANINGFUL_DELTA = 0.5  # in units of that feature's std dev


@dataclass
class Recommendation:
    action: str
    parameter: str
    source: str
    confidence: float
    rationale: str = ""


def _nearest_successful_neighbors(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    pool = df[
        (df["Off_Spec"] == 0)
        & (df["Target_BasisWeight"] == row["Target_BasisWeight"])
        & (df["Time"].between(row["Time"] - TIME_WINDOW, row["Time"] + TIME_WINDOW))
    ]
    if pool.empty:
        pool = df[(df["Off_Spec"] == 0) & (df["Target_BasisWeight"] == row["Target_BasisWeight"])]
    if pool.empty:
        return pool

    std = df[FEATURES].std().replace(0, 1)
    dist = ((pool[FEATURES] - row[FEATURES]) / std).pow(2).sum(axis=1).pow(0.5)
    nearest = pool.loc[dist.sort_values().index[:K_NEIGHBORS]]
    return nearest


def recommend(row: pd.Series, df: pd.DataFrame) -> list[Recommendation]:
    recs: list[Recommendation] = []
    std = df[FEATURES].std().replace(0, 1)

    # 1) Hard recipe limits first - these override everything else.
    for feat, (lo, hi) in RECIPE_LIMITS.items():
        val = row[feat]
        if val < lo:
            recs.append(Recommendation(
                action=f"Increase {feat.replace('_', ' ')} toward {lo} (below recipe minimum)",
                parameter=feat,
                source="Recipe / actuator limits",
                confidence=99,
                rationale=f"{feat} = {val:.2f} is below the recipe minimum of {lo}.",
            ))
        elif val > hi:
            recs.append(Recommendation(
                action=f"Reduce {feat.replace('_', ' ')} toward {hi} (above recipe maximum)",
                parameter=feat,
                source="Recipe / actuator limits",
                confidence=99,
                rationale=f"{feat} = {val:.2f} is above the recipe maximum of {hi}.",
            ))

    # 2) Data-driven setpoint suggestions from similar historical successes.
    neighbors = _nearest_successful_neighbors(df, row)
    if not neighbors.empty:
        ref = neighbors[FEATURES].median()
        n = len(neighbors)
        for feat in FEATURES:
            if feat in RECIPE_LIMITS:
                continue  # already handled by hard limits above
            delta = row[feat] - ref[feat]
            if abs(delta) < MIN_MEANINGFUL_DELTA * std[feat]:
                continue
            direction = "Reduce" if delta > 0 else "Increase"
            similarity_pct = 100 * (1 - min(abs(delta) / (std[feat] * 3), 1))
            recs.append(Recommendation(
                action=f"{direction} {feat.replace('_', ' ')} toward {ref[feat]:.2f}",
                parameter=feat,
                source=f"Historical successful transitions (n={n} similar cases)",
                confidence=round(60 + 0.35 * similarity_pct, 1),
                rationale=(
                    f"Current {feat} = {row[feat]:.2f} vs. median "
                    f"{ref[feat]:.2f} in {n} similar in-spec transitions."
                ),
            ))

    if not recs:
        recs.append(Recommendation(
            action="No action required - system tracking within historical norms",
            parameter="-",
            source="Historical successful transitions",
            confidence=95,
            rationale="All monitored parameters are within the normal band "
                       "observed in successful past transitions.",
        ))

    # Highest confidence first, cap to a reasonable number for the UI.
    recs.sort(key=lambda r: r.confidence, reverse=True)
    return recs[:5]


def main():
    df = pd.read_csv("data/paper_grade_data.csv")
    sample = df.iloc[0]

    print("Current State\n")
    print(sample[FEATURES + ["Target_BasisWeight", "Time", "Off_Spec"]])

    print("\nRecommendations\n")
    for r in recommend(sample, df):
        print(f"• [{r.source}, conf={r.confidence}%] {r.action}")
        print(f"    ↳ {r.rationale}")


if __name__ == "__main__":
    main()
