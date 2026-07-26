
import pandas as pd
import numpy as np
from pathlib import Path

RNG = np.random.default_rng(42)

GRADES = [60, 70, 80, 90, 100]
N_TRANSITIONS = 500
STEPS_PER_TRANSITION = 60
SPEC_BAND_PCT = 0.025  # +/- 2.5% of target is "in spec"


def simulate_transition(transition_id: int) -> list[dict]:
    grade_from = RNG.choice(GRADES)
    grade_to = RNG.choice(GRADES)
    while grade_to == grade_from:
        grade_to = RNG.choice(GRADES)

    target_bw = grade_to

    # Per-transition "quality" of the grade change (varies transition to
    # transition, e.g. due to how well the operator/controller executes it).
    # This is what a real learning system would be trying to predict/explain.
    execution_quality = RNG.normal(1.0, 0.35)
    execution_quality = max(execution_quality, 0.2)

    rows = []
    settled_step = STEPS_PER_TRANSITION  # default: never settles in window

    for t in range(STEPS_PER_TRANSITION):
        stock_flow = RNG.normal(100 + target_bw * 0.6, 3)
        filler_flow = RNG.normal(15 + target_bw * 0.08, 1)
        steam_pressure = RNG.normal(4 + target_bw / 100, 0.25)
        machine_speed = RNG.normal(500 - target_bw, 10)
        moisture = RNG.normal(6.5, 0.4)
        ash = RNG.normal(12, 0.8)
        caliper = RNG.normal(target_bw / 10, 0.2)

        # --- hidden / under-monitored sensors -----------------------------
        wire_vacuum = RNG.normal(18 + (500 - machine_speed) * 0.01, 0.6)
        refiner_load = RNG.normal(65 + filler_flow * 0.3, 2.5)
        draw_tension = RNG.normal(2.2 + abs(grade_to - grade_from) * 0.01, 0.15)

        # Deviation decays exponentially towards 0 as the machine stabilizes,
        # but execution_quality and the hidden sensors add real variance that
        # is NOT visible to the standard control loops.
        base_deviation = np.exp(-t / 18) * execution_quality * RNG.normal(5, 1.5)
        hidden_penalty = (
            0.4 * max(wire_vacuum - 18, 0)
            + 0.3 * max(refiner_load - 65, 0) / 10
            + 0.6 * max(draw_tension - 2.2, 0) * 10
        )
        deviation = base_deviation + hidden_penalty * np.exp(-t / 25)

        actual_bw = target_bw + deviation
        deviation_pct = abs(actual_bw - target_bw) / target_bw
        off_spec = deviation_pct > SPEC_BAND_PCT

        if not off_spec and settled_step == STEPS_PER_TRANSITION:
            settled_step = t

        rows.append({
            "Transition_ID": transition_id,
            "Time": t,
            "Grade_From": grade_from,
            "Grade_To": grade_to,
            "Stock_Flow": round(stock_flow, 2),
            "Filler_Flow": round(filler_flow, 2),
            "Steam_Pressure": round(steam_pressure, 2),
            "Machine_Speed": round(machine_speed, 2),
            "Moisture": round(moisture, 2),
            "Ash": round(ash, 2),
            "Caliper": round(caliper, 2),
            "Wire_Vacuum": round(wire_vacuum, 2),
            "Refiner_Load": round(refiner_load, 2),
            "Draw_Tension": round(draw_tension, 2),
            "Target_BasisWeight": target_bw,
            "Actual_BasisWeight": round(actual_bw, 2),
            "Deviation_Pct": round(deviation_pct * 100, 3),
            "Off_Spec": int(off_spec),
        })

    # Back-fill "steps remaining until the transition settles" for every row
    # in this transition -- this is the regression target used to predict /
    # reduce stabilization time.
    for row in rows:
        row["Time_To_Stable"] = max(settled_step - row["Time"], 0)

    return rows


def main():
    all_rows = []
    for transition in range(N_TRANSITIONS):
        all_rows.extend(simulate_transition(transition))

    df = pd.DataFrame(all_rows)

    output = Path("data")
    output.mkdir(exist_ok=True)
    df.to_csv(output / "paper_grade_data.csv", index=False)

    print("Dataset generated successfully!")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nOff-Spec rate:", df["Off_Spec"].mean().round(3))


if __name__ == "__main__":
    main()
