import pandas as pd

Z_THRESHOLD = 2.0  # transitions beyond this many std devs are flagged


def summarize_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Transition_ID with summary stats used for anomaly
    detection and trend charting."""
    grouped = df.groupby("Transition_ID")

    summary = grouped.agg(
        Grade_From=("Grade_From", "first"),
        Grade_To=("Grade_To", "first"),
        Target_BasisWeight=("Target_BasisWeight", "first"),
        Peak_Deviation_Pct=("Deviation_Pct", "max"),
        Mean_Deviation_Pct=("Deviation_Pct", "mean"),
        Off_Spec_Rate=("Off_Spec", "mean"),
        Stabilization_Time=("Time_To_Stable", "first"),  # value at Time=0
    ).reset_index()

    return summary


def detect_anomalies(summary: pd.DataFrame, z_thresh: float = Z_THRESHOLD) -> pd.DataFrame:
    """Adds Is_Anomaly (bool) and Anomaly_Reasons (str) columns, using a
    per-target-grade z-score so a naturally "harder" grade change isn't
    unfairly flagged against easier ones."""
    metrics = ["Peak_Deviation_Pct", "Stabilization_Time", "Off_Spec_Rate"]

    out = summary.copy()
    out["Is_Anomaly"] = False
    out["Anomaly_Reasons"] = ""

    for target_bw, group in summary.groupby("Target_BasisWeight"):
        if len(group) < 5:
            continue  # not enough peers to judge an outlier meaningfully

        for metric in metrics:
            mean = group[metric].mean()
            std = group[metric].std()
            if std == 0 or pd.isna(std):
                continue

            z = (group[metric] - mean) / std
            flagged = group.index[z > z_thresh]

            for idx in flagged:
                reason = (
                    f"{metric.replace('_', ' ')} is "
                    f"{z.loc[idx]:.1f}σ above the norm for "
                    f"{int(target_bw)}gsm transitions"
                )
                out.loc[idx, "Is_Anomaly"] = True
                existing = out.loc[idx, "Anomaly_Reasons"]
                out.loc[idx, "Anomaly_Reasons"] = (
                    f"{existing}; {reason}" if existing else reason
                )

    return out


def main():
    df = pd.read_csv("data/paper_grade_data.csv")
    summary = summarize_transitions(df)
    flagged = detect_anomalies(summary)

    n_anom = int(flagged["Is_Anomaly"].sum())
    print(f"Analyzed {len(flagged)} transitions, flagged {n_anom} anomalies.\n")

    if n_anom:
        print(flagged[flagged["Is_Anomaly"]][
            ["Transition_ID", "Grade_From", "Grade_To",
             "Peak_Deviation_Pct", "Stabilization_Time",
             "Off_Spec_Rate", "Anomaly_Reasons"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()