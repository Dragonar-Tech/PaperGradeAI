
import csv
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("data/feedback_log.csv")
FIELDS = [
    "timestamp",
    "transition_id",
    "time_step",
    "parameter",
    "action",
    "source",
    "confidence",
    "decision",
]


def log_decision(transition_id, time_step, parameter, action, source, confidence, decision: str):
    """decision must be 'accepted' or 'rejected'."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    is_new = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "transition_id": transition_id,
            "time_step": time_step,
            "parameter": parameter,
            "action": action,
            "source": source,
            "confidence": confidence,
            "decision": decision,
        })


def load_feedback_summary():
    import pandas as pd

    if not LOG_PATH.exists():
        return None

    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return None

    summary = (
        df.groupby("source")["decision"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    return summary
