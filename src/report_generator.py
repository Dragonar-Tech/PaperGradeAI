import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _trajectory_chart_png(t_df: pd.DataFrame, time_step: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_df["Time"], t_df["Actual_BasisWeight"], label="Actual", color="#1f77b4")
    ax.plot(t_df["Time"], t_df["Target_BasisWeight"], label="Target",
            color="#888888", linestyle="--")
    ax.axvline(time_step, color="red", linestyle=":", label="Current step")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Basis Weight")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_report_pdf(
    transition_id: int,
    time_step: int,
    row: pd.Series,
    t_df: pd.DataFrame,
    risk: float,
    eta,
    rationale_result: dict,
    recommendations: list,
) -> io.BytesIO:
    """Returns an in-memory PDF (BytesIO) ready for a Streamlit
    download_button."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    story = []

    story.append(Paragraph("Paper Grade Intelligence — Session Report", title_style))
    story.append(Paragraph(
        f"Transition {transition_id} · Grade {int(row['Grade_From'])} → "
        f"{int(row['Grade_To'])} gsm · Time step {time_step}", body
    ))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", small
    ))
    story.append(Spacer(1, 0.5 * cm))

    # --- Risk summary table ------------------------------------------
    eta_text = f"{eta:.1f} steps" if eta is not None else "n/a"
    summary_data = [
        ["Off-Spec Risk", f"{risk*100:.1f}%"],
        ["Est. Time to Stable", eta_text],
        ["Actual Basis Weight", f"{row['Actual_BasisWeight']:.2f}"],
        ["Target Basis Weight", f"{row['Target_BasisWeight']:.2f}"],
    ]
    tbl = Table(summary_data, colWidths=[6 * cm, 6 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5 * cm))

    # --- Trajectory chart ----------------------------------------------
    story.append(Paragraph("Basis Weight Trajectory", h2))
    chart_buf = _trajectory_chart_png(t_df, time_step)
    story.append(Image(chart_buf, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # --- Rationale -------------------------------------------------------
    story.append(Paragraph("Why this risk level?", h2))
    story.append(Paragraph(rationale_result.get("source", ""), small))
    for d in rationale_result.get("top_drivers", []):
        story.append(Paragraph(f"• {d['text']}", body))
    story.append(Spacer(1, 0.4 * cm))

    # --- Recommendations -------------------------------------------------
    story.append(Paragraph("Recommendations", h2))
    rec_data = [["Action", "Source", "Confidence"]]
    for r in recommendations:
        rec_data.append([r.action, r.source, f"{r.confidence}%"])
    rec_tbl = Table(rec_data, colWidths=[8 * cm, 6 * cm, 2.5 * cm], repeatRows=1)
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(rec_tbl)

    doc.build(story)
    buf.seek(0)
    return buf