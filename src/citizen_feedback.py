from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.config import DATA_OUTPUTS_DIR
from src.utils import get_timestamp


FEEDBACK_COLUMNS = [
    "feedback_id",
    "timestamp",
    "zone_or_ap",
    "rating",
    "issue_type",
    "comment",
    "sentiment_label",
    "sentiment_score",
    "source",
]


def get_feedback_file_path() -> Path:
    """Devuelve la ruta persistente del buzón ciudadano."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_OUTPUTS_DIR / "citizen_feedback.csv"


def simple_sentiment_analysis(text: str, rating: int | None = None) -> tuple[str, float]:
    """Clasifica sentimiento de forma ligera y trazable."""
    normalized_text = str(text or "").strip().lower()
    score = 0.0

    if rating is not None:
        if int(rating) <= 2:
            score -= 0.7
        elif int(rating) == 3:
            score += 0.0
        else:
            score += 0.7

    negative_terms = ["lento", "cae", "no conecta", "falla", "malo", "inestable", "demora", "desconecta"]
    positive_terms = ["buena", "excelente", "rapido", "rápido", "funciona", "estable", "mejoro", "mejoró"]

    if any(term in normalized_text for term in negative_terms):
        score -= 0.5
    if any(term in normalized_text for term in positive_terms):
        score += 0.5

    score = max(-1.0, min(1.0, score))
    if score <= -0.2:
        return "negativo", round(score, 2)
    if score >= 0.2:
        return "positivo", round(score, 2)
    return "neutral", round(score, 2)


def load_citizen_feedback() -> pd.DataFrame:
    """Carga el historial de reportes ciudadanos anónimos."""
    feedback_path = get_feedback_file_path()
    if not feedback_path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    try:
        feedback_df = pd.read_csv(feedback_path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    missing_columns = [column for column in FEEDBACK_COLUMNS if column not in feedback_df.columns]
    for column in missing_columns:
        feedback_df[column] = ""
    return feedback_df[FEEDBACK_COLUMNS].copy()


def save_citizen_feedback(
    zone_or_ap: str,
    rating: int,
    issue_type: str,
    comment: str,
    source: str = "platform",
) -> pd.DataFrame:
    """Guarda un reporte ciudadano anónimo."""
    sentiment_label, sentiment_score = simple_sentiment_analysis(comment, rating=rating)
    feedback_row = {
        "feedback_id": f"FDB-{uuid4().hex[:10].upper()}",
        "timestamp": get_timestamp(),
        "zone_or_ap": str(zone_or_ap or "").strip() or "Sin zona",
        "rating": int(rating),
        "issue_type": str(issue_type or "").strip() or "sin_categoria",
        "comment": str(comment or "").strip(),
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "source": str(source or "platform"),
    }

    existing_df = load_citizen_feedback()
    updated_df = pd.concat([existing_df, pd.DataFrame([feedback_row])], ignore_index=True)
    updated_df.to_csv(get_feedback_file_path(), index=False)
    return updated_df


def summarize_citizen_feedback(feedback_df: pd.DataFrame) -> dict[str, object]:
    """Resume el buzón ciudadano sin exponer datos personales."""
    if feedback_df is None or feedback_df.empty:
        return {
            "total_reportes": 0,
            "rating_promedio": 0.0,
            "problemas_mas_frecuentes": [],
            "zonas_con_mas_reportes": [],
            "zone_report_counts": {},
            "sentimiento_general": "Sin reportes",
            "ultimos_comentarios_anonimos": [],
        }

    summary_df = feedback_df.copy()
    summary_df["rating"] = pd.to_numeric(summary_df.get("rating"), errors="coerce")
    summary_df["sentiment_score"] = pd.to_numeric(summary_df.get("sentiment_score"), errors="coerce").fillna(0.0)

    problem_counts = summary_df.get("issue_type", pd.Series(dtype="object")).astype(str).value_counts().head(5)
    zone_counts = summary_df.get("zone_or_ap", pd.Series(dtype="object")).astype(str).value_counts().head(5)
    avg_sentiment = float(summary_df["sentiment_score"].mean()) if not summary_df.empty else 0.0

    if avg_sentiment >= 0.2:
        sentiment_label = "Predominio positivo"
    elif avg_sentiment <= -0.2:
        sentiment_label = "Predominio negativo"
    else:
        sentiment_label = "Mixto / neutral"

    recent_comments = summary_df.sort_values("timestamp", ascending=False).head(5)
    recent_payload = []
    for _, row in recent_comments.iterrows():
        comment_text = str(row.get("comment", "")).strip()
        if not comment_text:
            continue
        recent_payload.append(
            {
                "timestamp": row.get("timestamp", ""),
                "zone_or_ap": row.get("zone_or_ap", ""),
                "rating": row.get("rating", ""),
                "comment": comment_text[:180],
            }
        )

    return {
        "total_reportes": int(len(summary_df)),
        "rating_promedio": round(float(summary_df["rating"].dropna().mean()), 2) if summary_df["rating"].notna().any() else 0.0,
        "problemas_mas_frecuentes": [
            {"issue_type": issue_type, "count": int(count)}
            for issue_type, count in problem_counts.items()
        ],
        "zonas_con_mas_reportes": [
            {"zone_or_ap": zone_or_ap, "count": int(count)}
            for zone_or_ap, count in zone_counts.items()
        ],
        "zone_report_counts": {str(zone_or_ap): int(count) for zone_or_ap, count in zone_counts.items()},
        "sentimiento_general": sentiment_label,
        "ultimos_comentarios_anonimos": recent_payload,
    }
