from __future__ import annotations

import re

import pandas as pd


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Normaliza estructuras a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def _normalize_text(value: object) -> str:
    """Reduce ruido para comparaciones simples de zona."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def explain_citizen_recommendation(row: pd.Series | dict[str, object]) -> str:
    """Genera una explicación clara y breve de la recomendación ciudadana."""
    row_data = row if isinstance(row, dict) else row.to_dict()
    zone_name = str(row_data.get("zone_name") or row_data.get("ap_name") or "esta zona")
    status = str(row_data.get("citizen_status", "sin clasificar")).lower()
    best_hours = str(row_data.get("best_hours", "sin evidencia horaria"))

    if status in {"excelente", "buena"}:
        return (
            f"{zone_name} se recomienda porque muestra buena disponibilidad, menor inestabilidad relativa "
            f"y actividad reciente. La franja sugerida es {best_hours}."
        )
    if status == "regular":
        return (
            f"{zone_name} puede usarse con precaución. Tiene evidencia de actividad, pero conviene priorizar "
            f"las mejores horas detectadas ({best_hours})."
        )
    if status == "inestable":
        return (
            f"{zone_name} presenta señales de inestabilidad. Solo se recomienda si no hay alternativas mejores "
            f"y evitando las franjas más sensibles."
        )
    return f"{zone_name} no tiene evidencia suficiente para una recomendación fuerte todavía."


def recommend_best_wifi_zones(
    citizen_scores: pd.DataFrame,
    user_zone: str | None = None,
    time_preference: str | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recomienda dónde conectarse según experiencia agregada y preferencia simple."""
    scores_df = _safe_dataframe(citizen_scores)
    if scores_df.empty:
        return pd.DataFrame(
            columns=[
                "zone_name",
                "ap_name",
                "score",
                "estado",
                "mejor_horario",
                "motivo",
                "precaucion",
            ]
        )

    filtered_df = scores_df.copy()
    filtered_df["zone_name"] = filtered_df.get("zone_name", filtered_df.get("ap_name", "")).astype(str)
    filtered_df["ap_name"] = filtered_df.get("ap_name", filtered_df["zone_name"]).astype(str)
    filtered_df["best_hours"] = filtered_df.get("best_hours", "Sin evidencia horaria").astype(str)
    filtered_df["avoid_hours"] = filtered_df.get("avoid_hours", "Sin evidencia horaria").astype(str)

    normalized_user_zone = _normalize_text(user_zone)
    if normalized_user_zone:
        zone_mask = filtered_df["zone_name"].map(_normalize_text).str.contains(normalized_user_zone, na=False)
        ap_mask = filtered_df["ap_name"].map(_normalize_text).str.contains(normalized_user_zone, na=False)
        preferred_df = filtered_df[zone_mask | ap_mask].copy()
        if not preferred_df.empty:
            filtered_df = preferred_df

    normalized_time = str(time_preference or "").strip()
    if normalized_time:
        preferred_time_df = filtered_df[
            filtered_df["best_hours"].str.contains(normalized_time, case=False, na=False)
            | ~filtered_df["avoid_hours"].str.contains(normalized_time, case=False, na=False)
        ].copy()
        if not preferred_time_df.empty:
            filtered_df = preferred_time_df

    filtered_df = filtered_df.sort_values(
        by=["citizen_experience_score", "data_confidence_score", "availability_score"],
        ascending=[False, False, False],
    ).head(top_n)

    filtered_df["score"] = pd.to_numeric(filtered_df.get("citizen_experience_score"), errors="coerce").round(2)
    filtered_df["estado"] = filtered_df.get("citizen_status", "")
    filtered_df["mejor_horario"] = filtered_df.get("best_hours", "Sin evidencia horaria")
    filtered_df["motivo"] = filtered_df.apply(explain_citizen_recommendation, axis=1)
    filtered_df["precaucion"] = filtered_df["avoid_hours"].map(
        lambda value: (
            f"Evita o revisa con cautela estas franjas: {value}."
            if value and str(value) != "Sin evidencia horaria"
            else "Sin precauciones horarias adicionales con la evidencia actual."
        )
    )

    return filtered_df[
        ["zone_name", "ap_name", "score", "estado", "mejor_horario", "motivo", "precaucion"]
    ].reset_index(drop=True)


def build_citizen_alerts(citizen_scores: pd.DataFrame) -> pd.DataFrame:
    """Construye alertas ciudadanas sin rastreo individual."""
    scores_df = _safe_dataframe(citizen_scores)
    if scores_df.empty:
        return pd.DataFrame(columns=["zone_name", "ap_name", "tipo_alerta", "severidad", "mensaje"])

    rows: list[dict[str, object]] = []
    for _, row in scores_df.iterrows():
        zone_name = str(row.get("zone_name") or row.get("ap_name") or "Sin zona")
        ap_name = str(row.get("ap_name") or zone_name)
        score = float(pd.to_numeric(row.get("citizen_experience_score"), errors="coerce") or 0)
        status = str(row.get("citizen_status", ""))
        availability = float(pd.to_numeric(row.get("availability_score"), errors="coerce") or 0)
        stability = float(pd.to_numeric(row.get("stability_score"), errors="coerce") or 0)
        confidence = float(pd.to_numeric(row.get("data_confidence_score"), errors="coerce") or 0)

        if status == "Inestable" or score < 55:
            rows.append(
                {
                    "zone_name": zone_name,
                    "ap_name": ap_name,
                    "tipo_alerta": "Zona inestable",
                    "severidad": "Alta" if score < 45 else "Media",
                    "mensaje": "La evidencia agregada sugiere experiencia irregular o desconexiones frecuentes.",
                }
            )
        if availability <= 35:
            rows.append(
                {
                    "zone_name": zone_name,
                    "ap_name": ap_name,
                    "tipo_alerta": "Disponibilidad baja",
                    "severidad": "Alta",
                    "mensaje": "El AP o zona presenta baja disponibilidad relativa con la evidencia actual.",
                }
            )
        if stability <= 45:
            rows.append(
                {
                    "zone_name": zone_name,
                    "ap_name": ap_name,
                    "tipo_alerta": "Desconexiones altas",
                    "severidad": "Media",
                    "mensaje": "La estabilidad estimada es baja por señales de desconexión por encima de lo deseable.",
                }
            )
        if confidence < 40:
            rows.append(
                {
                    "zone_name": zone_name,
                    "ap_name": ap_name,
                    "tipo_alerta": "Baja evidencia",
                    "severidad": "Media",
                    "mensaje": "La recomendación se apoya en poca evidencia agregada; conviene validar más histórico.",
                }
            )
        if status in {"Excelente", "Buena"} and score >= 70:
            rows.append(
                {
                    "zone_name": zone_name,
                    "ap_name": ap_name,
                    "tipo_alerta": "Mejor experiencia",
                    "severidad": "Baja",
                    "mensaje": "Esta zona aparece entre las opciones con mejor experiencia estimada para conectarse.",
                }
            )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "zone_name": "Global",
                    "ap_name": "Global",
                    "tipo_alerta": "Sin alertas relevantes",
                    "severidad": "Baja",
                    "mensaje": "No se detectaron alertas ciudadanas adicionales con la evidencia actual.",
                }
            ]
        )

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
