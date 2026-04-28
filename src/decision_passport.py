from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.utils import get_timestamp, sanitize_filename


def _to_list(value: object) -> list[str]:
    """Normaliza un valor a lista de strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    text = str(value)
    if not text:
        return []
    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


def _confidence_label(data_confidence_score: float) -> str:
    """Traduce un score de confianza a etiqueta cualitativa."""
    if data_confidence_score >= 80:
        return "Alto"
    if data_confidence_score >= 55:
        return "Medio"
    return "Bajo"


def generate_decision_passport(
    zone_id: str,
    row_data: dict[str, Any] | pd.Series,
    work_order: dict[str, Any] | pd.Series | None = None,
    recommendation: dict[str, Any] | pd.Series | None = None,
) -> dict[str, object]:
    """Construye un Pasaporte de Decision auditable para una zona."""
    row_dict = row_data.to_dict() if isinstance(row_data, pd.Series) else dict(row_data)
    work_order_dict = (
        work_order.to_dict()
        if isinstance(work_order, pd.Series)
        else dict(work_order)
        if work_order is not None
        else {}
    )
    recommendation_dict = (
        recommendation.to_dict()
        if isinstance(recommendation, pd.Series)
        else dict(recommendation)
        if recommendation is not None
        else {}
    )

    digest_base = f"{zone_id}-{row_dict.get('final_impact_score', 0)}-{get_timestamp()}"
    decision_id = f"DP-{sanitize_filename(zone_id)}-{hashlib.md5(digest_base.encode('utf-8')).hexdigest()[:8]}"

    technical_evidence = [
        f"Score final de impacto: {row_dict.get('final_impact_score', 0)}",
        f"Clasificacion: {row_dict.get('classification', 'Sin clasificar')}",
        f"Severidad tecnica: {row_dict.get('technical_severity_score', 0)}",
        f"Demanda relativa: {row_dict.get('demand_score', 0)}",
    ]

    if work_order_dict:
        technical_evidence.append(
            f"Orden asociada: {work_order_dict.get('tipo_alerta', 'Sin alerta')} con prioridad {work_order_dict.get('prioridad', 'N/A')}."
        )

    contextual_evidence = []
    social_score = row_dict.get("social_criticality_score")
    if social_score is not None and not pd.isna(social_score):
        contextual_evidence.append(
            f"Criticidad territorial aproximada: {social_score}."
        )
    weather_score = row_dict.get("weather_context_score")
    if weather_score is not None and not pd.isna(weather_score):
        contextual_evidence.append(
            f"Contexto climatico ponderado: {weather_score}."
        )
    if recommendation_dict:
        contextual_evidence.append(
            recommendation_dict.get("justificacion", "Sin justificacion estrategica adicional.")
        )

    limitations = _to_list(row_dict.get("limitations"))
    data_used = _to_list(row_dict.get("evidence_fields"))
    missing_data = []

    if not data_used:
        missing_data.append("No se registraron campos de evidencia.")

    for limitation in limitations:
        if "No hay" in limitation or "insuficiente" in limitation.lower() or "limitado" in limitation.lower():
            missing_data.append(limitation)

    passport = {
        "decision_id": decision_id,
        "zona": zone_id,
        "clasificacion": row_dict.get("classification", "Sin clasificar"),
        "score_final": row_dict.get("final_impact_score", 0),
        "por_que_importa": (
            f"La zona '{zone_id}' aparece como {row_dict.get('classification', 'Sin clasificar')} "
            "en el indice de impacto ciudadano y operativo."
        ),
        "evidencia_tecnica": technical_evidence,
        "evidencia_contextual": contextual_evidence or ["Sin evidencia contextual adicional disponible."],
        "accion_recomendada": (
            work_order_dict.get("accion_recomendada")
            or recommendation_dict.get("tipo_recomendacion")
            or "Mantener monitoreo y validar en campo."
        ),
        "orden_trabajo_asociada": work_order_dict.get("id"),
        "nivel_confianza": _confidence_label(float(row_dict.get("data_confidence_score", 0) or 0)),
        "limitaciones": limitations or ["Sin limitaciones registradas."],
        "datos_usados": data_used,
        "datos_faltantes": sorted(set(missing_data)),
        "mensaje_para_jurado": (
            "Este pasaporte resume una decision auditable: evidencia tecnica, contexto territorial, "
            "limitaciones y accion recomendada sin afirmar causalidad no demostrada."
        ),
    }

    return passport


def generate_passports_for_top_zones(
    impact_scores_df: pd.DataFrame,
    work_orders: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """Genera pasaportes para las zonas de mayor score final."""
    if impact_scores_df.empty:
        return []

    work_orders = work_orders if work_orders is not None else pd.DataFrame()
    recommendations = recommendations if recommendations is not None else pd.DataFrame()

    top_zones_df = impact_scores_df.sort_values(
        by=["final_impact_score", "technical_severity_score"],
        ascending=[False, False],
    ).head(top_n)

    passports = []
    for _, row in top_zones_df.iterrows():
        zone_id = str(row["zona"])
        work_order_row = (
            work_orders[work_orders["zona"].astype(str) == zone_id].iloc[0]
            if not work_orders.empty and (work_orders["zona"].astype(str) == zone_id).any()
            else None
        )
        recommendation_row = (
            recommendations[recommendations["zona_o_territorio"].astype(str) == zone_id].iloc[0]
            if not recommendations.empty and (recommendations["zona_o_territorio"].astype(str) == zone_id).any()
            else None
        )
        passports.append(
            generate_decision_passport(
                zone_id=zone_id,
                row_data=row,
                work_order=work_order_row,
                recommendation=recommendation_row,
            )
        )

    return passports
