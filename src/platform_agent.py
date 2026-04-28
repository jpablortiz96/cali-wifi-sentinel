from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data_quality import build_dataset_profile
from src.gemini_client import generate_gemini_text, is_gemini_configured


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame de forma segura."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    if isinstance(data, dict):
        try:
            return pd.DataFrame([data])
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def _top_records(dataframe: pd.DataFrame | None, limit: int = 5) -> list[dict[str, object]]:
    """Extrae pocos registros para contexto conversacional."""
    if dataframe is None or dataframe.empty:
        return []

    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_platform_agent_context(
    results: dict[str, object],
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Construye contexto resumido del producto, dataset y análisis ejecutado."""
    profile = build_dataset_profile(df) if isinstance(df, pd.DataFrame) and not df.empty else {}
    impact_scores_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    recommendations_df = _safe_dataframe(results.get("recommendations"))
    operational_mart_df = _safe_dataframe(results.get("operational_mart"))
    meraki_anomalies_df = _safe_dataframe(results.get("meraki_anomalies"))
    passports = results.get("decision_passports", []) if isinstance(results.get("decision_passports"), list) else []
    human_review_df = _safe_dataframe(results.get("human_review_log"))
    audit_log_df = _safe_dataframe(results.get("audit_log"))

    return {
        "flow": [
            "1. Carga e inspección",
            "2. Mapeo de columnas",
            "3. Mission Control o Simulación Operativa",
            "4. Vista Ejecutiva 360",
            "5. Evidencia operativa",
        ],
        "available_modules": [
            "Carga e Inspección",
            "Mapeo de Columnas",
            "Mission Control",
            "Simulación Operativa",
            "Vista Ejecutiva 360",
            "Portal Ciudadano",
            "Experiencia Ciudadana",
            "Buzón Ciudadano",
            "Equidad Digital",
            "Agente Operativo",
            "Impacto Ciudadano",
            "Cuadrillas",
            "Pasaporte de Decisión",
            "Agente Estratégico",
            "Agente Conversacional",
            "Agente Ciudadano",
            "Validación Humana",
            "Blindaje Técnico",
            "Auditoría Operativa",
            "Paquete de Evidencia",
        ],
        "dataset_profile": {
            "rows": int(profile.get("total_rows", 0)),
            "columns": int(profile.get("total_columns", 0)),
            "column_names": profile.get("column_names", [])[:25],
            "candidate_columns": profile.get("candidate_columns", {}),
            "duplicated_rows": int(profile.get("duplicated_rows", 0)),
        },
        "schema_mapping": {key: value for key, value in (schema_mapping or {}).items() if value},
        "results_summary": {
            "trace_id": results.get("trace_id"),
            "confidence_level": results.get("confidence_level", "Baja"),
            "readiness": results.get("readiness", {}),
            "quality_gate_report": results.get("quality_gate_report", {}),
            "is_meraki_mode": bool(results.get("is_meraki_mode")),
            "wifi_package_summary": results.get("wifi_package_summary", {}) if isinstance(results.get("wifi_package_summary"), dict) else {},
            "work_orders_count": int(len(work_orders_df)),
            "impact_scores_count": int(len(impact_scores_df)),
            "recommendations_count": int(len(recommendations_df)),
            "operational_mart_count": int(len(operational_mart_df)),
            "meraki_anomalies_count": int(len(meraki_anomalies_df)),
            "passports_count": len(passports),
            "human_review_count": int(len(human_review_df)),
            "audit_events_count": int(len(audit_log_df)),
            "citizen_scores_count": int(len(_safe_dataframe(results.get("citizen_experience_scores")))),
            "citizen_feedback_count": int(len(_safe_dataframe(results.get("citizen_feedback")))),
        },
        "top_work_orders": _top_records(work_orders_df, limit=5),
        "top_impact_scores": _top_records(impact_scores_df, limit=5),
        "top_recommendations": _top_records(recommendations_df, limit=5),
        "top_citizen_scores": _top_records(_safe_dataframe(results.get("citizen_experience_scores")), limit=5),
        "top_operational_mart_rows": _top_records(operational_mart_df, limit=5),
        "top_meraki_anomalies": _top_records(meraki_anomalies_df, limit=5),
        "top_passports": passports[:5],
        "human_review_summary": {
            "pendientes": int(human_review_df["estado_revision"].astype(str).eq("pendiente").sum())
            if not human_review_df.empty and "estado_revision" in human_review_df.columns
            else 0,
            "aprobadas": int(human_review_df["estado_revision"].astype(str).eq("aprobada").sum())
            if not human_review_df.empty and "estado_revision" in human_review_df.columns
            else 0,
        },
        "audit_summary": _top_records(audit_log_df, limit=10),
        "limitations": results.get("limitations", []),
    }


def _fallback_platform_answer(question: str, context: dict[str, object]) -> str:
    """Respuesta básica cuando Gemini no está configurado."""
    lower_question = question.strip().lower()
    results_summary = context.get("results_summary", {})
    mapped_columns = context.get("schema_mapping", {})

    if any(term in lower_question for term in ["mission control", "ciclo"]):
        return (
            "Gemini no está configurado. Mission Control ejecuta readiness, órdenes, scoring, cuadrillas, "
            "pasaportes y validación técnica usando el dataset cargado."
        )
    if any(term in lower_question for term in ["mapeo", "columna", "schema"]):
        return (
            "Gemini no está configurado. El mapeo actual incluye estas columnas: "
            f"{', '.join(f'{k}={v}' for k, v in mapped_columns.items()) or 'ninguna aún'}."
        )
    if any(term in lower_question for term in ["orden", "alerta"]):
        return (
            "Gemini no está configurado. El sistema registra "
            f"{results_summary.get('work_orders_count', 0)} órdenes de trabajo en el resultado actual."
        )
    if any(term in lower_question for term in ["score", "impacto"]):
        return (
            "Gemini no está configurado. El resultado actual contiene "
            f"{results_summary.get('impact_scores_count', 0)} zonas con score de impacto."
        )
    if any(term in lower_question for term in ["validacion", "humana"]):
        return (
            "Gemini no está configurado. Usa la pestaña Validación Humana para aprobar, rechazar "
            "o marcar órdenes para visita técnica."
        )
    if any(term in lower_question for term in ["portal ciudadano", "experiencia ciudadana", "equidad digital", "buzon"]):
        return (
            "Gemini no está configurado. La capa ciudadana usa datos agregados para recomendar zonas, "
            "mostrar alertas, recibir feedback anónimo y calcular un proxy responsable de equidad digital."
        )
    if any(term in lower_question for term in ["auditoria", "trazabilidad"]):
        return (
            "Gemini no está configurado. La pestaña Auditoría Operativa resume eventos, warnings y errores "
            "del ciclo, la simulación y la revisión humana."
        )
    return (
        "Gemini no está configurado. Puedo orientarte con el flujo de uso: carga un dataset, mapea columnas, "
        "ejecuta Mission Control o Simulación Operativa y luego revisa Vista Ejecutiva 360, Validación Humana y Evidencia."
    )


def answer_platform_question(question: str, context: dict[str, object]) -> str:
    """Responde preguntas sobre la plataforma y el análisis operativo actual."""
    clean_question = str(question or "").strip()
    if not clean_question:
        return "Escribe una pregunta sobre la plataforma, el dataset cargado o el análisis generado."

    if not is_gemini_configured():
        return _fallback_platform_answer(clean_question, context)

    prompt = (
        "Eres el agente conversacional interno de Cali WiFi Sentinel 360.\n"
        "Solo puedes responder preguntas sobre la plataforma, el dataset cargado, el análisis realizado, "
        "las órdenes, scores, recomendaciones, pasaportes, validación y auditoría.\n"
        "Si la pregunta es externa o no relacionada, redirige amablemente.\n"
        "No inventes datos. Si falta evidencia, dilo.\n"
        "Responde en español y de forma breve.\n\n"
        "Contexto resumido de la plataforma y el análisis:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Pregunta del usuario: {clean_question}"
    )
    return generate_gemini_text(prompt)
