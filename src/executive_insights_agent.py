from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard_insights import (
    build_executive_dashboard_summary,
    build_next_best_actions,
    build_risk_alerts,
    build_top_findings,
)
from src.gemini_client import generate_gemini_text, is_gemini_configured
from src.profile_storage import convert_to_serializable


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def _top_records(dataframe: pd.DataFrame | None, limit: int) -> list[dict[str, object]]:
    """Extrae pocos registros serializables para evitar prompts demasiado largos."""
    if dataframe is None or dataframe.empty:
        return []

    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_insights_context(
    results: dict[str, object],
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Construye el contexto ejecutivo a partir de resultados reales ya generados."""
    impact_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    recommendations_df = _safe_dataframe(results.get("recommendations"))
    replay_timeline_df = _safe_dataframe(results.get("replay_timeline"))
    human_review_df = _safe_dataframe(results.get("human_review_log"))
    operational_mart_df = _safe_dataframe(results.get("operational_mart"))
    meraki_anomalies_df = _safe_dataframe(results.get("meraki_anomalies"))
    passports = results.get("decision_passports", []) if isinstance(results.get("decision_passports"), list) else []
    audit_log = results.get("audit_log", []) if isinstance(results.get("audit_log"), list) else []

    top_zones = []
    if not impact_df.empty and "final_impact_score" in impact_df.columns:
        top_zones = _top_records(impact_df.sort_values("final_impact_score", ascending=False), limit=10)

    top_orders = []
    if not work_orders_df.empty:
        sort_column = "final_impact_score" if "final_impact_score" in work_orders_df.columns else None
        if sort_column:
            top_orders = _top_records(work_orders_df.sort_values(sort_column, ascending=False), limit=10)
        else:
            top_orders = _top_records(work_orders_df, limit=10)

    top_recommendations = _top_records(recommendations_df, limit=5)
    top_passports = passports[:5]
    recent_audit = audit_log[-20:]

    return {
        "trace_id": results.get("trace_id"),
        "confidence_level": results.get("confidence_level", "Baja"),
        "readiness": results.get("readiness", {}),
        "quality_gate_report": results.get("quality_gate_report", {}),
        "executive_summary_base": build_executive_dashboard_summary(results, df=df),
        "top_findings_base": build_top_findings(results)[:10],
        "risk_alerts_base": build_risk_alerts(results)[:10],
        "next_best_actions_base": build_next_best_actions(results)[:10],
        "top_work_orders": top_orders,
        "top_zones": top_zones,
        "crew_plan": results.get("crew_plan", {}),
        "top_recommendations": top_recommendations,
        "top_decision_passports": top_passports,
        "replay_timeline": _top_records(replay_timeline_df, limit=10),
        "human_review_summary": {
            "total": int(len(human_review_df)),
            "pendientes": int(human_review_df["estado_revision"].astype(str).eq("pendiente").sum())
            if not human_review_df.empty and "estado_revision" in human_review_df.columns
            else 0,
            "aprobadas": int(human_review_df["estado_revision"].astype(str).eq("aprobada").sum())
            if not human_review_df.empty and "estado_revision" in human_review_df.columns
            else 0,
        },
        "audit_log_recent": recent_audit,
        "limitations": results.get("limitations", []),
        "schema_mapping": {
            key: value
            for key, value in (schema_mapping or {}).items()
            if value
        },
        "is_meraki_mode": bool(results.get("is_meraki_mode")),
        "wifi_package_summary": results.get("wifi_package_summary", {}) if isinstance(results.get("wifi_package_summary"), dict) else {},
        "operational_mart_top": _top_records(operational_mart_df, limit=10),
        "meraki_anomalies_top": _top_records(meraki_anomalies_df, limit=10),
        "dataset_summary": {
            "total_rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "total_columns": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
        },
    }


def generate_executive_insights_with_gemini(context: dict[str, object]) -> str:
    """Genera análisis ejecutivo con Gemini usando solo contexto resumido real."""
    prompt = (
        "Eres un agente ejecutivo de analisis operativo para una red WiFi publica.\n"
        "Debes analizar unicamente los datos entregados.\n"
        "No inventes valores.\n"
        "No afirmes causalidad tecnica si no existe evidencia.\n"
        "Si el contexto proviene del paquete Meraki / Zonas WiFi Inteligentes, interpreta los resultados como evidencia por AP y zona. "
        "Ese paquete incluye eventos, clientes, access points y métricas horarias, pero no garantiza coordenadas exactas del AP.\n"
        "Diferencia hechos observados, interpretacion, alertas, recomendaciones, limitaciones y nivel de confianza.\n"
        "Responde en espanol, en Markdown claro y ejecutivo.\n\n"
        "La salida debe tener exactamente estas secciones:\n"
        "## Hechos observados\n"
        "## Alertas relevantes\n"
        "## Interpretacion operativa\n"
        "## Recomendaciones inmediatas\n"
        "## Riesgos y limitaciones\n"
        "## Nivel de confianza\n\n"
        "Contexto resumido de resultados reales del sistema:\n"
        f"{json.dumps(convert_to_serializable(context), ensure_ascii=False, indent=2)}"
    )
    return generate_gemini_text(prompt)


def build_deterministic_insights_fallback(context: dict[str, object]) -> str:
    """Fallback determinístico cuando Gemini no está disponible o no se solicita."""
    findings = context.get("top_findings_base", [])
    alerts = context.get("risk_alerts_base", [])
    actions = context.get("next_best_actions_base", [])
    limitations = context.get("limitations", [])

    def bullet_block(items: list[str], empty_message: str) -> str:
        if not items:
            return f"- {empty_message}"
        return "\n".join(f"- {item}" for item in items[:5])

    return (
        "Gemini no está configurado o no se ha solicitado análisis asistido. "
        "Se muestra análisis determinístico básico.\n\n"
        "## Hechos observados\n"
        f"{bullet_block(findings, 'No hay hallazgos suficientes todavía.')}\n\n"
        "## Alertas relevantes\n"
        f"{bullet_block(alerts, 'No se detectan alertas adicionales con la evidencia actual.')}\n\n"
        "## Interpretacion operativa\n"
        f"- {context.get('executive_summary_base', 'No hay resumen operativo disponible.')}\n\n"
        "## Recomendaciones inmediatas\n"
        f"{bullet_block(actions, 'Completar el flujo operativo para generar acciones.')}\n\n"
        "## Riesgos y limitaciones\n"
        f"{bullet_block(limitations, 'El análisis depende del dataset cargado y del mapeo actual.')}\n\n"
        "## Nivel de confianza\n"
        f"- {context.get('confidence_level', 'Baja')}"
    )


def get_or_generate_dashboard_insights(
    results: dict[str, object],
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
    force_refresh: bool = False,
) -> dict[str, object]:
    """Obtiene hallazgos cacheados o genera análisis nuevo solo cuando se solicita."""
    context = build_insights_context(results, df=df, schema_mapping=schema_mapping)
    context_payload = json.dumps(convert_to_serializable(context), ensure_ascii=False, sort_keys=True)
    context_hash = hashlib.md5(context_payload.encode("utf-8")).hexdigest()
    trace_id = str(results.get("trace_id") or "sin-trace")
    cache_key = f"dashboard_insights::{trace_id}::{context_hash}"

    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]

    if force_refresh and is_gemini_configured():
        markdown = generate_executive_insights_with_gemini(context)
        source = "gemini"
    else:
        markdown = build_deterministic_insights_fallback(context)
        source = "fallback"

    payload = {
        "markdown": markdown,
        "source": source,
        "cache_key": cache_key,
        "trace_id": trace_id,
        "context_hash": context_hash,
    }
    st.session_state[cache_key] = payload
    st.session_state["latest_dashboard_insights_key"] = cache_key
    return payload
