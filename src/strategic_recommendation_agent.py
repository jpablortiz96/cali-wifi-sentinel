from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.gemini_client import generate_gemini_text, is_gemini_configured
from src.profile_storage import convert_to_serializable
from src.strategic_recommendations import RECOMMENDATION_COLUMNS, generate_strategic_recommendations


EXTRA_RECOMMENDATION_COLUMNS = ["evidencia_usada", "limitaciones", "source"]


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper."""
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


def _top_records(dataframe: pd.DataFrame | None, limit: int = 10) -> list[dict[str, object]]:
    """Extrae pocos registros serializables para prompts y resúmenes."""
    if dataframe is None or dataframe.empty:
        return []

    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def _ensure_recommendation_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Asegura columnas estándar de recomendaciones estratégicas."""
    normalized = dataframe.copy() if isinstance(dataframe, pd.DataFrame) else pd.DataFrame()
    for column_name in RECOMMENDATION_COLUMNS + EXTRA_RECOMMENDATION_COLUMNS:
        if column_name not in normalized.columns:
            normalized[column_name] = [] if column_name in {"evidencia_usada", "limitaciones"} else ""
    return normalized[RECOMMENDATION_COLUMNS + EXTRA_RECOMMENDATION_COLUMNS]


def build_recommendation_context(
    results: dict[str, object],
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Construye contexto resumido real para recomendaciones estratégicas."""
    impact_scores_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    osm_context_df = _safe_dataframe(results.get("osm_context"))
    weather_context_df = _safe_dataframe(results.get("weather_context"))
    recommendations_df = _safe_dataframe(results.get("recommendations"))
    decision_passports = results.get("decision_passports", []) if isinstance(results.get("decision_passports"), list) else []
    audit_log = results.get("audit_log", []) if isinstance(results.get("audit_log"), list) else []
    operational_mart_df = _safe_dataframe(results.get("operational_mart"))
    social_roi_df = _safe_dataframe(results.get("social_roi_scores"))
    digital_equity_df = _safe_dataframe(results.get("digital_equity_proxy"))
    wifi_package_summary = results.get("wifi_package_summary", {})
    socioeconomic_validation = (
        results.get("socioeconomic_validation", {})
        if isinstance(results.get("socioeconomic_validation"), dict)
        else {}
    )

    top_scores = []
    if not impact_scores_df.empty and "final_impact_score" in impact_scores_df.columns:
        top_scores = _top_records(
            impact_scores_df.sort_values("final_impact_score", ascending=False),
            limit=10,
        )

    top_orders = []
    if not work_orders_df.empty:
        sort_column = "final_impact_score" if "final_impact_score" in work_orders_df.columns else None
        if sort_column:
            top_orders = _top_records(work_orders_df.sort_values(sort_column, ascending=False), limit=10)
        else:
            top_orders = _top_records(work_orders_df, limit=10)

    top_social_roi = []
    if not social_roi_df.empty and "social_roi_score" in social_roi_df.columns:
        top_social_roi = _top_records(
            social_roi_df.sort_values("social_roi_score", ascending=False),
            limit=10,
        )

    osm_summary = {}
    if not osm_context_df.empty:
        numeric_cols = [
            column_name
            for column_name in [
                "poi_total",
                "education_count",
                "health_count",
                "transport_count",
                "parks_count",
                "civic_count",
                "community_count",
                "social_criticality_score",
            ]
            if column_name in osm_context_df.columns
        ]
        if numeric_cols:
            osm_summary = osm_context_df[numeric_cols].mean(numeric_only=True).round(2).to_dict()

    weather_summary = {}
    if not weather_context_df.empty:
        numeric_cols = [
            column_name
            for column_name in [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
            ]
            if column_name in weather_context_df.columns
        ]
        if numeric_cols:
            weather_summary = weather_context_df[numeric_cols].mean(numeric_only=True).round(2).to_dict()
        if "weather_classification" in weather_context_df.columns:
            weather_summary["weather_context_top"] = (
                weather_context_df["weather_classification"].astype(str).value_counts().head(5).to_dict()
            )

    missing_data: list[str] = []
    mapped_fields = {key: value for key, value in (schema_mapping or {}).items() if value}
    if not mapped_fields.get("zone_col"):
        missing_data.append("No hay columna de zona mapeada.")
    if not mapped_fields.get("latitude_col") or not mapped_fields.get("longitude_col"):
        missing_data.append("No hay coordenadas completas para criticidad territorial fina.")
    if not mapped_fields.get("date_col"):
        missing_data.append("No hay fecha mapeada para analizar persistencia temporal.")
    if impact_scores_df.empty:
        missing_data.append("No hay impact scores disponibles.")
    if osm_context_df.empty:
        missing_data.append("No hay contexto OSM disponible.")
    if weather_context_df.empty:
        missing_data.append("No hay contexto climático disponible.")
    if social_roi_df.empty:
        missing_data.append("No hay score de retorno social cargado o calculado.")
    if not socioeconomic_validation.get("is_valid"):
        missing_data.append("No hay validación socioeconómica suficiente para orientar inversión social.")

    return {
        "trace_id": results.get("trace_id"),
        "confidence_level": results.get("confidence_level", "Baja"),
        "readiness": results.get("readiness", {}),
        "quality_gate_report": results.get("quality_gate_report", {}),
        "socioeconomic_validation": socioeconomic_validation,
        "schema_mapping": mapped_fields,
        "dataset_summary": {
            "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
            "columns": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
        },
        "top_work_orders": top_orders,
        "top_impact_scores": top_scores,
        "top_existing_recommendations": _top_records(recommendations_df, limit=10),
        "top_social_roi": top_social_roi,
        "digital_equity_top": _top_records(digital_equity_df, limit=10),
        "osm_context_summary": osm_summary,
        "weather_context_summary": weather_summary,
        "decision_passports_top": decision_passports[:5],
        "crew_plan": results.get("crew_plan", {}),
        "audit_log_recent": audit_log[-20:],
        "limitations": results.get("limitations", []),
        "missing_data": missing_data,
        "is_meraki_mode": bool(results.get("is_meraki_mode")),
        "wifi_package_summary": wifi_package_summary if isinstance(wifi_package_summary, dict) else {},
        "operational_mart_top": _top_records(operational_mart_df, limit=10),
    }


def generate_strategic_recommendations_with_gemini(context: dict[str, object]) -> str:
    """Solicita a Gemini recomendaciones estratégicas en JSON estricto."""
    prompt = (
        "Eres un agente estratégico para una red WiFi pública.\n"
        "Genera recomendaciones de mantenimiento e inversión basadas únicamente en el contexto entregado.\n"
        "No inventes valores, coordenadas ni población.\n"
        "Si el contexto proviene del paquete Meraki / Zonas WiFi Inteligentes, interpreta los resultados como evidencia por AP y zona a partir de eventos, clientes, estados y métricas horarias.\n"
        "No afirmes causalidad climática.\n"
        "Si la evidencia es limitada, dilo.\n"
        "Responde SOLO con JSON válido y estricto.\n"
        "Usa este esquema exacto:\n"
        "{\n"
        '  "recommendations": [\n'
        "    {\n"
        '      "zona_o_territorio": "...",\n'
        '      "tipo_recomendacion": "mantenimiento | inversion | validacion_datos | capacidad | ubicacion | comunicacion_ciudadana | auditoria",\n'
        '      "justificacion": "...",\n'
        '      "impacto_estimado": "Alto | Medio | Bajo",\n'
        '      "esfuerzo_estimado": "Alto | Medio | Bajo",\n'
        '      "nivel_confianza": "Alta | Media | Baja",\n'
        '      "evidencia_usada": ["..."],\n'
        '      "limitaciones": ["..."]\n'
        "    }\n"
        "  ],\n"
        '  "summary": "...",\n'
        '  "limitations": ["..."]\n'
        "}\n\n"
        "Contexto real resumido:\n"
        f"{json.dumps(convert_to_serializable(context), ensure_ascii=False, indent=2)}"
    )
    return generate_gemini_text(prompt)


def parse_gemini_recommendations(raw_text: str) -> dict[str, object]:
    """Parsea JSON de Gemini sin romper la app si la salida es imperfecta."""
    clean_text = str(raw_text or "").strip()
    if not clean_text:
        return {
            "recommendations_df": _ensure_recommendation_schema(pd.DataFrame()),
            "summary": "",
            "limitations": ["Gemini no devolvió contenido."],
            "raw_text": clean_text,
            "parsed": False,
        }

    candidates = [clean_text]
    json_match = re.search(r"\{[\s\S]*\}", clean_text)
    if json_match:
        candidates.append(json_match.group(0))

    payload: dict[str, object] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return {
            "recommendations_df": _ensure_recommendation_schema(pd.DataFrame()),
            "summary": "",
            "limitations": ["No fue posible parsear JSON estructurado desde Gemini."],
            "raw_text": clean_text,
            "parsed": False,
        }

    recommendations = payload.get("recommendations", [])
    recommendations_df = _safe_dataframe(recommendations)
    if not recommendations_df.empty:
        recommendations_df["source"] = "gemini"
    recommendations_df = _ensure_recommendation_schema(recommendations_df)

    return {
        "recommendations_df": recommendations_df,
        "summary": str(payload.get("summary", "")).strip(),
        "limitations": payload.get("limitations", []) if isinstance(payload.get("limitations"), list) else [],
        "raw_text": clean_text,
        "parsed": True,
    }


def generate_deterministic_recommendations_fallback(context: dict[str, object]) -> dict[str, object]:
    """Genera recomendaciones básicas cuando Gemini no está disponible."""
    recommendations: list[dict[str, object]] = []
    top_scores = context.get("top_impact_scores", []) if isinstance(context.get("top_impact_scores"), list) else []
    top_orders = context.get("top_work_orders", []) if isinstance(context.get("top_work_orders"), list) else []
    top_social_roi = context.get("top_social_roi", []) if isinstance(context.get("top_social_roi"), list) else []
    missing_data = context.get("missing_data", []) if isinstance(context.get("missing_data"), list) else []

    for row in top_social_roi[:3]:
        zona = str(row.get("zone_name", row.get("zona", "Zona prioritaria social")))
        score = float(row.get("social_roi_score", 0) or 0)
        label = str(row.get("social_roi_label", "Retorno social medio"))
        recommendations.append(
            {
                "zona_o_territorio": zona,
                "tipo_recomendacion": "inversion",
                "justificacion": (
                    f"La zona muestra un Social ROI de {score:.2f} ({label}). "
                    "Conviene priorizar mejoras no solo por falla técnica sino por retorno social esperado."
                ),
                "impacto_estimado": "Alto" if score >= 60 else "Medio",
                "esfuerzo_estimado": "Medio",
                "nivel_confianza": context.get("confidence_level", "Baja"),
                "evidencia_usada": [f"Social ROI {score:.2f}", label],
                "limitaciones": missing_data[:3],
                "source": "fallback",
            }
        )

    for row in top_scores[:5]:
        zona = str(row.get("zona", "Zona prioritaria"))
        classification = str(row.get("classification", "Observacion"))
        final_score = float(row.get("final_impact_score", 0) or 0)
        recommendations.append(
            {
                "zona_o_territorio": zona,
                "tipo_recomendacion": "mantenimiento" if classification in {"Critico", "Alto"} else "auditoria",
                "justificacion": (
                    f"La zona presenta score final {final_score:.2f} y clasificación {classification}. "
                    "Se recomienda intervención o validación según la evidencia actual."
                ),
                "impacto_estimado": "Alto" if classification in {"Critico", "Alto"} else "Medio",
                "esfuerzo_estimado": "Medio",
                "nivel_confianza": context.get("confidence_level", "Baja"),
                "evidencia_usada": [f"Score final {final_score:.2f}", f"Clasificación {classification}"],
                "limitaciones": missing_data[:2],
                "source": "fallback",
            }
        )

    if not recommendations and top_orders:
        for row in top_orders[:5]:
            zona = str(row.get("zona", "Zona operativa"))
            prioridad = str(row.get("prioridad", "Observacion"))
            recommendations.append(
                {
                    "zona_o_territorio": zona,
                    "tipo_recomendacion": "validacion_datos",
                    "justificacion": (
                        f"La zona tiene una orden preliminar con prioridad {prioridad}. "
                        "Conviene validar el caso y completar evidencia operativa."
                    ),
                    "impacto_estimado": "Medio",
                    "esfuerzo_estimado": "Bajo",
                    "nivel_confianza": context.get("confidence_level", "Baja"),
                    "evidencia_usada": [f"Prioridad {prioridad}"],
                    "limitaciones": missing_data[:2],
                    "source": "fallback",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "zona_o_territorio": "Global",
                "tipo_recomendacion": "validacion_datos",
                "justificacion": (
                    "Gemini no está configurado. Se muestra una recomendación determinística básica: "
                    "completar mapeo, validar la calidad de datos y volver a ejecutar Mission Control."
                ),
                "impacto_estimado": "Medio",
                "esfuerzo_estimado": "Bajo",
                "nivel_confianza": context.get("confidence_level", "Baja"),
                "evidencia_usada": ["Fallback determinístico"],
                "limitaciones": missing_data[:3],
                "source": "fallback",
            }
        )

    recommendations_df = _ensure_recommendation_schema(pd.DataFrame(recommendations))
    return {
        "recommendations_df": recommendations_df,
        "summary": "Gemini no está configurado. Se muestran recomendaciones determinísticas básicas.",
        "limitations": missing_data,
        "raw_text": "",
        "parsed": True,
    }


def get_or_generate_strategic_recommendations(
    results: dict[str, object],
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
    force_refresh: bool = False,
) -> dict[str, object]:
    """Obtiene recomendaciones desde caché de sesión o genera nuevas al solicitarlo."""
    context = build_recommendation_context(results, df=df, schema_mapping=schema_mapping)
    context_payload = json.dumps(convert_to_serializable(context), ensure_ascii=False, sort_keys=True)
    context_hash = hashlib.md5(context_payload.encode("utf-8")).hexdigest()
    trace_id = str(results.get("trace_id") or "sin-trace")
    cache_key = f"strategic_recommendations::{trace_id}::{context_hash}"

    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]

    if force_refresh and is_gemini_configured():
        raw_text = generate_strategic_recommendations_with_gemini(context)
        parsed = parse_gemini_recommendations(raw_text)
        source = "gemini" if parsed.get("parsed") else "gemini_unparsed"
        if parsed["recommendations_df"].empty:
            fallback = generate_deterministic_recommendations_fallback(context)
            parsed = fallback
            source = "fallback_after_gemini"
    else:
        parsed = generate_deterministic_recommendations_fallback(context)
        source = "fallback"

    payload = {
        "recommendations_df": parsed.get("recommendations_df", _ensure_recommendation_schema(pd.DataFrame())),
        "summary": parsed.get("summary", ""),
        "limitations": parsed.get("limitations", []),
        "raw_text": parsed.get("raw_text", ""),
        "parsed": bool(parsed.get("parsed", False)),
        "source": source,
        "cache_key": cache_key,
        "trace_id": trace_id,
    }
    st.session_state[cache_key] = payload
    st.session_state["latest_strategic_recommendations_key"] = cache_key
    return payload
