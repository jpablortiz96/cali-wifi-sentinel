from __future__ import annotations

import json

import pandas as pd

from src.data_quality import build_dataset_profile
from src.gemini_client import generate_gemini_text
from src.schema_mapper import SchemaMapping


def _safe_numeric_summary(dataframe: pd.DataFrame, column_name: str | None) -> dict[str, object] | None:
    """Resume una columna numerica sin enviar toda la serie."""
    if not column_name:
        return None

    numeric_series = pd.to_numeric(dataframe[column_name], errors="coerce").dropna()
    if numeric_series.empty:
        return None

    return {
        "columna": column_name,
        "count": int(numeric_series.count()),
        "mean": round(float(numeric_series.mean()), 2),
        "median": round(float(numeric_series.median()), 2),
        "min": round(float(numeric_series.min()), 2),
        "max": round(float(numeric_series.max()), 2),
    }


def _dataframe_top_records(dataframe: pd.DataFrame | None, limit: int = 5) -> list[dict[str, object]]:
    """Convierte un DataFrame a pocos registros serializables."""
    if dataframe is None or dataframe.empty:
        return []

    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_technical_context(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    work_orders: pd.DataFrame | None,
    recommendations: pd.DataFrame | None,
    impact_scores: pd.DataFrame | None = None,
    decision_passports: list[dict[str, object]] | None = None,
    osm_context: pd.DataFrame | None = None,
    weather_context: pd.DataFrame | None = None,
    calendar_context: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Construye un contexto tecnico resumido apto para Gemini."""
    profile = build_dataset_profile(dataframe)
    mapped_columns = {
        field_key: value
        for field_key, value in schema_mapping.items()
        if value
    }

    quality_warnings: list[str] = []
    null_percentages = profile.get("null_percentage_by_column", {})

    for field_key, column_name in mapped_columns.items():
        null_percentage = float(null_percentages.get(column_name, 0.0))
        if null_percentage >= 30:
            quality_warnings.append(
                f"La columna mapeada '{column_name}' para '{field_key}' tiene {null_percentage:.2f}% de nulos."
            )

    if not schema_mapping.get("latitude_col") or not schema_mapping.get("longitude_col"):
        if not schema_mapping.get("territory_col"):
            quality_warnings.append(
                "No hay coordenadas ni territorio mapeado para analisis geoespacial detallado."
            )

    if not any(schema_mapping.get(field_key) for field_key in ["connections_col", "traffic_col", "status_col"]):
        quality_warnings.append(
            "No hay metricas operativas mapeadas para interpretar rendimiento tecnico."
        )

    aggregated_metrics: dict[str, object] = {
        "connections_summary": _safe_numeric_summary(dataframe, schema_mapping.get("connections_col")),
        "traffic_summary": _safe_numeric_summary(dataframe, schema_mapping.get("traffic_col")),
    }

    zone_col = schema_mapping.get("zone_col")
    if zone_col and schema_mapping.get("connections_col"):
        zone_connections = (
            pd.DataFrame(
                {
                    "zona": dataframe[zone_col].astype(str),
                    "connections_value": pd.to_numeric(dataframe[schema_mapping["connections_col"]], errors="coerce"),
                }
            )
            .dropna(subset=["connections_value"])
            .groupby("zona", dropna=False)["connections_value"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )
        aggregated_metrics["top_zones_by_connections"] = zone_connections

    if zone_col and schema_mapping.get("traffic_col"):
        zone_traffic = (
            pd.DataFrame(
                {
                    "zona": dataframe[zone_col].astype(str),
                    "traffic_value": pd.to_numeric(dataframe[schema_mapping["traffic_col"]], errors="coerce"),
                }
            )
            .dropna(subset=["traffic_value"])
            .groupby("zona", dropna=False)["traffic_value"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )
        aggregated_metrics["top_zones_by_traffic"] = zone_traffic

    if schema_mapping.get("status_col"):
        status_distribution = (
            dataframe[schema_mapping["status_col"]]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .to_dict()
        )
        aggregated_metrics["status_distribution_top"] = status_distribution

    context = {
        "dataset_summary": {
            "total_rows": int(profile.get("total_rows", 0)),
            "total_columns": int(profile.get("total_columns", 0)),
            "duplicated_rows": int(profile.get("duplicated_rows", 0)),
            "date_like_columns": profile.get("date_like_columns", []),
        },
        "mapped_columns": mapped_columns,
        "aggregated_metrics": aggregated_metrics,
        "top_work_orders": _dataframe_top_records(work_orders, limit=5),
        "top_recommendations": _dataframe_top_records(recommendations, limit=5),
        "top_impact_scores": _dataframe_top_records(impact_scores, limit=5),
        "top_decision_passports": decision_passports[:5] if decision_passports else [],
        "osm_context_summary": _dataframe_top_records(osm_context, limit=5),
        "weather_context_summary": _dataframe_top_records(weather_context, limit=5),
        "calendar_context_summary": _dataframe_top_records(calendar_context, limit=5),
        "data_quality_warnings": quality_warnings[:10],
    }

    # Resume limitaciones principales para que Gemini no sobreafirme.
    limitations = []
    if impact_scores is None or impact_scores.empty:
        limitations.append("No hay indice de impacto disponible.")
    if osm_context is None or osm_context.empty:
        limitations.append("No hay contexto OSM disponible.")
    if weather_context is None or weather_context.empty:
        limitations.append("No hay contexto climatico disponible.")
    if calendar_context is None or calendar_context.empty:
        limitations.append("No hay variables de calendario disponibles.")

    context["limitations"] = limitations
    return context


def build_orchestrated_context(results: dict[str, object]) -> dict[str, object]:
    """Resume el ciclo autonomo para preguntas tecnicas operativas."""
    readiness = results.get("readiness", {})
    work_orders = results.get("work_orders")
    recommendations = results.get("recommendations")
    impact_scores = results.get("impact_scores")
    crew_plan = results.get("crew_plan", {})
    decision_passports = results.get("decision_passports", [])
    event_log = results.get("agent_event_log", [])
    quality_gate_report = results.get("quality_gate_report", {})
    audit_summary = results.get("audit_summary", {})
    replay_timeline = results.get("replay_timeline")
    operational_mart = results.get("operational_mart")
    meraki_anomalies = results.get("meraki_anomalies")
    wifi_package_summary = results.get("wifi_package_summary", {})
    citizen_scores = results.get("citizen_experience_scores")
    digital_equity = results.get("digital_equity_proxy")
    citizen_feedback_summary = results.get("citizen_feedback_summary", {})

    recommended_df = crew_plan.get("recommended_zones") if isinstance(crew_plan, dict) else None
    waiting_df = crew_plan.get("waiting_zones") if isinstance(crew_plan, dict) else None
    replay_summary = _dataframe_top_records(replay_timeline, limit=5) if isinstance(replay_timeline, pd.DataFrame) else []

    return {
        "autonomous_cycle": True,
        "trace_id": results.get("trace_id"),
        "readiness": readiness,
        "quality_gate_report": quality_gate_report,
        "top_work_orders": _dataframe_top_records(work_orders, limit=5),
        "top_recommendations": _dataframe_top_records(recommendations, limit=5),
        "top_impact_scores": _dataframe_top_records(impact_scores, limit=5),
        "crew_plan": {
            "coverage_territorial": crew_plan.get("coverage_territorial") if isinstance(crew_plan, dict) else None,
            "riesgo_no_atencion": crew_plan.get("riesgo_no_atencion") if isinstance(crew_plan, dict) else None,
            "recommended_zones": _dataframe_top_records(recommended_df, limit=5),
            "waiting_zones": _dataframe_top_records(waiting_df, limit=5),
        },
        "top_decision_passports": decision_passports[:5] if decision_passports else [],
        "executive_summary": results.get("executive_summary", ""),
        "limitations": results.get("limitations", []),
        "confidence_level": results.get("confidence_level", "Baja"),
        "agent_event_log": event_log[:10],
        "audit_summary": audit_summary,
        "replay_timeline_summary": replay_summary,
        "is_meraki_mode": bool(results.get("is_meraki_mode")),
        "wifi_package_summary": wifi_package_summary if isinstance(wifi_package_summary, dict) else {},
        "operational_mart_top": _dataframe_top_records(operational_mart, limit=5),
        "meraki_anomalies_top": _dataframe_top_records(meraki_anomalies, limit=5),
        "citizen_scores_top": _dataframe_top_records(citizen_scores, limit=5),
        "digital_equity_top": _dataframe_top_records(digital_equity, limit=5),
        "citizen_feedback_summary": citizen_feedback_summary if isinstance(citizen_feedback_summary, dict) else {},
    }


def _context_to_text(context: dict[str, object]) -> str:
    """Convierte el contexto tecnico a texto compacto para Gemini."""
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


def answer_technical_question(question: str, context: dict[str, object]) -> str:
    """Responde preguntas tecnicas usando Gemini sobre un contexto resumido."""
    clean_question = question.strip()
    if not clean_question:
        return "Escribe una pregunta tecnica para consultar el contexto del dataset."

    prompt = (
        "Eres el copiloto tecnico de la red WiFi publica de Cali para Cali WiFi Sentinel 360.\n"
        "Responde solo con el contexto resumido entregado.\n"
        "No inventes datos.\n"
        "Diferencia observaciones de hipotesis.\n"
        "Si el contexto corresponde al paquete Meraki / Zonas WiFi Inteligentes, reconoce que el dataset incluye APs, eventos, clientes y metricas horarias, pero no coordenadas exactas de los APs.\n"
        "No afirmes causalidad climatica.\n"
        "No afirmes numero de personas afectadas si no existe poblacion observada.\n"
        "Puedes responder preguntas como: que hizo cada agente, que zona atender primero, "
        "por que una orden es prioritaria, que tan confiable es el analisis, que limitaciones "
        "tiene el dataset, como se alinea con el reto oficial y que trazabilidad existe.\n"
        "Responde en espanol, de forma corta y ejecutiva.\n"
        "Separa tu respuesta en estas secciones:\n"
        "- Hechos observados\n"
        "- Interpretacion\n"
        "- Recomendacion\n"
        "- Limitaciones\n"
        "- Nivel de confianza\n\n"
        "Contexto tecnico resumido:\n"
        f"{_context_to_text(context)}\n\n"
        f"Pregunta del usuario: {clean_question}"
    )

    return generate_gemini_text(prompt)
