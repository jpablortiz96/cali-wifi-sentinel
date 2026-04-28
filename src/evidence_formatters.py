from __future__ import annotations

from typing import Any

import pandas as pd


def truncate_text(value: object, max_chars: int = 180) -> str:
    """Recorta textos largos para que las tablas sean legibles."""
    if value is None:
        return ""

    if isinstance(value, list):
        text = " | ".join(str(item) for item in value if item not in (None, ""))
    else:
        text = str(value)

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def flatten_nested_dict(
    data: dict[str, Any],
    parent_key: str = "",
    sep: str = "_",
) -> dict[str, Any]:
    """Aplana diccionarios anidados para convertirlos en tablas simples."""
    flattened: dict[str, Any] = {}

    for key, value in (data or {}).items():
        safe_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_nested_dict(value, parent_key=safe_key, sep=sep))
        else:
            flattened[safe_key] = value

    return flattened


def safe_to_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper la UI."""
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        return pd.DataFrame({"valor": [truncate_text(item, max_chars=500) for item in data]})

    if isinstance(data, dict):
        if not data:
            return pd.DataFrame()

        if any(isinstance(value, (pd.DataFrame, list, dict)) for value in data.values()):
            flattened = flatten_nested_dict(data)
            return pd.DataFrame([flattened]) if flattened else pd.DataFrame()

        return pd.DataFrame([data])

    try:
        return pd.DataFrame(data)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _pick_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    """Devuelve la primera columna existente de una lista de candidatos."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _list_to_text(value: object, max_chars: int = 180) -> str:
    """Convierte listas o valores mixtos en texto corto."""
    return truncate_text(value, max_chars=max_chars)


def format_work_orders_for_display(work_orders: object) -> pd.DataFrame:
    """Normaliza ordenes de trabajo a una tabla ejecutiva."""
    dataframe = safe_to_dataframe(work_orders)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "Zona",
                "Tipo de alerta",
                "Prioridad",
                "Evidencia",
                "Accion recomendada",
                "Nivel de confianza",
                "Estado revision",
                "Score impacto",
            ]
        )

    output = pd.DataFrame()
    output["ID"] = dataframe.get(_pick_column(dataframe, ["id", "order_id"]), "")
    zone_column = _pick_column(dataframe, ["zone_name", "zona", "zone"])
    output["Zona"] = dataframe.get(zone_column, "")
    output["Tipo de alerta"] = dataframe.get(_pick_column(dataframe, ["tipo_alerta", "alert_type"]), "")
    output["Prioridad"] = dataframe.get(_pick_column(dataframe, ["prioridad", "priority"]), "")
    output["Evidencia"] = dataframe.get(_pick_column(dataframe, ["evidencia", "evidence"]), "").map(
        lambda value: truncate_text(value, max_chars=160)
    )
    output["Accion recomendada"] = dataframe.get(
        _pick_column(dataframe, ["accion_recomendada", "recommended_action"]),
        "",
    ).map(lambda value: truncate_text(value, max_chars=160))
    output["Nivel de confianza"] = dataframe.get(
        _pick_column(dataframe, ["nivel_confianza", "confidence_level"]),
        "",
    )
    output["Estado revision"] = dataframe.get(
        _pick_column(dataframe, ["estado_revision", "review_status"]),
        "",
    )

    score_column = _pick_column(dataframe, ["final_impact_score", "score_final", "impact_score"])
    if score_column:
        output["Score impacto"] = pd.to_numeric(dataframe[score_column], errors="coerce").round(2)
    else:
        output["Score impacto"] = None

    return output.fillna("")


def format_impact_scores_for_display(impact_scores: object) -> pd.DataFrame:
    """Normaliza el indice de impacto para vista ejecutiva."""
    dataframe = safe_to_dataframe(impact_scores)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Zona",
                "Score final",
                "Clasificacion",
                "Severidad tecnica",
                "Demanda",
                "Criticidad social",
                "Confianza datos",
                "Explicacion",
                "Limitaciones",
            ]
        )

    output = pd.DataFrame()
    zone_column = _pick_column(dataframe, ["zone_name", "zona", "zone"])
    output["Zona"] = dataframe.get(zone_column, "")
    output["Score final"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["final_impact_score", "score_final"]), None),
        errors="coerce",
    ).round(2)
    output["Clasificacion"] = dataframe.get(_pick_column(dataframe, ["classification", "clasificacion"]), "")
    output["Severidad tecnica"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["technical_severity_score", "severidad_tecnica"]), None),
        errors="coerce",
    ).round(2)
    output["Demanda"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["demand_score", "demanda"]), None),
        errors="coerce",
    ).round(2)
    output["Criticidad social"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["social_criticality_score", "criticidad_social"]), None),
        errors="coerce",
    ).round(2)
    output["Confianza datos"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["data_confidence_score", "confianza_datos"]), None),
        errors="coerce",
    ).round(2)
    output["Explicacion"] = dataframe.get(
        _pick_column(dataframe, ["explanation_short", "explicacion"]),
        "",
    ).map(lambda value: truncate_text(value, max_chars=160))
    output["Limitaciones"] = dataframe.get(
        _pick_column(dataframe, ["limitations", "limitaciones"]),
        "",
    ).map(lambda value: _list_to_text(value, max_chars=160))

    return output.fillna("")


def format_crew_plan_for_display(crew_plan: object) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Prepara el plan de cuadrillas para vista legible."""
    if isinstance(crew_plan, dict):
        recommended_df = safe_to_dataframe(crew_plan.get("recommended_zones"))
        waiting_df = safe_to_dataframe(crew_plan.get("waiting_zones"))
        coverage = crew_plan.get("coverage_territorial", "Sin datos")
        risk = crew_plan.get("riesgo_no_atencion", "Sin datos")
        explanation = crew_plan.get("explanation", "")
        summary_text = f"{coverage} {risk} {explanation}".strip()
        return recommended_df, waiting_df, summary_text

    dataframe = safe_to_dataframe(crew_plan)
    if dataframe.empty:
        return pd.DataFrame(), pd.DataFrame(), "No hay plan de cuadrillas disponible."

    return dataframe, pd.DataFrame(), "Se recibio un plan tabular simplificado."


def format_passports_for_display(passports: object) -> pd.DataFrame:
    """Resume pasaportes de decision en una tabla compacta."""
    dataframe = safe_to_dataframe(passports)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "ID decision",
                "Zona",
                "Clasificacion",
                "Score",
                "Accion recomendada",
                "Nivel confianza",
                "Por que importa",
                "Datos usados",
                "Limitaciones",
            ]
        )

    output = pd.DataFrame()
    output["ID decision"] = dataframe.get(_pick_column(dataframe, ["decision_id", "id_decision"]), "")
    output["Zona"] = dataframe.get(_pick_column(dataframe, ["zona", "zone"]), "")
    output["Clasificacion"] = dataframe.get(_pick_column(dataframe, ["clasificacion", "classification"]), "")
    output["Score"] = pd.to_numeric(
        dataframe.get(_pick_column(dataframe, ["score_final", "final_impact_score"]), None),
        errors="coerce",
    ).round(2)
    output["Accion recomendada"] = dataframe.get(
        _pick_column(dataframe, ["accion_recomendada", "recommended_action"]),
        "",
    ).map(lambda value: truncate_text(value, max_chars=140))
    output["Nivel confianza"] = dataframe.get(
        _pick_column(dataframe, ["nivel_confianza", "confidence_level"]),
        "",
    )
    output["Por que importa"] = dataframe.get(
        _pick_column(dataframe, ["por_que_importa", "why_it_matters"]),
        "",
    ).map(lambda value: truncate_text(value, max_chars=160))
    output["Datos usados"] = dataframe.get(
        _pick_column(dataframe, ["datos_usados", "evidence_fields"]),
        "",
    ).map(lambda value: _list_to_text(value, max_chars=120))
    output["Limitaciones"] = dataframe.get(
        _pick_column(dataframe, ["limitaciones", "limitations"]),
        "",
    ).map(lambda value: _list_to_text(value, max_chars=140))

    return output.fillna("")


def format_recommendations_for_display(recommendations: object) -> pd.DataFrame:
    """Normaliza recomendaciones estrategicas a columnas legibles."""
    dataframe = safe_to_dataframe(recommendations)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Zona o territorio",
                "Tipo recomendacion",
                "Justificacion",
                "Impacto estimado",
                "Esfuerzo estimado",
                "Nivel confianza",
            ]
        )

    output = pd.DataFrame()
    output["Zona o territorio"] = dataframe.get(
        _pick_column(dataframe, ["zona_o_territorio", "zona", "territorio"]),
        "",
    )
    output["Tipo recomendacion"] = dataframe.get(
        _pick_column(dataframe, ["tipo_recomendacion", "recommendation_type"]),
        "",
    )
    output["Justificacion"] = dataframe.get(
        _pick_column(dataframe, ["justificacion", "justification"]),
        "",
    ).map(lambda value: truncate_text(value, max_chars=180))
    output["Impacto estimado"] = dataframe.get(
        _pick_column(dataframe, ["impacto_estimado", "estimated_impact"]),
        "",
    )
    output["Esfuerzo estimado"] = dataframe.get(
        _pick_column(dataframe, ["esfuerzo_estimado", "estimated_effort"]),
        "",
    )
    output["Nivel confianza"] = dataframe.get(
        _pick_column(dataframe, ["nivel_confianza", "confidence_level"]),
        "",
    )

    return output.fillna("")


def format_quality_gate_for_display(
    quality_gate_report: object,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convierte quality gate a vistas simples."""
    report = quality_gate_report if isinstance(quality_gate_report, dict) else {}
    summary_dict = {
        "quality_gate": report.get("quality_gate", "Sin evaluar"),
        "demo_readiness": report.get("demo_readiness", "Sin evaluar"),
        "operational_status": report.get("operational_readiness", {}).get("operational_status", "Sin evaluar"),
        "operational_score": report.get("operational_readiness", {}).get("score", 0),
    }

    critical_issues_df = safe_to_dataframe(
        [{"problema_critico": truncate_text(item, max_chars=220)} for item in report.get("critical_issues", [])]
    )
    warnings_df = safe_to_dataframe(
        [{"advertencia": truncate_text(item, max_chars=220)} for item in report.get("warnings", [])]
    )
    recommendations_df = safe_to_dataframe(
        [{"recomendacion": truncate_text(item, max_chars=220)} for item in report.get("recommendations", [])]
    )
    return summary_dict, critical_issues_df, warnings_df, recommendations_df


def format_audit_log_for_display(audit_log: object) -> pd.DataFrame:
    """Deja la auditoria en una tabla limpia sin metadata cruda como vista principal."""
    dataframe = safe_to_dataframe(audit_log)
    if dataframe.empty:
        return pd.DataFrame(columns=["Timestamp", "Modulo", "Accion", "Estado", "Mensaje"])

    output = pd.DataFrame()
    output["Timestamp"] = dataframe.get(_pick_column(dataframe, ["timestamp", "Timestamp"]), "")
    output["Modulo"] = dataframe.get(_pick_column(dataframe, ["module", "Modulo"]), "")
    output["Accion"] = dataframe.get(_pick_column(dataframe, ["action", "Accion"]), "")
    output["Estado"] = dataframe.get(_pick_column(dataframe, ["status", "Estado"]), "")
    output["Mensaje"] = dataframe.get(_pick_column(dataframe, ["message", "Mensaje"]), "").map(
        lambda value: truncate_text(value, max_chars=180)
    )
    return output.fillna("")
