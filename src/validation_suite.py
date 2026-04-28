from __future__ import annotations

from typing import Any

import pandas as pd

from src.gemini_client import is_gemini_configured
from src.schema_mapper import SchemaMapping, get_module_readiness


def _empty_schema_validation() -> dict[str, object]:
    return {
        "status": "error",
        "critical_issues": ["No hay dataset cargado para validar."],
        "warnings": [],
        "recommendations": ["Carga un CSV o Excel y define el mapeo de columnas."],
    }


def _safe_numeric_ratio(series: pd.Series) -> float:
    """Calcula que porcentaje de una serie se puede interpretar como numerico."""
    if series.empty:
        return 0.0
    numeric_values = pd.to_numeric(series, errors="coerce")
    return float(numeric_values.notna().mean() * 100)


def _safe_datetime_ratio(series: pd.Series) -> float:
    """Calcula que porcentaje de una serie se puede interpretar como fecha."""
    if series.empty:
        return 0.0
    parsed_dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return float(parsed_dates.notna().mean() * 100)


def _coordinates_look_valid(series: pd.Series, min_value: float, max_value: float) -> tuple[float, float]:
    """Entrega cobertura numerica y rango valido aproximado para coordenadas."""
    numeric_values = pd.to_numeric(series, errors="coerce")
    valid_numeric_ratio = float(numeric_values.notna().mean() * 100) if len(series) else 0.0
    valid_range_ratio = float(numeric_values.between(min_value, max_value).mean() * 100) if len(series) else 0.0
    return valid_numeric_ratio, valid_range_ratio


def validate_schema_mapping(
    dataframe: pd.DataFrame | None,
    schema_mapping: SchemaMapping,
) -> dict[str, object]:
    """Valida si el mapeo tiene consistencia minima para operar."""
    if dataframe is None or dataframe.empty:
        return _empty_schema_validation()

    critical_issues: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    for logical_field, column_name in schema_mapping.items():
        if column_name and column_name not in dataframe.columns:
            critical_issues.append(
                f"La columna seleccionada para '{logical_field}' no existe en el dataset actual."
            )

    zone_col = schema_mapping.get("zone_col")
    if zone_col and zone_col in dataframe.columns:
        non_null_ratio = float(dataframe[zone_col].notna().mean() * 100)
        if non_null_ratio == 0:
            critical_issues.append("La columna de zona esta completamente vacia.")
        elif non_null_ratio < 60:
            warnings.append(
                f"La columna de zona '{zone_col}' solo tiene {non_null_ratio:.2f}% de valores no nulos."
            )
    elif not zone_col:
        critical_issues.append("No hay columna de zona mapeada.")

    connections_col = schema_mapping.get("connections_col")
    if connections_col and connections_col in dataframe.columns:
        numeric_ratio = _safe_numeric_ratio(dataframe[connections_col])
        if numeric_ratio < 50:
            warnings.append(
                f"La columna '{connections_col}' parece poco numerica ({numeric_ratio:.2f}% interpretable)."
            )

    traffic_col = schema_mapping.get("traffic_col")
    if traffic_col and traffic_col in dataframe.columns:
        numeric_ratio = _safe_numeric_ratio(dataframe[traffic_col])
        if numeric_ratio < 50:
            warnings.append(
                f"La columna '{traffic_col}' parece poco numerica ({numeric_ratio:.2f}% interpretable)."
            )

    date_col = schema_mapping.get("date_col")
    if date_col and date_col in dataframe.columns:
        date_ratio = _safe_datetime_ratio(dataframe[date_col])
        if date_ratio < 50:
            warnings.append(
                f"La columna '{date_col}' no se convierte bien a fecha ({date_ratio:.2f}% interpretable)."
            )
    elif not date_col:
        recommendations.append("Mapear una columna de fecha mejoraria historico, simulacion y persistencia.")

    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    if latitude_col and latitude_col in dataframe.columns:
        numeric_ratio, range_ratio = _coordinates_look_valid(dataframe[latitude_col], -90, 90)
        if numeric_ratio < 70 or range_ratio < 70:
            warnings.append(
                f"La latitud '{latitude_col}' tiene cobertura/rango limitado ({numeric_ratio:.2f}% numerica, {range_ratio:.2f}% en rango)."
            )
    if longitude_col and longitude_col in dataframe.columns:
        numeric_ratio, range_ratio = _coordinates_look_valid(dataframe[longitude_col], -180, 180)
        if numeric_ratio < 70 or range_ratio < 70:
            warnings.append(
                f"La longitud '{longitude_col}' tiene cobertura/rango limitado ({numeric_ratio:.2f}% numerica, {range_ratio:.2f}% en rango)."
            )

    status_col = schema_mapping.get("status_col")
    if status_col and status_col in dataframe.columns:
        non_null_ratio = float(dataframe[status_col].notna().mean() * 100)
        if non_null_ratio < 40:
            warnings.append(
                f"La columna de estado '{status_col}' tiene pocos valores no nulos ({non_null_ratio:.2f}%)."
            )
    elif not status_col:
        recommendations.append("Mapear una columna de estado fortaleceria el agente operativo.")

    if critical_issues:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    if not recommendations and status == "ok":
        recommendations.append("El mapeo actual es util para seguir con analisis operativo controlado.")

    return {
        "status": status,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def validate_operational_readiness(
    dataframe: pd.DataFrame | None,
    schema_mapping: SchemaMapping,
) -> dict[str, object]:
    """Evalua que tan listo esta el dataset para operacion controlada."""
    if dataframe is None or dataframe.empty:
        return {
            "operational_status": "No listo",
            "score": 0,
            "ready_modules": [],
            "limited_modules": [],
            "blocked_modules": ["No hay dataset cargado."],
            "next_actions": ["Cargar un CSV o Excel valido."],
        }

    ready_modules: list[str] = []
    limited_modules: list[str] = []
    blocked_modules: list[str] = []
    next_actions: list[str] = []
    score = 0

    has_zone = bool(schema_mapping.get("zone_col"))
    has_operational_metric = bool(
        schema_mapping.get("connections_col")
        or schema_mapping.get("traffic_col")
        or schema_mapping.get("status_col")
    )
    has_date = bool(schema_mapping.get("date_col"))
    has_geography = bool(schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col"))
    has_territory = bool(schema_mapping.get("territory_col"))
    sufficient_volume = len(dataframe) >= 10

    if has_zone:
        score += 20
        ready_modules.append("Identificacion de zonas")
    else:
        blocked_modules.append("Identificacion de zonas")
        next_actions.append("Mapear una columna de zona.")

    if has_operational_metric:
        score += 25
        ready_modules.append("Deteccion operativa basica")
    else:
        blocked_modules.append("Deteccion operativa basica")
        next_actions.append("Mapear conexiones, trafico o estado.")

    if has_date:
        score += 15
        ready_modules.append("Historico / persistencia")
    else:
        limited_modules.append("Historico / persistencia")
        next_actions.append("Agregar o mapear fecha para ordenar eventos y analizar persistencia.")

    if has_geography:
        score += 20
        ready_modules.append("Contexto geoespacial puntual")
    elif has_territory:
        score += 12
        limited_modules.append("Contexto geoespacial puntual")
        ready_modules.append("Priorizacion territorial agregada")
        next_actions.append("Agregar latitud y longitud para contexto espacial mas preciso.")
    else:
        limited_modules.append("Priorizacion geoespacial")
        next_actions.append("Agregar coordenadas o territorio.")

    if sufficient_volume:
        score += 10
        ready_modules.append("Volumen minimo de evidencia")
    else:
        limited_modules.append("Volumen minimo de evidencia")
        next_actions.append("Cargar mas registros para mejorar estabilidad del analisis.")

    module_readiness = get_module_readiness(dataframe, schema_mapping)
    if module_readiness["conversational"]["ready"]:
        score += 10
        ready_modules.append("Agente conversacional tecnico")
    else:
        limited_modules.append("Agente conversacional tecnico")

    score = min(int(score), 100)
    if not has_zone or not has_operational_metric:
        status = "No listo"
    elif score >= 70:
        status = "Listo"
    else:
        status = "Limitado"

    if not next_actions:
        next_actions.append("El dataset ya puede usarse para una simulacion operativa controlada.")

    return {
        "operational_status": status,
        "score": score,
        "ready_modules": ready_modules,
        "limited_modules": limited_modules,
        "blocked_modules": blocked_modules,
        "next_actions": next_actions,
    }


def validate_analysis_outputs(results: dict[str, Any] | None) -> dict[str, object]:
    """Valida si la ejecucion genero outputs utiles para una demo operativa."""
    expected_fields = {
        "work_orders": "Ordenes de trabajo",
        "impact_scores": "Indice de impacto",
        "crew_plan": "Plan de cuadrillas",
        "decision_passports": "Pasaportes de decision",
    }

    available_outputs: list[str] = []
    missing_outputs: list[str] = []
    explanations: list[str] = []

    if not results:
        missing_outputs = list(expected_fields.values())
        explanations.append("Aun no se ha ejecutado un ciclo o analisis operativo.")
        return {
            "status": "warning",
            "available_outputs": available_outputs,
            "missing_outputs": missing_outputs,
            "explanations": explanations,
        }

    for key, label in expected_fields.items():
        value = results.get(key)
        if isinstance(value, pd.DataFrame):
            if value.empty:
                missing_outputs.append(label)
                explanations.append(f"{label}: vacio o sin evidencia suficiente.")
            else:
                available_outputs.append(label)
        elif isinstance(value, list):
            if value:
                available_outputs.append(label)
            else:
                missing_outputs.append(label)
                explanations.append(f"{label}: no se generaron elementos aun.")
        elif isinstance(value, dict):
            if value:
                available_outputs.append(label)
            else:
                missing_outputs.append(label)
                explanations.append(f"{label}: no contiene resultados utiles.")
        else:
            missing_outputs.append(label)
            explanations.append(f"{label}: no esta presente en la ejecucion actual.")

    status = "ok" if not missing_outputs else "warning"
    return {
        "status": status,
        "available_outputs": available_outputs,
        "missing_outputs": missing_outputs,
        "explanations": explanations,
    }


def build_quality_gate_report(
    dataframe: pd.DataFrame | None,
    schema_mapping: SchemaMapping,
    results: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Construye una compuerta de calidad antes de operar o demostrar."""
    if dataframe is None or dataframe.empty:
        return {
            "quality_gate": "Bloqueado",
            "demo_readiness": "Baja",
            "critical_issues": ["No hay dataset cargado."],
            "warnings": [],
            "recommendations": ["Cargar un CSV o Excel antes de ejecutar el sistema."],
            "operational_readiness": validate_operational_readiness(dataframe, schema_mapping),
            "schema_validation": _empty_schema_validation(),
            "output_validation": validate_analysis_outputs(results),
        }

    schema_validation = validate_schema_mapping(dataframe, schema_mapping)
    operational_readiness = validate_operational_readiness(dataframe, schema_mapping)
    output_validation = validate_analysis_outputs(results)

    critical_issues = list(schema_validation["critical_issues"])
    warnings = list(schema_validation["warnings"])
    recommendations = list(schema_validation["recommendations"])

    has_zone = bool(schema_mapping.get("zone_col"))
    has_metric = bool(
        schema_mapping.get("connections_col")
        or schema_mapping.get("traffic_col")
        or schema_mapping.get("status_col")
    )

    if not has_zone or not has_metric:
        quality_gate = "Bloqueado"
        demo_readiness = "Baja"
        if not has_zone:
            critical_issues.append("Sin zona mapeada no se puede priorizar operativamente.")
        if not has_metric:
            critical_issues.append("Sin metrica operativa no se puede detectar degradacion basica.")
    else:
        quality_gate = "Aprobado"
        demo_readiness = "Alta" if operational_readiness["score"] >= 70 else "Media"

    if not schema_mapping.get("latitude_col") or not schema_mapping.get("longitude_col"):
        if quality_gate != "Bloqueado":
            quality_gate = "Aprobado con advertencias"
        warnings.append("No hay coordenadas completas; el contexto geoespacial puntual queda limitado.")

    if not schema_mapping.get("date_col"):
        if quality_gate != "Bloqueado":
            quality_gate = "Aprobado con advertencias"
        warnings.append("No hay fecha; la simulacion operativa usara orden de filas y no temporalidad real.")

    if not is_gemini_configured():
        if quality_gate != "Bloqueado":
            quality_gate = "Aprobado con advertencias"
        warnings.append("Gemini no esta configurado; el chat tecnico funcionara solo cuando se agregue la API key.")

    if "tipo_dato" in dataframe.columns and dataframe["tipo_dato"].astype(str).eq("SINTETICO_NO_OFICIAL").any():
        if quality_gate != "Bloqueado":
            quality_gate = "Aprobado con advertencias"
        warnings.append("El dataset activo contiene datos sinteticos y no debe presentarse como informacion oficial.")

    duplicate_ratio = float(dataframe.duplicated().mean() * 100) if len(dataframe) else 0.0
    if duplicate_ratio > 25:
        warnings.append(
            f"El dataset tiene {duplicate_ratio:.2f}% de filas duplicadas; conviene depurarlo antes de operar."
        )

    if output_validation["status"] == "warning" and results:
        warnings.extend(output_validation["explanations"])

    recommendations.extend(operational_readiness["next_actions"])
    recommendations = list(dict.fromkeys(recommendations))
    warnings = list(dict.fromkeys(warnings))
    critical_issues = list(dict.fromkeys(critical_issues))

    return {
        "quality_gate": quality_gate,
        "demo_readiness": demo_readiness,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "recommendations": recommendations,
        "operational_readiness": operational_readiness,
        "schema_validation": schema_validation,
        "output_validation": output_validation,
    }
