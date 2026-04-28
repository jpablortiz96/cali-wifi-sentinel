from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema_mapper import SchemaMapping, get_module_readiness


READINESS_WEIGHTS = {
    "zone_col": 15,
    "date_col": 15,
    "connections_col": 15,
    "traffic_col": 10,
    "status_col": 15,
    "coordinates": 15,
    "territory_col": 10,
    "data_quality": 5,
}


def _null_percentage(dataframe: pd.DataFrame, column_name: str | None) -> float:
    """Calcula porcentaje de nulos de una columna mapeada."""
    if not column_name or column_name not in dataframe.columns or dataframe.empty:
        return 100.0
    return float(dataframe[column_name].isna().mean() * 100)


def _build_check(
    item: str,
    status: str,
    weight: int,
    reason: str,
) -> dict[str, object]:
    """Construye un check individual del readiness score."""
    return {
        "item": item,
        "status": status,
        "weight": weight,
        "reason": reason,
    }


def _score_from_status(status: str, weight: int) -> float:
    """Asigna puntaje a un check segun su estado."""
    if status == "ok":
        return float(weight)
    if status == "partial":
        return round(weight * 0.5, 2)
    return 0.0


def _classify_score(score: float) -> str:
    """Clasifica el readiness global."""
    if score >= 85:
        return "Excelente"
    if score >= 65:
        return "Bueno"
    if score >= 40:
        return "Limitado"
    return "Insuficiente"


def _column_check(
    dataframe: pd.DataFrame,
    column_name: str | None,
    weight: int,
    item: str,
    missing_reason: str,
    partial_reason_prefix: str,
    ok_reason_prefix: str,
) -> dict[str, object]:
    """Evalua una columna simple por existencia y nulos."""
    if not column_name or column_name not in dataframe.columns:
        return _build_check(item, "missing", weight, missing_reason)

    null_pct = _null_percentage(dataframe, column_name)
    if null_pct >= 40:
        return _build_check(
            item,
            "partial",
            weight,
            f"{partial_reason_prefix} pero tiene {null_pct:.2f}% de nulos.",
        )

    return _build_check(
        item,
        "ok",
        weight,
        f"{ok_reason_prefix} con {null_pct:.2f}% de nulos.",
    )


def _date_check(dataframe: pd.DataFrame, date_col: str | None) -> dict[str, object]:
    """Evalua utilidad temporal del dataset."""
    weight = READINESS_WEIGHTS["date_col"]
    if not date_col or date_col not in dataframe.columns:
        return _build_check(
            "Columna de fecha / historico",
            "missing",
            weight,
            "No hay fecha mapeada, por lo que no se puede analizar persistencia ni historico.",
        )

    parsed_dates = pd.to_datetime(dataframe[date_col], errors="coerce", dayfirst=True)
    valid_ratio = float(parsed_dates.notna().mean() * 100) if len(parsed_dates) else 0.0
    unique_dates = int(parsed_dates.dropna().nunique())

    if valid_ratio < 60 or unique_dates < 2:
        return _build_check(
            "Columna de fecha / historico",
            "partial",
            weight,
            f"Existe '{date_col}', pero solo {valid_ratio:.2f}% pudo interpretarse como fecha y hay {unique_dates} fechas unicas.",
        )

    return _build_check(
        "Columna de fecha / historico",
        "ok",
        weight,
        f"Existe '{date_col}' con {valid_ratio:.2f}% de fechas validas y {unique_dates} fechas unicas.",
    )


def _connections_check(
    dataframe: pd.DataFrame,
    connections_col: str | None,
    traffic_col: str | None,
) -> dict[str, object]:
    """Evalua si existe una metrica de uso/conexiones suficientemente util."""
    weight = READINESS_WEIGHTS["connections_col"]

    if not connections_col and traffic_col:
        return _build_check(
            "Columna de conexiones o uso",
            "partial",
            weight,
            "No hay conexiones mapeadas, pero hay trafico como proxy parcial de uso.",
        )

    return _column_check(
        dataframe=dataframe,
        column_name=connections_col,
        weight=weight,
        item="Columna de conexiones o uso",
        missing_reason="No hay una columna clara de conexiones o uso mapeada.",
        partial_reason_prefix=f"Existe '{connections_col}'",
        ok_reason_prefix=f"Existe '{connections_col}'",
    )


def _coordinates_check(
    dataframe: pd.DataFrame,
    latitude_col: str | None,
    longitude_col: str | None,
) -> dict[str, object]:
    """Evalua georreferenciacion puntual."""
    weight = READINESS_WEIGHTS["coordinates"]

    if not latitude_col and not longitude_col:
        return _build_check(
            "Latitud y longitud",
            "missing",
            weight,
            "No hay coordenadas mapeadas para contexto espacial de precision.",
        )

    if not latitude_col or not longitude_col:
        return _build_check(
            "Latitud y longitud",
            "partial",
            weight,
            "Solo una de las coordenadas esta mapeada; la georreferenciacion es incompleta.",
        )

    lat_nulls = _null_percentage(dataframe, latitude_col)
    lon_nulls = _null_percentage(dataframe, longitude_col)
    if lat_nulls >= 40 or lon_nulls >= 40:
        return _build_check(
            "Latitud y longitud",
            "partial",
            weight,
            f"Existen coordenadas, pero latitud tiene {lat_nulls:.2f}% de nulos y longitud {lon_nulls:.2f}%.",
        )

    return _build_check(
        "Latitud y longitud",
        "ok",
        weight,
        f"Existen coordenadas con nulos controlados: latitud {lat_nulls:.2f}% y longitud {lon_nulls:.2f}%.",
    )


def _data_quality_check(dataframe: pd.DataFrame) -> dict[str, object]:
    """Evalua duplicados y carga global de nulos."""
    weight = READINESS_WEIGHTS["data_quality"]
    if dataframe.empty:
        return _build_check(
            "Calidad basica de datos",
            "missing",
            weight,
            "El dataset esta vacio.",
        )

    duplicate_ratio = float(dataframe.duplicated().mean() * 100)
    null_ratio = float(dataframe.isna().mean().mean() * 100)

    if duplicate_ratio > 25 or null_ratio > 35:
        return _build_check(
            "Calidad basica de datos",
            "partial",
            weight,
            f"Hay {duplicate_ratio:.2f}% de duplicados y una carga promedio de nulos de {null_ratio:.2f}%.",
        )

    return _build_check(
        "Calidad basica de datos",
        "ok",
        weight,
        f"Duplicados {duplicate_ratio:.2f}% y nulos promedio {null_ratio:.2f}%.",
    )


def _alignment_status(ready: bool, missing_count: int) -> str:
    """Traduce readiness del modulo a etiqueta de negocio."""
    if ready:
        return "Listo"
    if missing_count <= 1:
        return "Limitado"
    return "Incompleto"


def calculate_data_readiness(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
) -> dict[str, Any]:
    """Calcula que tan preparado esta el dataset para el reto oficial."""
    checks: list[dict[str, object]] = []

    checks.append(
        _column_check(
            dataframe=dataframe,
            column_name=schema_mapping.get("zone_col"),
            weight=READINESS_WEIGHTS["zone_col"],
            item="Columna de zona",
            missing_reason="No hay una columna de zona mapeada.",
            partial_reason_prefix=f"Existe '{schema_mapping.get('zone_col')}'",
            ok_reason_prefix=f"Existe '{schema_mapping.get('zone_col')}'",
        )
    )
    checks.append(_date_check(dataframe, schema_mapping.get("date_col")))
    checks.append(
        _connections_check(
            dataframe,
            schema_mapping.get("connections_col"),
            schema_mapping.get("traffic_col"),
        )
    )
    checks.append(
        _column_check(
            dataframe=dataframe,
            column_name=schema_mapping.get("traffic_col"),
            weight=READINESS_WEIGHTS["traffic_col"],
            item="Columna de trafico",
            missing_reason="No hay una columna de trafico mapeada.",
            partial_reason_prefix=f"Existe '{schema_mapping.get('traffic_col')}'",
            ok_reason_prefix=f"Existe '{schema_mapping.get('traffic_col')}'",
        )
    )
    checks.append(
        _column_check(
            dataframe=dataframe,
            column_name=schema_mapping.get("status_col"),
            weight=READINESS_WEIGHTS["status_col"],
            item="Estado del punto de acceso",
            missing_reason="No hay una columna de estado del punto de acceso mapeada.",
            partial_reason_prefix=f"Existe '{schema_mapping.get('status_col')}'",
            ok_reason_prefix=f"Existe '{schema_mapping.get('status_col')}'",
        )
    )
    checks.append(
        _coordinates_check(
            dataframe,
            schema_mapping.get("latitude_col"),
            schema_mapping.get("longitude_col"),
        )
    )
    checks.append(
        _column_check(
            dataframe=dataframe,
            column_name=schema_mapping.get("territory_col"),
            weight=READINESS_WEIGHTS["territory_col"],
            item="Territorio / comuna / barrio",
            missing_reason="No hay una columna territorial mapeada.",
            partial_reason_prefix=f"Existe '{schema_mapping.get('territory_col')}'",
            ok_reason_prefix=f"Existe '{schema_mapping.get('territory_col')}'",
        )
    )
    checks.append(_data_quality_check(dataframe))

    total_score = round(sum(_score_from_status(check["status"], int(check["weight"])) for check in checks), 2)
    classification = _classify_score(total_score)

    strengths = [
        str(check["reason"])
        for check in checks
        if check["status"] == "ok"
    ]
    gaps = [
        str(check["reason"])
        for check in checks
        if check["status"] != "ok"
    ]

    recommended_next_actions = []
    for check in checks:
        item = str(check["item"])
        status = str(check["status"])
        if status == "missing":
            recommended_next_actions.append(f"Completar {item.lower()}.")
        elif status == "partial":
            recommended_next_actions.append(f"Mejorar calidad o cobertura de {item.lower()}.")

    if not recommended_next_actions:
        recommended_next_actions.append("El dataset ya tiene una base solida para la demo del reto.")

    module_readiness = get_module_readiness(dataframe, schema_mapping)
    official_alignment = {
        "agente_operativo": _alignment_status(
            bool(module_readiness["operational"]["ready"]),
            len(module_readiness["operational"]["missing"]),
        ),
        "agente_conversacional": _alignment_status(
            bool(module_readiness["conversational"]["ready"]),
            len(module_readiness["conversational"]["missing"]),
        ),
        "agente_estrategico": _alignment_status(
            bool(module_readiness["strategic"]["ready"]),
            len(module_readiness["strategic"]["missing"]),
        ),
    }

    return {
        "score": total_score,
        "classification": classification,
        "checks": checks,
        "strengths": strengths,
        "gaps": gaps,
        "recommended_next_actions": recommended_next_actions,
        "official_challenge_alignment": official_alignment,
    }
