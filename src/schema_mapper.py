from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import normalize_text


SchemaMapping = dict[str, str | None]


MAPPING_FIELD_CONFIG = {
    "date_col": {
        "label": "Fecha",
        "keywords": ["fecha", "date", "dia", "periodo", "mes", "ano", "year"],
        "preferred_dtypes": ["datetime", "object", "string"],
    },
    "zone_col": {
        "label": "Zona",
        "keywords": ["zona", "punto", "sitio", "ubicacion", "lugar", "nombre", "area"],
        "preferred_dtypes": ["object", "string", "category"],
    },
    "connections_col": {
        "label": "Conexiones",
        "keywords": [
            "conexion",
            "conexiones",
            "usuarios",
            "usuario",
            "sesiones",
            "visitas",
            "cantidad",
            "total",
        ],
        "preferred_dtypes": ["number"],
    },
    "traffic_col": {
        "label": "Trafico",
        "keywords": [
            "trafico",
            "traffic",
            "bytes",
            "mb",
            "gb",
            "consumo",
            "datos",
            "download",
            "upload",
            "ancho",
        ],
        "preferred_dtypes": ["number"],
    },
    "status_col": {
        "label": "Estado del punto de acceso",
        "keywords": [
            "estado",
            "status",
            "health",
            "salud",
            "activo",
            "inactivo",
            "offline",
            "error",
            "down",
        ],
        "preferred_dtypes": ["object", "string", "category"],
    },
    "latitude_col": {
        "label": "Latitud",
        "keywords": ["latitud", "latitude", "lat", "coord_y", "y"],
        "preferred_dtypes": ["number"],
    },
    "longitude_col": {
        "label": "Longitud",
        "keywords": ["longitud", "longitude", "lng", "lon", "coord_x", "x"],
        "preferred_dtypes": ["number"],
    },
    "territory_col": {
        "label": "Comuna/barrio/sector",
        "keywords": ["comuna", "barrio", "localidad", "sector", "territorio"],
        "preferred_dtypes": ["object", "string", "category"],
    },
}


def list_available_columns(dataframe: pd.DataFrame) -> list[str]:
    """Entrega las columnas disponibles en el dataset."""
    return [str(column) for column in dataframe.columns.tolist()]


def _dtype_matches(column_dtype: Any, preferred_groups: list[str]) -> bool:
    """Evalua si el tipo de dato encaja con la expectativa del campo."""
    if "number" in preferred_groups and pd.api.types.is_numeric_dtype(column_dtype):
        return True

    if "datetime" in preferred_groups and pd.api.types.is_datetime64_any_dtype(column_dtype):
        return True

    if any(group in preferred_groups for group in ["object", "string", "category"]):
        return bool(
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
            or isinstance(column_dtype, pd.CategoricalDtype)
        )

    return False


def suggest_columns_for_field(dataframe: pd.DataFrame, field_key: str) -> list[str]:
    """Sugiere columnas ordenadas por afinidad heuristica para un campo de mapeo."""
    config = MAPPING_FIELD_CONFIG[field_key]
    scored_candidates: list[tuple[int, str]] = []

    for column in dataframe.columns:
        column_name = str(column)
        normalized_name = normalize_text(column_name)
        score = 0

        for keyword in config["keywords"]:
            if normalized_name == keyword:
                score += 8
            elif normalized_name.startswith(keyword):
                score += 6
            elif keyword in normalized_name:
                score += 4

        if _dtype_matches(dataframe[column].dtype, config["preferred_dtypes"]):
            score += 2

        if field_key == "date_col" and pd.api.types.is_datetime64_any_dtype(dataframe[column].dtype):
            score += 3

        if score > 0:
            scored_candidates.append((score, column_name))

    scored_candidates.sort(key=lambda item: (-item[0], item[1].lower()))
    return [column_name for _, column_name in scored_candidates]


def suggest_schema_mapping(dataframe: pd.DataFrame) -> SchemaMapping:
    """Construye un mapeo sugerido, siempre editable por el usuario."""
    suggestions: SchemaMapping = {field_key: None for field_key in MAPPING_FIELD_CONFIG}
    reserved_columns: set[str] = set()

    for field_key in MAPPING_FIELD_CONFIG:
        candidates = suggest_columns_for_field(dataframe, field_key)

        for candidate in candidates:
            if candidate not in reserved_columns:
                suggestions[field_key] = candidate
                reserved_columns.add(candidate)
                break

    return suggestions


def build_schema_mapping(raw_mapping: dict[str, str | None]) -> SchemaMapping:
    """Normaliza el mapeo seleccionado en la interfaz."""
    normalized_mapping: SchemaMapping = {}

    for field_key in MAPPING_FIELD_CONFIG:
        value = raw_mapping.get(field_key)
        if value in (None, "", "Sin seleccionar"):
            normalized_mapping[field_key] = None
        else:
            normalized_mapping[field_key] = str(value)

    return normalized_mapping


def validate_operational_mapping(schema_mapping: SchemaMapping) -> dict[str, object]:
    """Valida si el Agente Operativo cuenta con lo minimo para trabajar."""
    has_zone = bool(schema_mapping.get("zone_col"))
    available_metrics = [
        field_key
        for field_key in ["connections_col", "traffic_col", "status_col"]
        if schema_mapping.get(field_key)
    ]

    missing = []
    if not has_zone:
        missing.append("zone_col")
    if not available_metrics:
        missing.append("connections_col o traffic_col o status_col")

    return {
        "ready": has_zone and bool(available_metrics),
        "missing": missing,
        "available_metrics": available_metrics,
    }


def validate_conversational_mapping(
    dataframe: pd.DataFrame | None,
    schema_mapping: SchemaMapping,
) -> dict[str, object]:
    """Valida si el Agente Conversacional puede responder con contexto tecnico."""
    dataset_loaded = dataframe is not None and not dataframe.empty
    mapped_fields = [field_key for field_key, value in schema_mapping.items() if value]

    missing = []
    if not dataset_loaded:
        missing.append("dataset cargado")
    if dataset_loaded and not mapped_fields:
        missing.append("al menos un campo mapeado")

    return {
        "ready": dataset_loaded and bool(mapped_fields),
        "missing": missing,
        "mapped_fields": mapped_fields,
    }


def validate_strategic_mapping(schema_mapping: SchemaMapping) -> dict[str, object]:
    """Valida si el Agente Estrategico tiene base minima para recomendar."""
    has_zone = bool(schema_mapping.get("zone_col"))
    has_coordinates = bool(schema_mapping.get("latitude_col")) and bool(
        schema_mapping.get("longitude_col")
    )
    has_territory = bool(schema_mapping.get("territory_col"))

    missing = []
    if not has_zone:
        missing.append("zone_col")
    if not (has_coordinates or has_territory):
        missing.append("latitude_col y longitude_col o territory_col")

    return {
        "ready": has_zone and (has_coordinates or has_territory),
        "missing": missing,
        "has_coordinates": has_coordinates,
        "has_territory": has_territory,
    }


def get_module_readiness(
    dataframe: pd.DataFrame | None,
    schema_mapping: SchemaMapping,
) -> dict[str, dict[str, object]]:
    """Consolida el estado de alistamiento de los modulos oficiales."""
    return {
        "operational": validate_operational_mapping(schema_mapping),
        "conversational": validate_conversational_mapping(dataframe, schema_mapping),
        "strategic": validate_strategic_mapping(schema_mapping),
    }
