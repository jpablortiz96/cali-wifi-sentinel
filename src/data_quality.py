from __future__ import annotations

import json
import warnings
from typing import Any

import pandas as pd

from src.utils import normalize_text


CANDIDATE_KEYWORDS = {
    "fecha": ["fecha", "date", "dia", "periodo", "mes", "ano", "year"],
    "zona": [
        "zona",
        "punto",
        "sitio",
        "ubicacion",
        "lugar",
        "nombre",
        "area",
    ],
    "uso_conectividad": [
        "conexion",
        "conexiones",
        "usuarios",
        "usuario",
        "sesiones",
        "visitas",
        "uso",
        "trafico",
        "cantidad",
        "total",
    ],
    "ubicacion_geografica": [
        "lat",
        "latitud",
        "latitude",
        "lon",
        "lng",
        "longitud",
        "longitude",
        "coordenada",
    ],
    "territorio": ["comuna", "barrio", "localidad", "sector"],
}

DATE_LIKE_NAME_KEYWORDS = ["fecha", "date", "dia", "periodo", "mes", "ano", "year"]


def get_dataframe_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """Entrega un resumen general sin hacer supuestos sobre el esquema."""
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns.tolist()],
    }


def get_column_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla simple con nombre de columna y tipo de dato."""
    return pd.DataFrame(
        {
            "columna": [str(column) for column in dataframe.columns],
            "tipo_de_dato": [str(dtype) for dtype in dataframe.dtypes],
        }
    )


def get_null_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cuenta nulos y calcula su porcentaje por columna."""
    total_rows = len(dataframe)
    null_counts = dataframe.isna().sum()

    if total_rows == 0:
        null_percentages = pd.Series([0.0] * len(dataframe.columns), index=dataframe.columns)
    else:
        null_percentages = (null_counts / total_rows * 100).round(2)

    return pd.DataFrame(
        {
            "columna": [str(column) for column in dataframe.columns],
            "valores_nulos": null_counts.astype(int).values,
            "porcentaje_nulos": null_percentages.values,
        }
    )


def build_quality_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepara una vista consolidada de calidad basica por columna."""
    total_rows = len(dataframe)
    null_counts = dataframe.isna().sum()
    non_null_counts = dataframe.notna().sum()

    if total_rows == 0:
        null_percentages = pd.Series([0.0] * len(dataframe.columns), index=dataframe.columns)
    else:
        null_percentages = (null_counts / total_rows * 100).round(2)

    return pd.DataFrame(
        {
            "columna": [str(column) for column in dataframe.columns],
            "tipo_de_dato": [str(dtype) for dtype in dataframe.dtypes],
            "registros_no_nulos": non_null_counts.astype(int).values,
            "registros_nulos": null_counts.astype(int).values,
            "porcentaje_nulos": null_percentages.values,
        }
    )


def detect_candidate_columns(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Sugiere columnas candidatas por heuristica de nombres, sin confirmarlas."""
    suggestions: dict[str, list[str]] = {
        "fecha": [],
        "zona": [],
        "uso_conectividad": [],
        "ubicacion_geografica": [],
        "territorio": [],
    }

    for column in dataframe.columns:
        original_name = str(column)
        normalized_name = normalize_text(original_name)

        for group_name, keywords in CANDIDATE_KEYWORDS.items():
            if any(keyword in normalized_name for keyword in keywords):
                suggestions[group_name].append(original_name)

    return suggestions


def _looks_like_date_series(series: pd.Series) -> bool:
    """Intenta reconocer columnas tipo fecha con una muestra pequena."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return False

    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_sample = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    success_ratio = float(parsed_sample.notna().mean())
    return success_ratio >= 0.6


def _detect_date_like_columns(dataframe: pd.DataFrame) -> list[str]:
    """Detecta columnas con pinta de fecha usando nombre o parsabilidad."""
    detected_columns: list[str] = []

    for column in dataframe.columns:
        column_name = str(column)
        normalized_name = normalize_text(column_name)
        series = dataframe[column]

        if any(keyword in normalized_name for keyword in DATE_LIKE_NAME_KEYWORDS):
            detected_columns.append(column_name)
            continue

        if _looks_like_date_series(series):
            detected_columns.append(column_name)

    return detected_columns


def _safe_sample_rows(dataframe: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    """Devuelve una muestra pequena orientada a inspeccion manual."""
    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_dataset_profile(dataframe: pd.DataFrame) -> dict[str, object]:
    """Construye un perfil estructural resumido del dataset."""
    total_rows = int(dataframe.shape[0])
    null_counts = dataframe.isna().sum()

    if total_rows == 0:
        null_percentages = pd.Series([0.0] * len(dataframe.columns), index=dataframe.columns)
    else:
        null_percentages = (null_counts / total_rows * 100).round(2)

    numeric_columns = [str(column) for column in dataframe.select_dtypes(include=["number"]).columns]
    text_columns = [
        str(column)
        for column in dataframe.select_dtypes(include=["object", "string", "category"]).columns
    ]

    return {
        "total_rows": total_rows,
        "total_columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns.tolist()],
        "dtypes": {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        "nulls_by_column": {str(column): int(value) for column, value in null_counts.items()},
        "null_percentage_by_column": {
            str(column): float(value) for column, value in null_percentages.items()
        },
        "duplicated_rows": int(dataframe.duplicated().sum()),
        "sample_rows": _safe_sample_rows(dataframe, limit=5),
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "date_like_columns": _detect_date_like_columns(dataframe),
        "candidate_columns": detect_candidate_columns(dataframe),
    }


def _truncate_text(value: Any, max_length: int = 120) -> str:
    """Recorta valores largos para no inflar el prompt enviado al modelo."""
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def _limit_list(values: object, limit: int = 60) -> list[Any]:
    """Recorta listas largas para mantener el prompt acotado."""
    if not isinstance(values, list):
        return []

    if len(values) <= limit:
        return values

    return values[:limit] + [f"... ({len(values) - limit} elementos adicionales)"]


def _limit_dict(values: object, limit: int = 60) -> dict[str, Any]:
    """Recorta diccionarios largos preservando el orden de insercion."""
    if not isinstance(values, dict):
        return {}

    limited_items = list(values.items())[:limit]
    truncated_dict = {str(key): value for key, value in limited_items}

    if len(values) > limit:
        truncated_dict["__truncado__"] = (
            f"Se omitieron {len(values) - limit} elementos para reducir el tamano del prompt."
        )

    return truncated_dict


def profile_to_text(profile: dict[str, object]) -> str:
    """Convierte el perfil a texto compacto para analisis con Gemini."""
    sample_rows = profile.get("sample_rows", [])
    formatted_sample_rows = []

    for row in sample_rows[:5]:
        if not isinstance(row, dict):
            continue

        formatted_row = {
            str(key): _truncate_text(value)
            for key, value in row.items()
        }
        formatted_sample_rows.append(formatted_row)

    text_lines = [
        "Perfil estructural del dataset:",
        f"- Total de filas: {profile.get('total_rows', 0)}",
        f"- Total de columnas: {profile.get('total_columns', 0)}",
        f"- Filas duplicadas: {profile.get('duplicated_rows', 0)}",
        "",
        "Columnas detectadas:",
        json.dumps(_limit_list(profile.get("column_names", [])), ensure_ascii=False, indent=2),
        "",
        "Tipos de datos por columna:",
        json.dumps(_limit_dict(profile.get("dtypes", {})), ensure_ascii=False, indent=2),
        "",
        "Valores nulos por columna:",
        json.dumps(
            _limit_dict(profile.get("nulls_by_column", {})),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Porcentaje de nulos por columna:",
        json.dumps(
            _limit_dict(profile.get("null_percentage_by_column", {})),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Columnas numericas:",
        json.dumps(
            _limit_list(profile.get("numeric_columns", [])),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Columnas de texto:",
        json.dumps(
            _limit_list(profile.get("text_columns", [])),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Columnas con pinta de fecha:",
        json.dumps(
            _limit_list(profile.get("date_like_columns", [])),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Columnas candidatas sugeridas por heuristica:",
        json.dumps(
            _limit_dict(profile.get("candidate_columns", {})),
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Muestra maxima de 5 filas:",
        json.dumps(formatted_sample_rows, ensure_ascii=False, indent=2),
    ]

    return "\n".join(text_lines)
