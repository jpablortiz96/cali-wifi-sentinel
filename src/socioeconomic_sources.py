from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data_loader import DataLoaderError, load_tabular_file
from src.external_sources import get_default_headers


SOCIOECONOMIC_COLUMN_ALIASES: dict[str, list[str]] = {
    "zona": ["zona", "zone", "zone_name", "nombre_zona", "ap_zone"],
    "comuna": ["comuna", "nombre_comuna"],
    "barrio": ["barrio", "nombre_barrio"],
    "corregimiento": ["corregimiento", "nombre_corregimiento"],
    "codigo_manzana": ["codigo_manzana", "manzana", "cod_manzana", "codigo_sector"],
    "municipio": ["municipio", "nombre_municipio", "ciudad"],
    "ipm": ["ipm", "indice_pobreza_multidimensional", "pobreza_multidimensional"],
    "nbi": ["nbi", "necesidades_basicas_insatisfechas"],
    "desempleo": ["desempleo", "tasa_desempleo", "unemployment_rate"],
    "poblacion": ["poblacion", "population", "habitantes"],
    "sisben_grupo_a_pct": ["sisben_grupo_a_pct", "sisben_a_pct", "grupo_a_pct", "sisben_a"],
    "sisben_grupo_b_pct": ["sisben_grupo_b_pct", "sisben_b_pct", "grupo_b_pct", "sisben_b"],
    "alfabetizacion_digital_proxy": [
        "alfabetizacion_digital_proxy",
        "alfabetizacion_digital",
        "digital_literacy_proxy",
        "conectividad_hogar_proxy",
    ],
    "fuente": ["fuente", "source", "origen"],
    "anio": ["anio", "year", "vigencia"],
}

AVAILABLE_SOCIOECONOMIC_INDICATORS = [
    "ipm",
    "nbi",
    "desempleo",
    "poblacion",
    "sisben_grupo_a_pct",
    "sisben_grupo_b_pct",
    "alfabetizacion_digital_proxy",
]

SENSITIVE_SOCIO_COLUMNS = [
    "nombre",
    "apellido",
    "cedula",
    "documento",
    "telefono",
    "correo",
    "email",
    "direccion",
    "client_id",
    "mac",
    "id_persona",
    "numero_ficha",
]


def _normalize_text(value: str) -> str:
    """Normaliza texto para comparar columnas con nombres heterogéneos."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return normalized


def _load_from_url(url: str) -> pd.DataFrame:
    """Descarga un archivo tabular remoto sin asumir un formato específico más allá de su nombre."""
    response = requests.get(url, headers=get_default_headers(), timeout=30)
    response.raise_for_status()
    remote_name = url.rstrip("/").rsplit("/", maxsplit=1)[-1] or "socioeconomic.csv"
    return load_tabular_file(remote_name, response.content)


def load_socioeconomic_file(uploaded_file_or_path: Any) -> pd.DataFrame:
    """Carga un dataset socioeconómico desde uploader, path local o URL pública."""
    if uploaded_file_or_path is None:
        return pd.DataFrame()

    if hasattr(uploaded_file_or_path, "getvalue") and hasattr(uploaded_file_or_path, "name"):
        file_bytes = uploaded_file_or_path.getvalue()
        return load_tabular_file(str(uploaded_file_or_path.name), file_bytes)

    raw_value = str(uploaded_file_or_path).strip()
    if not raw_value:
        return pd.DataFrame()

    if raw_value.lower().startswith(("http://", "https://")):
        return _load_from_url(raw_value)

    local_path = Path(raw_value)
    if not local_path.exists() or not local_path.is_file():
        raise DataLoaderError("El archivo socioeconómico indicado no existe o no es accesible.")

    return load_tabular_file(local_path.name, local_path.read_bytes())


def normalize_socioeconomic_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas frecuentes para que el resto del análisis pueda operar con un esquema consistente."""
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    normalized_df = dataframe.copy()
    normalized_map = {_normalize_text(column_name): column_name for column_name in normalized_df.columns}
    rename_map: dict[str, str] = {}

    for target_name, aliases in SOCIOECONOMIC_COLUMN_ALIASES.items():
        if target_name in normalized_df.columns:
            continue
        for alias in aliases:
            source_column = normalized_map.get(_normalize_text(alias))
            if source_column and source_column not in rename_map:
                rename_map[source_column] = target_name
                break

    if rename_map:
        normalized_df = normalized_df.rename(columns=rename_map)

    for numeric_column in AVAILABLE_SOCIOECONOMIC_INDICATORS + ["anio"]:
        if numeric_column in normalized_df.columns:
            normalized_df[numeric_column] = pd.to_numeric(normalized_df[numeric_column], errors="coerce")

    return normalized_df


def detect_socioeconomic_geo_level(dataframe: pd.DataFrame) -> str:
    """Detecta el nivel geográfico predominante del dataset socioeconómico."""
    if dataframe is None or dataframe.empty:
        return "desconocido"

    for level_name in ["zona", "comuna", "barrio", "corregimiento", "codigo_manzana", "municipio"]:
        if level_name in dataframe.columns and dataframe[level_name].notna().any():
            if level_name == "codigo_manzana":
                return "manzana"
            return level_name
    return "desconocido"


def validate_socioeconomic_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    """Valida estructura, nivel geográfico, indicadores y alertas de privacidad."""
    if dataframe is None or dataframe.empty:
        return {
            "is_valid": False,
            "level": "desconocido",
            "available_indicators": [],
            "warnings": ["El dataset socioeconómico está vacío."],
            "privacy_warnings": [],
        }

    normalized_df = normalize_socioeconomic_columns(dataframe)
    level = detect_socioeconomic_geo_level(normalized_df)
    available_indicators = [
        indicator_name
        for indicator_name in AVAILABLE_SOCIOECONOMIC_INDICATORS
        if indicator_name in normalized_df.columns and normalized_df[indicator_name].notna().any()
    ]

    warnings: list[str] = []
    if level == "desconocido":
        warnings.append("No se detectó un nivel geográfico claro para unir con zonas o territorios.")
    if not available_indicators:
        warnings.append("No se detectaron indicadores socioeconómicos agregados utilizables.")
    if len(normalized_df) < 3:
        warnings.append("El dataset socioeconómico tiene muy pocos registros para priorización territorial robusta.")

    privacy_warnings: list[str] = []
    normalized_columns = [_normalize_text(column_name) for column_name in normalized_df.columns]
    for sensitive_token in SENSITIVE_SOCIO_COLUMNS:
        if any(sensitive_token in column_name for column_name in normalized_columns):
            privacy_warnings.append(
                f"Se detectó una columna potencialmente sensible relacionada con `{sensitive_token}`. "
                "Usa solo agregados territoriales y evita procesar identificadores personales."
            )

    is_valid = level != "desconocido" and bool(available_indicators)
    return {
        "is_valid": is_valid,
        "level": level,
        "available_indicators": available_indicators,
        "warnings": warnings,
        "privacy_warnings": privacy_warnings,
    }
