from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Any

import pandas as pd

from src.external_sources import get_with_cache


FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
]


def has_weather_requirements(schema_mapping: dict[str, str | None]) -> bool:
    """Indica si hay coordenadas suficientes para consultar clima contextual."""
    return bool(schema_mapping.get("latitude_col")) and bool(schema_mapping.get("longitude_col"))


def _parse_date_value(date_value: object) -> date_type | None:
    """Convierte una fecha arbitraria a `date` si es posible."""
    if date_value in (None, "", pd.NaT):
        return None

    parsed_date = pd.to_datetime(date_value, errors="coerce", dayfirst=True)
    if pd.isna(parsed_date):
        return None

    return parsed_date.date()


def _safe_float(value: object) -> float | None:
    """Convierte a float devolviendo None si el valor no es valido."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_current_weather(response_data: dict[str, Any]) -> dict[str, object]:
    """Normaliza una respuesta actual o forecast a formato comun."""
    current_data = response_data.get("current", {}) or {}

    weather = {
        "temperature_2m": _safe_float(current_data.get("temperature_2m")),
        "relative_humidity_2m": _safe_float(current_data.get("relative_humidity_2m")),
        "precipitation": _safe_float(current_data.get("precipitation")),
        "rain": _safe_float(current_data.get("rain")),
        "wind_speed_10m": _safe_float(current_data.get("wind_speed_10m")),
        "weather_context_available": bool(current_data),
        "weather_note": (
            "Contexto meteorologico actual obtenido desde Open-Meteo."
            if current_data
            else "Open-Meteo no devolvio datos actuales utilizables."
        ),
    }
    return weather


def _extract_hourly_weather(response_data: dict[str, Any]) -> dict[str, object]:
    """Agrega una respuesta horaria a un resumen diario/contextual simple."""
    hourly_data = response_data.get("hourly", {}) or {}

    def series_mean(field_name: str) -> float | None:
        values = pd.Series(hourly_data.get(field_name, []), dtype="float64").dropna()
        if values.empty:
            return None
        return round(float(values.mean()), 2)

    def series_sum(field_name: str) -> float | None:
        values = pd.Series(hourly_data.get(field_name, []), dtype="float64").dropna()
        if values.empty:
            return None
        return round(float(values.sum()), 2)

    def series_max(field_name: str) -> float | None:
        values = pd.Series(hourly_data.get(field_name, []), dtype="float64").dropna()
        if values.empty:
            return None
        return round(float(values.max()), 2)

    is_available = any(hourly_data.get(field_name) for field_name in WEATHER_VARIABLES)

    return {
        "temperature_2m": series_mean("temperature_2m"),
        "relative_humidity_2m": series_mean("relative_humidity_2m"),
        "precipitation": series_sum("precipitation"),
        "rain": series_sum("rain"),
        "wind_speed_10m": series_max("wind_speed_10m"),
        "weather_context_available": is_available,
        "weather_note": (
            "Contexto meteorologico agregado desde Open-Meteo para la fecha consultada."
            if is_available
            else "Open-Meteo no devolvio series horarias utilizables para la fecha consultada."
        ),
    }


def get_weather_for_point(lat: float, lon: float, date_value: object = None) -> dict[str, object]:
    """Consulta Open-Meteo de forma opcional y devuelve un contexto normalizado."""
    parsed_date = _parse_date_value(date_value)
    today = datetime.utcnow().date()

    if parsed_date and parsed_date < today:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": parsed_date.isoformat(),
            "end_date": parsed_date.isoformat(),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "auto",
        }
        response_data = get_with_cache(
            source_name="open_meteo_archive",
            url=ARCHIVE_API_URL,
            params=params,
            ttl_hours=24 * 180,
        )
        if isinstance(response_data, dict):
            return _extract_hourly_weather(response_data)
    else:
        if parsed_date:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": parsed_date.isoformat(),
                "end_date": parsed_date.isoformat(),
                "hourly": ",".join(WEATHER_VARIABLES),
                "timezone": "auto",
            }
            response_data = get_with_cache(
                source_name="open_meteo_forecast_by_date",
                url=FORECAST_API_URL,
                params=params,
                ttl_hours=6,
            )
            if isinstance(response_data, dict):
                return _extract_hourly_weather(response_data)
        else:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ",".join(WEATHER_VARIABLES),
                "timezone": "auto",
            }
            response_data = get_with_cache(
                source_name="open_meteo_current",
                url=FORECAST_API_URL,
                params=params,
                ttl_hours=3,
            )
            if isinstance(response_data, dict):
                return _extract_current_weather(response_data)

    return {
        "temperature_2m": None,
        "relative_humidity_2m": None,
        "precipitation": None,
        "rain": None,
        "wind_speed_10m": None,
        "weather_context_available": False,
        "weather_note": (
            "No fue posible obtener contexto meteorologico. "
            "La app continua funcionando sin este enriquecimiento."
        ),
    }


def classify_weather_context(row: pd.Series | dict[str, object]) -> str:
    """Clasifica el clima como contexto y no como prueba causal."""
    get_value = row.get if isinstance(row, dict) else row.__getitem__

    precipitation = _safe_float(get_value("precipitation"))
    rain = _safe_float(get_value("rain"))
    temperature = _safe_float(get_value("temperature_2m"))
    wind_speed = _safe_float(get_value("wind_speed_10m"))

    if (precipitation or 0) > 0 or (rain or 0) > 0:
        return "lluvia_contextual"
    if temperature is not None and temperature >= 30:
        return "calor_contextual"
    if wind_speed is not None and wind_speed >= 25:
        return "viento_contextual"
    return "sin_contexto_climatico_relevante"


def enrich_weather_context(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    max_points: int = 25,
) -> pd.DataFrame:
    """Enriquece un conjunto limitado de zonas/coordenadas con clima contextual."""
    if not has_weather_requirements(schema_mapping):
        empty_df = pd.DataFrame()
        empty_df.attrs["warning"] = "No hay latitud y longitud mapeadas para consultar clima."
        return empty_df

    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    zone_col = schema_mapping.get("zone_col")
    date_col = schema_mapping.get("date_col")

    context_df = pd.DataFrame(
        {
            "zona": dataframe[zone_col].astype(str) if zone_col else "Zona no identificada",
            "latitud": pd.to_numeric(dataframe[latitude_col], errors="coerce"),
            "longitud": pd.to_numeric(dataframe[longitude_col], errors="coerce"),
        }
    )

    if date_col:
        parsed_dates = pd.to_datetime(dataframe[date_col], errors="coerce", dayfirst=True)
        context_df["fecha_referencia"] = parsed_dates.dt.date
    else:
        context_df["fecha_referencia"] = None

    unique_points_df = (
        context_df.dropna(subset=["latitud", "longitud"])
        .drop_duplicates(subset=["zona", "latitud", "longitud", "fecha_referencia"])
        .reset_index(drop=True)
    )

    warning_message = None
    if len(unique_points_df) > max_points:
        warning_message = (
            f"Se limitaron las consultas climaticas a {max_points} puntos unicos para controlar costo y latencia."
        )
        unique_points_df = unique_points_df.head(max_points)

    rows = []
    for _, row in unique_points_df.iterrows():
        weather_data = get_weather_for_point(
            lat=float(row["latitud"]),
            lon=float(row["longitud"]),
            date_value=row["fecha_referencia"],
        )
        rows.append(
            {
                "zona": row["zona"],
                "latitud": float(row["latitud"]),
                "longitud": float(row["longitud"]),
                "fecha_referencia": row["fecha_referencia"],
                **weather_data,
            }
        )

    weather_context_df = pd.DataFrame(rows)
    if not weather_context_df.empty:
        weather_context_df["weather_classification"] = weather_context_df.apply(
            classify_weather_context,
            axis=1,
        )

    if warning_message:
        weather_context_df.attrs["warning"] = warning_message

    return weather_context_df
