from __future__ import annotations

from typing import Any

import holidays
import pandas as pd

from src.utils import normalize_text


def _find_hour_column(dataframe: pd.DataFrame) -> str | None:
    """Busca una columna con informacion de hora si existe de forma separada."""
    for column in dataframe.columns:
        normalized_name = normalize_text(str(column))
        if any(keyword in normalized_name for keyword in ["hora", "hour", "time"]):
            return str(column)
    return None


def _infer_peak_flag(date_series: pd.Series, dataframe: pd.DataFrame) -> pd.Series:
    """Intenta inferir periodos pico potenciales con hora explicita o embebida."""
    hour_series = pd.Series(index=date_series.index, dtype="float64")

    if date_series.dt.hour.notna().any():
        hour_series = date_series.dt.hour.astype("float64")
    else:
        hour_column = _find_hour_column(dataframe)
        if hour_column:
            hour_series = pd.to_numeric(dataframe[hour_column], errors="coerce")

    return hour_series.isin([6, 7, 8, 17, 18, 19, 20])


def enrich_calendar_features(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
) -> pd.DataFrame:
    """Crea variables temporales locales sin depender de APIs externas."""
    date_col = schema_mapping.get("date_col")
    zone_col = schema_mapping.get("zone_col")

    if not date_col:
        empty_df = pd.DataFrame()
        empty_df.attrs["warning"] = "No hay columna de fecha mapeada para enriquecer calendario."
        return empty_df

    parsed_datetime = pd.to_datetime(dataframe[date_col], errors="coerce", dayfirst=True)
    valid_mask = parsed_datetime.notna()

    if not valid_mask.any():
        empty_df = pd.DataFrame()
        empty_df.attrs["warning"] = "La columna de fecha no pudo convertirse de forma util."
        return empty_df

    valid_datetimes = parsed_datetime.loc[valid_mask]
    years = sorted(valid_datetimes.dt.year.dropna().astype(int).unique().tolist())

    try:
        colombia_holidays = holidays.country_holidays("CO", years=years or None)
    except Exception:  # noqa: BLE001
        colombia_holidays = {}

    calendar_df = pd.DataFrame(
        {
            "zona": dataframe.loc[valid_mask, zone_col].astype(str) if zone_col else "Zona no identificada",
            "fecha": valid_datetimes.dt.date,
            "dia_semana": valid_datetimes.dt.day_name(),
            "es_fin_de_semana": valid_datetimes.dt.weekday >= 5,
            "mes": valid_datetimes.dt.month,
            "anio": valid_datetimes.dt.year,
            "es_festivo_colombia": valid_datetimes.dt.date.map(lambda date_value: date_value in colombia_holidays),
            "periodo_pico_potencial": _infer_peak_flag(parsed_datetime, dataframe).loc[valid_mask].fillna(False),
            "calendar_context_available": True,
        }
    ).reset_index(drop=True)

    return calendar_df
