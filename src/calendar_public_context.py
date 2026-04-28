from __future__ import annotations

import pandas as pd

from src.external_sources import get_with_cache


def get_colombia_holidays(year: int) -> pd.DataFrame:
    """Consulta festivos de Colombia con caché usando Nager.Date."""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{int(year)}/CO"
    payload = get_with_cache(
        "nager_colombia_holidays",
        url,
        ttl_hours=24 * 30,
    )
    if not payload or not isinstance(payload, list):
        return pd.DataFrame(columns=["date", "localName", "name", "types"])

    holidays_df = pd.DataFrame(payload)
    expected_columns = ["date", "localName", "name", "types"]
    for column in expected_columns:
        if column not in holidays_df.columns:
            holidays_df[column] = None
    holidays_df["date"] = pd.to_datetime(holidays_df["date"], errors="coerce").dt.date
    return holidays_df[expected_columns].copy()


def enrich_hourly_with_public_calendar(hourly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Agrega contexto de fin de semana y festivo a métricas horarias."""
    if hourly_metrics is None or hourly_metrics.empty:
        return pd.DataFrame()

    hourly_df = hourly_metrics.copy()
    timestamp_column = "timestamp_hour" if "timestamp_hour" in hourly_df.columns else None
    if not timestamp_column:
        return pd.DataFrame()

    hourly_df[timestamp_column] = pd.to_datetime(hourly_df[timestamp_column], errors="coerce")
    hourly_df = hourly_df.dropna(subset=[timestamp_column])
    if hourly_df.empty:
        return pd.DataFrame()

    hourly_df["date"] = hourly_df[timestamp_column].dt.date
    hourly_df["day_name"] = hourly_df[timestamp_column].dt.day_name()
    hourly_df["hour"] = hourly_df[timestamp_column].dt.hour
    hourly_df["is_weekend"] = hourly_df[timestamp_column].dt.dayofweek >= 5
    hourly_df["year"] = hourly_df[timestamp_column].dt.year

    holiday_frames = [get_colombia_holidays(year) for year in sorted(hourly_df["year"].dropna().unique())]
    holidays_df = pd.concat(holiday_frames, ignore_index=True) if holiday_frames else pd.DataFrame()
    holiday_lookup = {}
    if not holidays_df.empty:
        holiday_lookup = holidays_df.set_index("date")["localName"].astype(str).to_dict()

    hourly_df["holiday_name"] = hourly_df["date"].map(holiday_lookup)
    hourly_df["is_holiday"] = hourly_df["holiday_name"].notna()
    hourly_df["public_context_label"] = hourly_df.apply(
        lambda row: (
            f"Festivo: {row['holiday_name']}"
            if row["is_holiday"]
            else ("Fin de semana" if row["is_weekend"] else "Dia habil")
        ),
        axis=1,
    )
    return hourly_df.drop(columns=["year"])


def summarize_calendar_impact(hourly_metrics_enriched: pd.DataFrame) -> dict[str, object]:
    """Resume efectos observables por festivos, fines de semana y horas pico."""
    if hourly_metrics_enriched is None or hourly_metrics_enriched.empty:
        return {
            "avg_connections_holidays": 0.0,
            "avg_connections_non_holidays": 0.0,
            "avg_disconnection_holidays": 0.0,
            "top_weekend_aps": [],
            "top_peak_hour_aps": [],
            "limitations": ["No hay métricas horarias enriquecidas para analizar calendario público."],
        }

    hourly_df = hourly_metrics_enriched.copy()
    hourly_df["total_connections"] = pd.to_numeric(hourly_df.get("total_connections"), errors="coerce").fillna(0)
    hourly_df["disconnection_rate"] = pd.to_numeric(hourly_df.get("disconnection_rate"), errors="coerce").fillna(0)

    holiday_mask = hourly_df.get("is_holiday", pd.Series([False] * len(hourly_df), index=hourly_df.index)).astype(bool)
    peak_mask = hourly_df.get("hour", pd.Series([0] * len(hourly_df), index=hourly_df.index)).between(7, 9) | hourly_df.get("hour", pd.Series([0] * len(hourly_df), index=hourly_df.index)).between(18, 20)
    weekend_mask = hourly_df.get("is_weekend", pd.Series([False] * len(hourly_df), index=hourly_df.index)).astype(bool)

    weekend_top = []
    if "ap_name" in hourly_df.columns:
        weekend_top_df = (
            hourly_df[weekend_mask]
            .groupby("ap_name", dropna=False)["total_connections"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
        )
        weekend_top = [{"ap_name": str(ap_name), "avg_connections": float(value)} for ap_name, value in weekend_top_df.items()]

        peak_top_df = (
            hourly_df[peak_mask]
            .groupby("ap_name", dropna=False)["total_connections"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
        )
        peak_top = [{"ap_name": str(ap_name), "avg_connections": float(value)} for ap_name, value in peak_top_df.items()]
    else:
        peak_top = []

    return {
        "avg_connections_holidays": round(float(hourly_df.loc[holiday_mask, "total_connections"].mean()), 2) if holiday_mask.any() else 0.0,
        "avg_connections_non_holidays": round(float(hourly_df.loc[~holiday_mask, "total_connections"].mean()), 2) if (~holiday_mask).any() else 0.0,
        "avg_disconnection_holidays": round(float(hourly_df.loc[holiday_mask, "disconnection_rate"].mean()), 3) if holiday_mask.any() else 0.0,
        "top_weekend_aps": weekend_top,
        "top_peak_hour_aps": peak_top,
        "limitations": [
            "El contexto de festivos se basa en calendario público nacional y no representa eventos hiperlocales."
        ],
    }
