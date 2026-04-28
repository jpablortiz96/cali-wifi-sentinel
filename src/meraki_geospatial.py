from __future__ import annotations

from typing import Any

import pandas as pd

from src.external_sources import get_with_cache


def has_exact_coordinates(package: dict[str, object]) -> bool:
    """Indica si el paquete tiene coordenadas exactas disponibles."""
    for table_key in ["access_points", "hourly_metrics", "events", "clients"]:
        table_df = package.get(table_key)
        if not isinstance(table_df, pd.DataFrame) or table_df.empty:
            continue
        lowered = {str(column).lower() for column in table_df.columns}
        if {"lat", "lon"}.issubset(lowered) or {"latitude", "longitude"}.issubset(lowered):
            return True
    return False


def build_approx_zone_geocoding(
    package: dict[str, object],
    allow_external_geocoding: bool = False,
) -> pd.DataFrame:
    """Geocodifica zonas sólo si el usuario lo habilita explícitamente."""
    access_points_df = package.get("access_points", pd.DataFrame())
    if not isinstance(access_points_df, pd.DataFrame) or access_points_df.empty:
        return pd.DataFrame()

    if not allow_external_geocoding:
        geocoding_df = pd.DataFrame()
        geocoding_df.attrs["warning"] = (
            "La geocodificación aproximada por zona está desactivada. No se generaron coordenadas."
        )
        return geocoding_df

    rows: list[dict[str, object]] = []
    zones = (
        access_points_df["ap_name"].astype(str).dropna().map(lambda value: value.split("_", maxsplit=1)[-1])
        .drop_duplicates()
        .tolist()
    )
    for zone_name in zones:
        query = f"{zone_name} Cali, Colombia"
        response = get_with_cache(
            source_name="nominatim_zone_geocoding",
            url="https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
            },
            ttl_hours=24 * 30,
        )
        if not isinstance(response, list) or not response:
            continue
        first = response[0]
        rows.append(
            {
                "zone_name": zone_name,
                "latitud": pd.to_numeric(first.get("lat"), errors="coerce"),
                "longitud": pd.to_numeric(first.get("lon"), errors="coerce"),
                "geocoding_type": "approx_zone_centroid",
                "confidence": "baja",
                "source": "Nominatim/OSM",
            }
        )

    return pd.DataFrame(rows)


def merge_geocoding_with_operational_mart(
    operational_mart: pd.DataFrame,
    geocoding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une coordenadas aproximadas a la tabla operativa sin afirmarlas como exactas."""
    if operational_mart is None or operational_mart.empty:
        return pd.DataFrame()
    if geocoding_df is None or geocoding_df.empty:
        return operational_mart.copy()

    merged_df = operational_mart.merge(geocoding_df, on="zone_name", how="left")
    return merged_df
