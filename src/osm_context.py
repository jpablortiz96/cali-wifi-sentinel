from __future__ import annotations

from typing import Any

import pandas as pd

from src.external_sources import post_with_cache


OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"


def has_osm_requirements(schema_mapping: dict[str, str | None]) -> bool:
    """Indica si hay coordenadas suficientes para consultar OSM."""
    return bool(schema_mapping.get("latitude_col")) and bool(schema_mapping.get("longitude_col"))


def build_overpass_query(lat: float, lon: float, radius_m: int = 600) -> str:
    """Construye una consulta Overpass con equipamientos relevantes cercanos."""
    return f"""
[out:json][timeout:25];
(
  nwr(around:{radius_m},{lat},{lon})["amenity"="school"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="university"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="college"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="hospital"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="clinic"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="library"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="community_centre"];
  nwr(around:{radius_m},{lat},{lon})["amenity"="townhall"];
  nwr(around:{radius_m},{lat},{lon})["leisure"="park"];
  nwr(around:{radius_m},{lat},{lon})["public_transport"="platform"];
  nwr(around:{radius_m},{lat},{lon})["highway"="bus_stop"];
);
out center tags;
""".strip()


def _extract_tag_counts(elements: list[dict[str, Any]]) -> dict[str, int]:
    """Agrupa elementos OSM por categorias de criticidad social."""
    counts = {
        "poi_total": 0,
        "education_count": 0,
        "health_count": 0,
        "transport_count": 0,
        "parks_count": 0,
        "civic_count": 0,
        "community_count": 0,
    }

    for element in elements:
        tags = element.get("tags", {}) or {}
        amenity = tags.get("amenity")
        leisure = tags.get("leisure")
        public_transport = tags.get("public_transport")
        highway = tags.get("highway")

        counts["poi_total"] += 1

        if amenity in {"school", "university", "college"}:
            counts["education_count"] += 1
        elif amenity in {"hospital", "clinic"}:
            counts["health_count"] += 1
        elif amenity == "community_centre":
            counts["community_count"] += 1
        elif amenity in {"townhall", "library"}:
            counts["civic_count"] += 1

        if leisure == "park":
            counts["parks_count"] += 1

        if public_transport == "platform" or highway == "bus_stop":
            counts["transport_count"] += 1

    return counts


def query_osm_pois(lat: float, lon: float, radius_m: int = 600) -> dict[str, object]:
    """Consulta OSM Overpass con cache y devuelve conteos simplificados."""
    query = build_overpass_query(lat, lon, radius_m=radius_m)
    response_data = post_with_cache(
        source_name="osm_overpass",
        url=OVERPASS_API_URL,
        data={"data": query},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        ttl_hours=24 * 30,
    )

    if not isinstance(response_data, dict):
        return {
            "poi_total": 0,
            "education_count": 0,
            "health_count": 0,
            "transport_count": 0,
            "parks_count": 0,
            "civic_count": 0,
            "community_count": 0,
            "osm_context_available": False,
        }

    elements = response_data.get("elements", []) or []
    counts = _extract_tag_counts(elements)
    counts["osm_context_available"] = bool(elements)
    return counts


def calculate_social_criticality_score(osm_row: pd.Series | dict[str, object]) -> float:
    """Calcula criticidad territorial aproximada de 0 a 100."""
    get_value = osm_row.get if isinstance(osm_row, dict) else osm_row.__getitem__

    health_count = float(get_value("health_count") or 0)
    education_count = float(get_value("education_count") or 0)
    transport_count = float(get_value("transport_count") or 0)
    parks_count = float(get_value("parks_count") or 0)
    civic_count = float(get_value("civic_count") or 0)
    community_count = float(get_value("community_count") or 0)

    health_score = min(health_count, 2) / 2 * 25
    education_score = min(education_count, 3) / 3 * 25
    transport_score = min(transport_count, 4) / 4 * 20
    civic_score = min(civic_count, 2) / 2 * 10
    community_score = min(community_count, 2) / 2 * 10
    parks_score = min(parks_count, 2) / 2 * 10

    return round(
        health_score
        + education_score
        + transport_score
        + civic_score
        + community_score
        + parks_score,
        2,
    )


def enrich_osm_context(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    max_points: int = 25,
    radius_m: int = 600,
) -> pd.DataFrame:
    """Enriquece coordenadas unicas con contexto urbano basado en OSM."""
    if not has_osm_requirements(schema_mapping):
        empty_df = pd.DataFrame()
        empty_df.attrs["warning"] = "No hay latitud y longitud mapeadas para consultar OSM."
        return empty_df

    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    zone_col = schema_mapping.get("zone_col")

    context_df = pd.DataFrame(
        {
            "zona": dataframe[zone_col].astype(str) if zone_col else "Zona no identificada",
            "latitud": pd.to_numeric(dataframe[latitude_col], errors="coerce"),
            "longitud": pd.to_numeric(dataframe[longitude_col], errors="coerce"),
        }
    )

    unique_points_df = (
        context_df.dropna(subset=["latitud", "longitud"])
        .drop_duplicates(subset=["zona", "latitud", "longitud"])
        .reset_index(drop=True)
    )

    warning_message = None
    if len(unique_points_df) > max_points:
        warning_message = (
            f"Se limitaron las consultas OSM a {max_points} puntos unicos para controlar costo y latencia."
        )
        unique_points_df = unique_points_df.head(max_points)

    rows = []
    for _, row in unique_points_df.iterrows():
        osm_counts = query_osm_pois(
            lat=float(row["latitud"]),
            lon=float(row["longitud"]),
            radius_m=radius_m,
        )
        row_payload = {
            "zona": row["zona"],
            "latitud": float(row["latitud"]),
            "longitud": float(row["longitud"]),
            **osm_counts,
        }
        row_payload["social_criticality_score"] = calculate_social_criticality_score(row_payload)
        rows.append(row_payload)

    osm_context_df = pd.DataFrame(rows)
    if warning_message:
        osm_context_df.attrs["warning"] = warning_message

    return osm_context_df
