from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.external_sources import get_default_headers, post_with_cache, safe_post_json


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
FALLBACK_CALI_CENTER = {"lat": 3.4516, "lon": -76.5320}
FALLBACK_CALI_BOUNDARY = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Cali aproximado",
                "note": "Límite aproximado de referencia; no sustituye cartografía oficial.",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-76.6300, 3.3600],
                        [-76.4250, 3.3600],
                        [-76.4250, 3.5600],
                        [-76.6300, 3.5600],
                        [-76.6300, 3.3600],
                    ]
                ],
            },
        }
    ],
}

MAP_CLASSIFICATION_COLORS = {
    "Critico": "#ef4444",
    "Alto": "#f97316",
    "Medio": "#facc15",
    "Bajo": "#10b981",
    "Observacion": "#22d3ee",
    "Desconocido": "#cbd5e1",
}


def _series_or_default(dataframe: pd.DataFrame, column_name: str, default_value: object = "") -> pd.Series:
    """Devuelve una serie existente o una serie por defecto del mismo largo."""
    if column_name in dataframe.columns:
        return dataframe[column_name]
    return pd.Series([default_value] * len(dataframe), index=dataframe.index)


def get_zone_initials(zone_name: object) -> str:
    """Devuelve iniciales cortas de la zona para el mapa."""
    clean_name = str(zone_name or "").strip()
    if not clean_name:
        return "?"

    words = [word for word in clean_name.replace("-", " ").split() if word]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(word[0] for word in words[:2]).upper()


def get_criticality_emoji(classification: object) -> str:
    """Mapea la clasificación de criticidad a un emoji compacto."""
    normalized = str(classification or "").strip().lower()
    mapping = {
        "critico": "🔴",
        "crítico": "🔴",
        "critical": "🔴",
        "alto": "🟠",
        "high": "🟠",
        "medio": "🟡",
        "medium": "🟡",
        "bajo": "🟢",
        "low": "🟢",
        "observacion": "🔵",
        "observación": "🔵",
        "observation": "🔵",
    }
    return mapping.get(normalized, "⚪")


def get_marker_label(zone_name: object, classification: object, mode: str = "emoji_initials") -> str:
    """Devuelve una etiqueta compacta para mostrar encima del punto."""
    initials = get_zone_initials(zone_name)
    emoji = get_criticality_emoji(classification)

    if mode == "emoji":
        return emoji
    if mode == "initials":
        return initials
    return f"{emoji} {initials}"


def _normalize_classification(value: object) -> str:
    """Unifica etiquetas de criticidad para color y hover."""
    normalized = str(value or "").strip().lower()
    if normalized in {"critico", "crítico", "critical"}:
        return "Critico"
    if normalized in {"alto", "high"}:
        return "Alto"
    if normalized in {"medio", "medium"}:
        return "Medio"
    if normalized in {"bajo", "low"}:
        return "Bajo"
    if normalized in {"observacion", "observación", "observation"}:
        return "Observacion"
    return "Desconocido"


def _estimate_map_zoom(lat_range: float, lon_range: float) -> float:
    """Estima un zoom razonable según dispersión geográfica observada."""
    max_range = max(lat_range, lon_range)
    if max_range <= 0.03:
        return 13
    if max_range <= 0.07:
        return 12
    if max_range <= 0.16:
        return 11
    return 10


def _build_boundary_query() -> str:
    """Construye una consulta Overpass para el límite administrativo de Cali."""
    return """
    [out:json][timeout:25];
    (
      relation["boundary"="administrative"]["name"~"Santiago de Cali|Cali", i]["admin_level"~"6|7|8"];
    );
    out geom;
    """


def _overpass_boundary_to_geojson(payload: dict[str, Any] | list[Any] | None) -> dict[str, Any] | None:
    """Convierte una respuesta Overpass en un GeoJSON mínimo utilizable."""
    if not isinstance(payload, dict):
        return None

    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        return None

    relation = next((element for element in elements if isinstance(element, dict) and element.get("type") == "relation"), None)
    if relation is None:
        return None

    rings: list[list[list[float]]] = []
    for member in relation.get("members", []):
        if not isinstance(member, dict):
            continue
        if member.get("role") not in {"outer", ""}:
            continue
        geometry = member.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 3:
            continue

        coords = []
        for point in geometry:
            if not isinstance(point, dict):
                continue
            lat = point.get("lat")
            lon = point.get("lon")
            if lat is None or lon is None:
                continue
            coords.append([float(lon), float(lat)])

        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        rings.append(coords)

    if not rings:
        return None

    if len(rings) == 1:
        geometry = {"type": "Polygon", "coordinates": [rings[0]]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": relation.get("tags", {}).get("name", "Cali"),
                    "note": "Límite administrativo de referencia obtenido vía Overpass.",
                },
                "geometry": geometry,
            }
        ],
    }


def get_cali_boundary_geojson(force_refresh: bool = False) -> dict[str, Any]:
    """Obtiene el límite de Cali con caché y fallback aproximado."""
    request_payload = {"data": _build_boundary_query()}
    if force_refresh:
        raw_payload = safe_post_json(
            OVERPASS_URL,
            data=request_payload,
            headers=get_default_headers(),
            timeout=25,
        )
    else:
        raw_payload = post_with_cache(
            "cali_boundary_overpass",
            OVERPASS_URL,
            data=request_payload,
            headers=get_default_headers(),
            ttl_hours=720,
        )

    geojson = _overpass_boundary_to_geojson(raw_payload)
    if geojson:
        return geojson
    return FALLBACK_CALI_BOUNDARY


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    if isinstance(data, dict):
        try:
            return pd.DataFrame([data])
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def create_cali_priority_map_pro(
    df: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    impact_scores_df: pd.DataFrame | None = None,
    work_orders_df: pd.DataFrame | None = None,
    recommendations_df: pd.DataFrame | None = None,
    height: int = 760,
) -> go.Figure | None:
    """Construye un mapa ejecutivo grande de Cali sin depender de token."""
    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    zone_col = schema_mapping.get("zone_col")

    if not latitude_col or not longitude_col or latitude_col not in df.columns or longitude_col not in df.columns:
        return None

    base_df = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[latitude_col], errors="coerce"),
            "lon": pd.to_numeric(df[longitude_col], errors="coerce"),
            "zona": df[zone_col].astype(str) if zone_col and zone_col in df.columns else None,
        }
    ).dropna(subset=["lat", "lon"])
    if base_df.empty:
        return None

    if "zona" in base_df.columns and base_df["zona"].notna().any():
        base_df = base_df.groupby("zona", dropna=False)[["lat", "lon"]].mean().reset_index()
    else:
        base_df["zona"] = [f"registro_{index + 1}" for index in range(len(base_df))]

    impact_df = _safe_dataframe(impact_scores_df)
    if not impact_df.empty and "zona" in impact_df.columns:
        impact_merge_columns = [
            column_name
            for column_name in [
                "zona",
                "final_impact_score",
                "classification",
                "technical_severity_score",
                "demand_score",
                "social_criticality_score",
                "action_recommended",
                "accion_recomendada",
            ]
            if column_name in impact_df.columns
        ]
        base_df = base_df.merge(
            impact_df[impact_merge_columns].drop_duplicates(subset=["zona"]),
            on="zona",
            how="left",
        )

    orders_df = _safe_dataframe(work_orders_df)
    if not orders_df.empty and "zona" in orders_df.columns:
        orders_group = (
            orders_df.groupby("zona", dropna=False)
            .agg(
                order_count=("zona", "size"),
                prioridad=("prioridad", "first"),
                accion_recomendada=("accion_recomendada", "first"),
            )
            .reset_index()
        )
        base_df = base_df.merge(orders_group, on="zona", how="left")

    recommendations = _safe_dataframe(recommendations_df)
    if not recommendations.empty:
        rec_zone_col = next(
            (candidate for candidate in ["zona_o_territorio", "zona", "territorio"] if candidate in recommendations.columns),
            None,
        )
        if rec_zone_col:
            rec_group = (
                recommendations.groupby(rec_zone_col, dropna=False)
                .agg(
                    tipo_recomendacion=("tipo_recomendacion", "first"),
                    justificacion=("justificacion", "first"),
                )
                .reset_index()
                .rename(columns={rec_zone_col: "zona"})
            )
            base_df = base_df.merge(rec_group, on="zona", how="left")

    base_df["classification"] = _series_or_default(base_df, "classification", "Observacion").map(_normalize_classification)
    base_df["final_impact_score"] = pd.to_numeric(_series_or_default(base_df, "final_impact_score", 0), errors="coerce")
    base_df["order_count"] = pd.to_numeric(_series_or_default(base_df, "order_count", 0), errors="coerce").fillna(0)
    base_df["marker_label"] = base_df.apply(
        lambda row: get_marker_label(row.get("zona"), row.get("classification"), mode="emoji_initials"),
        axis=1,
    )
    base_df["priority_display"] = _series_or_default(base_df, "prioridad", "Sin prioridad").fillna("Sin prioridad")
    base_df["action_display"] = _series_or_default(base_df, "accion_recomendada", "").fillna("")
    base_df["recommendation_display"] = _series_or_default(base_df, "tipo_recomendacion", "").fillna("")

    if base_df["final_impact_score"].notna().any() and float(base_df["final_impact_score"].fillna(0).max()) > 0:
        base_df["bubble_size"] = base_df["final_impact_score"].fillna(24).clip(lower=24)
    elif base_df["order_count"].notna().any() and float(base_df["order_count"].fillna(0).max()) > 0:
        base_df["bubble_size"] = (base_df["order_count"].fillna(1) * 8).clip(lower=22)
    else:
        base_df["bubble_size"] = 24

    center_lat = float(base_df["lat"].mean()) if base_df["lat"].notna().any() else FALLBACK_CALI_CENTER["lat"]
    center_lon = float(base_df["lon"].mean()) if base_df["lon"].notna().any() else FALLBACK_CALI_CENTER["lon"]
    lat_range = float(base_df["lat"].max() - base_df["lat"].min()) if len(base_df) > 1 else 0.03
    lon_range = float(base_df["lon"].max() - base_df["lon"].min()) if len(base_df) > 1 else 0.03
    zoom = _estimate_map_zoom(lat_range, lon_range)

    fig = px.scatter_mapbox(
        base_df,
        lat="lat",
        lon="lon",
        color="classification",
        size="bubble_size",
        size_max=34,
        text="marker_label",
        hover_name="zona",
        hover_data={
            "classification": True,
            "final_impact_score": ":.2f",
            "priority_display": True,
            "order_count": True,
            "action_display": True,
            "recommendation_display": True,
            "lat": ":.5f",
            "lon": ":.5f",
        },
        color_discrete_map=MAP_CLASSIFICATION_COLORS,
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom,
        height=max(int(height), 720),
        mapbox_style="open-street-map",
    )
    fig.update_traces(
        mode="markers",
        marker={
            "opacity": 0.9,
            "sizemin": 22,
            "allowoverlap": True,
        },
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=base_df["lat"],
            lon=base_df["lon"],
            mode="text",
            text=base_df["marker_label"],
            textposition="middle center",
            textfont={
                "size": 13,
                "color": "#000000",
                "family": "Arial Black, Segoe UI Bold, Segoe UI, sans-serif",
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    boundary_geojson = get_cali_boundary_geojson(force_refresh=False)
    boundary_layers = []
    if boundary_geojson:
        boundary_layers.append(
            {
                "sourcetype": "geojson",
                "source": boundary_geojson,
                "type": "line",
                "color": "rgba(34, 211, 238, 0.75)",
                "line": {"width": 2},
            }
        )

    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": zoom,
            "layers": boundary_layers,
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor="rgba(7, 12, 22, 0)",
        font={"color": "#e2e8f0"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig
