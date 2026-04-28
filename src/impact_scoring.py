from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.schema_mapper import SchemaMapping
from src.utils import normalize_text


IMPACT_SCORE_COLUMNS = [
    "zona",
    "territorio",
    "ap_name",
    "status",
    "final_impact_score",
    "classification",
    "technical_severity_score",
    "demand_score",
    "social_criticality_score",
    "weather_context_score",
    "data_confidence_score",
    "ap_health_score",
    "explanation_short",
    "evidence_fields",
    "limitations",
]


def _safe_zone_series(dataframe: pd.DataFrame, zone_col: str) -> pd.Series:
    """Normaliza la columna de zona para agregaciones."""
    zone_series = dataframe[zone_col].copy().astype(object)
    zone_series = zone_series.where(pd.notna(zone_series), "Zona no identificada")
    return zone_series.astype(str)


def _safe_numeric_series(dataframe: pd.DataFrame, column_name: str | None) -> pd.Series | None:
    """Convierte una columna a numerico devolviendo None si no existe."""
    if not column_name:
        return None
    return pd.to_numeric(dataframe[column_name], errors="coerce")


def _normalize_series_to_100(series: pd.Series) -> pd.Series:
    """Normaliza una serie positiva a escala 0-100."""
    clean_series = series.dropna()
    if clean_series.empty:
        return pd.Series(index=series.index, dtype="float64")

    max_value = float(clean_series.max())
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)

    return (series / max_value * 100).clip(lower=0, upper=100)


def _status_signal_by_zone(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
) -> pd.Series:
    """Resume la severidad textual de estado por zona."""
    zone_col = schema_mapping.get("zone_col")
    status_col = schema_mapping.get("status_col")
    if not zone_col or not status_col:
        return pd.Series(dtype="float64")

    status_df = pd.DataFrame(
        {
            "zona": _safe_zone_series(dataframe, zone_col),
            "estado": dataframe[status_col].fillna("").astype(str).map(normalize_text),
        }
    )

    def status_to_score(value: str) -> float:
        if any(keyword in value for keyword in ["critico", "critical", "offline", "down"]):
            return 90.0
        if any(keyword in value for keyword in ["falla", "caido", "caida", "inactivo", "error"]):
            return 70.0
        return 0.0

    status_df["status_score"] = status_df["estado"].map(status_to_score)
    return status_df.groupby("zona", dropna=False)["status_score"].max()


def _technical_scores_from_work_orders(work_orders: pd.DataFrame) -> pd.Series:
    """Resume severidad tecnica preliminar por zona a partir de ordenes."""
    if work_orders is None or work_orders.empty:
        return pd.Series(dtype="float64")

    priority_weights = {"Alta": 90.0, "Media": 65.0, "Observacion": 35.0}
    confidence_bonus = {"Alto": 10.0, "Medio": 5.0, "Bajo": 0.0}

    temp_df = work_orders.copy()
    temp_df["priority_weight"] = temp_df["prioridad"].map(priority_weights).fillna(20.0)
    temp_df["confidence_bonus"] = temp_df["nivel_confianza"].map(confidence_bonus).fillna(0.0)
    aggregated = temp_df.groupby("zona", dropna=False).agg(
        avg_priority=("priority_weight", "mean"),
        max_priority=("priority_weight", "max"),
        avg_bonus=("confidence_bonus", "mean"),
        order_count=("id", "count"),
    )
    aggregated["technical_score"] = (
        aggregated["avg_priority"] * 0.45
        + aggregated["max_priority"] * 0.35
        + aggregated["avg_bonus"] * 1.0
        + aggregated["order_count"].clip(upper=4) * 5
    ).clip(upper=100)
    return aggregated["technical_score"]


def _metric_deficit_score(
    zone_means: pd.Series,
    overall_mean: float,
    multiplier: float,
) -> pd.Series:
    """Transforma bajo rendimiento relativo en puntaje de severidad."""
    if overall_mean <= 0:
        return pd.Series(0.0, index=zone_means.index)

    deficit_ratio = (overall_mean - zone_means) / overall_mean
    deficit_ratio = deficit_ratio.clip(lower=0)
    return (deficit_ratio * multiplier).clip(upper=100)


def _classify_score(score: float) -> str:
    """Clasifica el impacto final en categorias legibles."""
    if score >= 80:
        return "Critico"
    if score >= 60:
        return "Alto"
    if score >= 40:
        return "Medio"
    if score >= 20:
        return "Bajo"
    return "Observacion"


def _calculate_meraki_impact_scores(operational_mart: pd.DataFrame) -> pd.DataFrame:
    """Convierte el mart operativo Meraki en la misma estructura de impact scores."""
    if operational_mart.empty:
        return pd.DataFrame(columns=IMPACT_SCORE_COLUMNS)

    mart_df = operational_mart.copy()
    mart_df["ap_name"] = mart_df.get("ap_name", "").astype(str)
    mart_df["zona"] = mart_df["ap_name"]
    mart_df["territorio"] = mart_df.get("zone_name", mart_df["ap_name"]).astype(str)
    mart_df["status"] = mart_df.get("status", "").astype(str)

    mart_df["final_impact_score"] = pd.to_numeric(
        mart_df.get("operational_risk_score"),
        errors="coerce",
    ).fillna(0.0)
    mart_df["classification"] = mart_df.get("risk_classification", mart_df["final_impact_score"].map(_classify_score))
    mart_df["technical_severity_score"] = (
        pd.to_numeric(mart_df.get("status_risk"), errors="coerce").fillna(0.0)
        + pd.to_numeric(mart_df.get("disconnection_risk"), errors="coerce").fillna(0.0)
    ).clip(upper=100)
    mart_df["demand_score"] = pd.to_numeric(mart_df.get("demand_impact"), errors="coerce").fillna(0.0).clip(upper=100)
    mart_df["social_criticality_score"] = 0.0
    mart_df["weather_context_score"] = 0.0
    mart_df["data_confidence_score"] = pd.to_numeric(mart_df.get("evidence_level"), errors="coerce").fillna(0.0)
    mart_df["ap_health_score"] = pd.to_numeric(mart_df.get("ap_health_score"), errors="coerce").fillna(0.0)
    mart_df["explanation_short"] = mart_df.apply(
        lambda row: (
            f"Riesgo Meraki {float(row.get('final_impact_score', 0)):.1f} basado en estado {row.get('status', 'unknown')}, "
            f"desconexiones {float(row.get('max_disconnection_rate', 0) or 0):.2f} y demanda {float(row.get('demand_score', 0)):.1f}."
        ),
        axis=1,
    )
    mart_df["evidence_fields"] = (
        "status, total_connections, total_disconnections, disconnection_rate, unique_clients, usage_mb_total, connectivity_history"
    )
    mart_df["limitations"] = mart_df.get(
        "limitations",
        "Sin coordenadas exactas del AP. connectivity_history interpretado como texto exportado.",
    )
    mart_df = mart_df.sort_values(
        by=["final_impact_score", "technical_severity_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return mart_df[IMPACT_SCORE_COLUMNS]


def calculate_impact_scores(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    work_orders: pd.DataFrame | None = None,
    osm_context: pd.DataFrame | None = None,
    weather_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula un indice auditable de impacto ciudadano y operativo por zona."""
    if {
        "ap_name",
        "operational_risk_score",
        "demand_impact",
        "ap_health_score",
    }.issubset(set(dataframe.columns)):
        return _calculate_meraki_impact_scores(dataframe)

    zone_col = schema_mapping.get("zone_col")
    if not zone_col or dataframe.empty:
        return pd.DataFrame(columns=IMPACT_SCORE_COLUMNS)

    base_df = pd.DataFrame({"zona": _safe_zone_series(dataframe, zone_col)})
    territory_col = schema_mapping.get("territory_col")
    if territory_col:
        base_df["territorio"] = dataframe[territory_col].astype(object).where(
            pd.notna(dataframe[territory_col]),
            None,
        )
    else:
        base_df["territorio"] = None

    zone_summary = base_df.groupby("zona", dropna=False).agg(
        territorio=("territorio", lambda values: next((value for value in values if value), None)),
        registros=("zona", "count"),
    )

    technical_components = []

    work_order_scores = _technical_scores_from_work_orders(work_orders)
    if not work_order_scores.empty:
        technical_components.append(work_order_scores)

    status_signal = _status_signal_by_zone(dataframe, schema_mapping)
    if not status_signal.empty:
        technical_components.append(status_signal)

    connections_col = schema_mapping.get("connections_col")
    traffic_col = schema_mapping.get("traffic_col")
    date_col = schema_mapping.get("date_col")

    evidence_fields: list[str] = [zone_col]
    limitations_by_zone: dict[str, list[str]] = {zona: [] for zona in zone_summary.index.tolist()}

    connections_series = _safe_numeric_series(dataframe, connections_col)
    if connections_col and connections_series is not None:
        evidence_fields.append(connections_col)
        connections_df = pd.DataFrame(
            {
                "zona": _safe_zone_series(dataframe, zone_col),
                "connections_value": connections_series,
            }
        ).dropna(subset=["connections_value"])
        if not connections_df.empty:
            zone_connections = connections_df.groupby("zona", dropna=False)["connections_value"].mean()
            overall_connections_mean = float(connections_df["connections_value"].mean())
            technical_components.append(
                _metric_deficit_score(zone_connections, overall_connections_mean, multiplier=70)
            )
            demand_connections = _normalize_series_to_100(zone_connections)
        else:
            demand_connections = pd.Series(dtype="float64")
    else:
        demand_connections = pd.Series(dtype="float64")
        for zona in limitations_by_zone:
            limitations_by_zone[zona].append("No hay conexiones mapeadas.")

    traffic_series = _safe_numeric_series(dataframe, traffic_col)
    if traffic_col and traffic_series is not None:
        evidence_fields.append(traffic_col)
        traffic_df = pd.DataFrame(
            {
                "zona": _safe_zone_series(dataframe, zone_col),
                "traffic_value": traffic_series,
            }
        ).dropna(subset=["traffic_value"])
        if not traffic_df.empty:
            zone_traffic = traffic_df.groupby("zona", dropna=False)["traffic_value"].mean()
            overall_traffic_mean = float(traffic_df["traffic_value"].mean())
            technical_components.append(
                _metric_deficit_score(zone_traffic, overall_traffic_mean, multiplier=60)
            )
            demand_traffic = _normalize_series_to_100(zone_traffic)
        else:
            demand_traffic = pd.Series(dtype="float64")
    else:
        demand_traffic = pd.Series(dtype="float64")
        for zona in limitations_by_zone:
            limitations_by_zone[zona].append("No hay trafico mapeado.")

    technical_score = pd.concat(technical_components, axis=1).mean(axis=1) if technical_components else pd.Series(dtype="float64")

    demand_components = []
    if not demand_connections.empty:
        demand_components.append(demand_connections.rename("connections_demand"))
    if not demand_traffic.empty:
        demand_components.append(demand_traffic.rename("traffic_demand"))
    demand_score = pd.concat(demand_components, axis=1).mean(axis=1) if demand_components else pd.Series(dtype="float64")

    social_score = pd.Series(dtype="float64")
    if osm_context is not None and not osm_context.empty and "social_criticality_score" in osm_context.columns:
        evidence_fields.append("osm_context")
        social_score = (
            osm_context.groupby("zona", dropna=False)["social_criticality_score"]
            .mean()
            .astype(float)
        )
    else:
        for zona in limitations_by_zone:
            limitations_by_zone[zona].append("No hay contexto OSM disponible.")

    weather_score = pd.Series(dtype="float64")
    if weather_context is not None and not weather_context.empty:
        evidence_fields.append("weather_context")
        weather_mapping = {
            "lluvia_contextual": 70.0,
            "viento_contextual": 60.0,
            "calor_contextual": 50.0,
            "sin_contexto_climatico_relevante": 10.0,
        }
        temp_df = weather_context.copy()
        if "weather_classification" in temp_df.columns:
            temp_df["weather_score"] = temp_df["weather_classification"].map(weather_mapping).fillna(10.0)
            weather_score = temp_df.groupby("zona", dropna=False)["weather_score"].mean()
    else:
        for zona in limitations_by_zone:
            limitations_by_zone[zona].append("No hay contexto climatico disponible.")

    data_confidence_rows = []
    for zona, row in zone_summary.iterrows():
        zone_mask = _safe_zone_series(dataframe, zone_col) == zona
        zone_df = dataframe.loc[zone_mask]

        score = 0.0
        if zone_col:
            score += 15
        if date_col:
            score += 15
            evidence_fields.append(date_col)
        if connections_col or traffic_col:
            score += 20
        if schema_mapping.get("status_col"):
            score += 10
            evidence_fields.append(schema_mapping["status_col"])
        if schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col"):
            score += 20
            evidence_fields.extend([schema_mapping["latitude_col"], schema_mapping["longitude_col"]])
        elif territory_col:
            score += 15
            evidence_fields.append(territory_col)
        if len(zone_df) >= 3:
            score += 20
        else:
            limitations_by_zone[zona].append("Historico limitado para la zona.")

        if date_col and pd.to_datetime(zone_df[date_col], errors="coerce", dayfirst=True).notna().sum() < 2:
            limitations_by_zone[zona].append("Fecha insuficiente para analizar persistencia.")

        data_confidence_rows.append((zona, min(score, 100)))

    data_confidence_score = pd.Series(
        data={zona: score for zona, score in data_confidence_rows},
        dtype="float64",
    )

    result_df = zone_summary.copy()
    result_df["technical_severity_score"] = technical_score
    result_df["demand_score"] = demand_score
    result_df["social_criticality_score"] = social_score
    result_df["weather_context_score"] = weather_score
    result_df["data_confidence_score"] = data_confidence_score

    base_weights = {
        "technical_severity_score": 0.35,
        "demand_score": 0.25,
        "social_criticality_score": 0.25,
        "data_confidence_score": 0.10,
        "weather_context_score": 0.05,
    }

    final_scores = []
    classifications = []
    explanations = []
    evidence_fields_column = []
    limitations_column = []

    for zona, row in result_df.iterrows():
        active_components = []
        weighted_sum = 0.0
        weight_sum = 0.0

        for component_name, weight in base_weights.items():
            component_value = row.get(component_name)
            if pd.notna(component_value):
                weighted_sum += float(component_value) * weight
                weight_sum += weight
                active_components.append((component_name, float(component_value)))

        final_score = round(weighted_sum / weight_sum, 2) if weight_sum > 0 else 0.0
        final_scores.append(final_score)
        classifications.append(_classify_score(final_score))

        sorted_components = sorted(active_components, key=lambda item: item[1], reverse=True)
        top_drivers = ", ".join(
            f"{name.replace('_score', '')}: {value:.1f}"
            for name, value in sorted_components[:3]
        )
        explanations.append(
            f"Impacto explicado por {top_drivers}."
            if top_drivers
            else "No hubo suficiente evidencia para calcular drivers fuertes."
        )

        zone_evidence_fields = sorted(set(field for field in evidence_fields if field))
        evidence_fields_column.append(", ".join(zone_evidence_fields))

        zone_limitations = sorted(set(limitations_by_zone.get(zona, [])))
        limitations_column.append("; ".join(zone_limitations) if zone_limitations else "Sin limitaciones relevantes.")

    result_df["final_impact_score"] = final_scores
    result_df["classification"] = classifications
    result_df["explanation_short"] = explanations
    result_df["evidence_fields"] = evidence_fields_column
    result_df["limitations"] = limitations_column

    result_df = result_df.reset_index().rename(columns={"index": "zona"})
    result_df = result_df.sort_values(
        by=["final_impact_score", "technical_severity_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result_df[IMPACT_SCORE_COLUMNS]
