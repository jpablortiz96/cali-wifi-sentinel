from __future__ import annotations

from typing import Any

import pandas as pd

from src.meraki_features import extract_zone_name


CITIZEN_SCORE_COLUMNS = [
    "ap_name",
    "zone_name",
    "citizen_experience_score",
    "citizen_status",
    "stability_score",
    "availability_score",
    "perceived_capacity_score",
    "citizen_activity_score",
    "data_confidence_score",
    "best_hours",
    "avoid_hours",
    "explanation",
    "limitations",
]


def _empty_scores() -> pd.DataFrame:
    """Devuelve una tabla vacia con el esquema ciudadano esperado."""
    return pd.DataFrame(columns=CITIZEN_SCORE_COLUMNS)


def _safe_numeric_series(dataframe: pd.DataFrame, column_name: str, default_value: float = 0.0) -> pd.Series:
    """Convierte una columna a numerico o devuelve una serie por defecto."""
    if column_name in dataframe.columns:
        return pd.to_numeric(dataframe[column_name], errors="coerce").fillna(default_value)
    return pd.Series([default_value] * len(dataframe), index=dataframe.index, dtype="float64")


def _min_max_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    """Escala una serie a 0-100 de forma robusta."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.empty:
        return pd.Series(dtype="float64")

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value == min_value:
        base = pd.Series([70.0 if max_value > 0 else 0.0] * len(numeric), index=numeric.index)
    else:
        base = ((numeric - min_value) / (max_value - min_value)) * 100.0

    if invert:
        base = 100.0 - base
    return base.clip(0, 100).round(2)


def _status_to_availability(status: object) -> float:
    """Asigna una disponibilidad aproximada desde el estado operativo."""
    normalized = str(status or "").strip().lower()
    if normalized == "online":
        return 95.0
    if normalized == "dormant":
        return 55.0
    if normalized == "offline":
        return 20.0
    if normalized:
        return 60.0
    return 40.0


def _citizen_status_from_score(score: float, evidence_score: float) -> str:
    """Clasifica la experiencia estimada para uso ciudadano."""
    if evidence_score < 35:
        return "Sin evidencia suficiente"
    if score >= 85:
        return "Excelente"
    if score >= 70:
        return "Buena"
    if score >= 55:
        return "Regular"
    return "Inestable"


def _hour_list_from_patterns(
    patterns_df: pd.DataFrame,
    ap_name: str,
    target_column: str,
) -> str:
    """Construye una lista corta de horas recomendadas o a evitar por AP."""
    if patterns_df.empty or "ap_name" not in patterns_df.columns or target_column not in patterns_df.columns:
        return "Sin evidencia horaria"

    ap_rows = patterns_df[patterns_df["ap_name"].astype(str) == str(ap_name)].copy()
    if ap_rows.empty:
        return "Sin evidencia horaria"

    selected = ap_rows[ap_rows[target_column].astype(bool)].sort_values("hour")
    if selected.empty:
        return "Sin evidencia horaria"

    labels = [f"{int(hour):02d}:00" for hour in selected["hour"].dropna().tolist()[:3]]
    return " | ".join(labels) if labels else "Sin evidencia horaria"


def calculate_hourly_citizen_patterns(hourly_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcula patrones horarios agregados para apoyar recomendaciones ciudadanas."""
    if hourly_metrics is None or hourly_metrics.empty or "ap_name" not in hourly_metrics.columns:
        return pd.DataFrame(
            columns=[
                "ap_name",
                "zone_name",
                "hour",
                "day_name",
                "day_of_week",
                "avg_connections",
                "avg_unique_clients",
                "avg_disconnection_rate",
                "best_hour_label",
                "congested_hour_label",
                "unstable_hour_label",
                "is_best_hour",
                "is_congested_hour",
                "is_unstable_hour",
            ]
        )

    hourly_df = hourly_metrics.copy()
    hourly_df["timestamp_hour"] = pd.to_datetime(hourly_df.get("timestamp_hour"), errors="coerce")
    hourly_df = hourly_df.dropna(subset=["timestamp_hour"])
    if hourly_df.empty:
        return pd.DataFrame()

    hourly_df["zone_name"] = hourly_df["ap_name"].astype(str).map(extract_zone_name)
    hourly_df["hour"] = hourly_df["timestamp_hour"].dt.hour
    hourly_df["day_of_week"] = hourly_df["timestamp_hour"].dt.dayofweek
    hourly_df["day_name"] = hourly_df["timestamp_hour"].dt.day_name()
    hourly_df["total_connections"] = pd.to_numeric(hourly_df.get("total_connections"), errors="coerce").fillna(0)
    hourly_df["unique_clients"] = pd.to_numeric(hourly_df.get("unique_clients"), errors="coerce").fillna(0)
    hourly_df["disconnection_rate"] = pd.to_numeric(hourly_df.get("disconnection_rate"), errors="coerce").fillna(0)

    grouped = (
        hourly_df.groupby(["ap_name", "zone_name", "hour", "day_name", "day_of_week"], dropna=False)
        .agg(
            avg_connections=("total_connections", "mean"),
            avg_unique_clients=("unique_clients", "mean"),
            avg_disconnection_rate=("disconnection_rate", "mean"),
            samples=("timestamp_hour", "count"),
        )
        .reset_index()
    )
    grouped["avg_connections"] = grouped["avg_connections"].round(2)
    grouped["avg_unique_clients"] = grouped["avg_unique_clients"].round(2)
    grouped["avg_disconnection_rate"] = grouped["avg_disconnection_rate"].round(3)

    grouped["best_hour_score"] = (
        grouped["avg_connections"] * 1.2
        + grouped["avg_unique_clients"] * 0.6
        - grouped["avg_disconnection_rate"] * 35
    )
    grouped["congestion_score"] = (
        grouped["avg_unique_clients"] * 1.3
        + grouped["avg_connections"] * 0.8
        + grouped["avg_disconnection_rate"] * 25
    )
    grouped["instability_score"] = grouped["avg_disconnection_rate"] * 100 - grouped["avg_connections"] * 0.4

    best_idx = grouped.groupby("ap_name")["best_hour_score"].idxmax()
    congested_idx = grouped.groupby("ap_name")["congestion_score"].idxmax()
    unstable_idx = grouped.groupby("ap_name")["instability_score"].idxmax()

    grouped["is_best_hour"] = grouped.index.isin(best_idx)
    grouped["is_congested_hour"] = grouped.index.isin(congested_idx)
    grouped["is_unstable_hour"] = grouped.index.isin(unstable_idx)

    best_map = grouped.loc[best_idx, ["ap_name", "hour"]].assign(best_hour_label=lambda df: df["hour"].map(lambda x: f"{int(x):02d}:00"))
    congested_map = grouped.loc[congested_idx, ["ap_name", "hour"]].assign(congested_hour_label=lambda df: df["hour"].map(lambda x: f"{int(x):02d}:00"))
    unstable_map = grouped.loc[unstable_idx, ["ap_name", "hour"]].assign(unstable_hour_label=lambda df: df["hour"].map(lambda x: f"{int(x):02d}:00"))

    grouped = grouped.merge(best_map[["ap_name", "best_hour_label"]], on="ap_name", how="left")
    grouped = grouped.merge(congested_map[["ap_name", "congested_hour_label"]], on="ap_name", how="left")
    grouped = grouped.merge(unstable_map[["ap_name", "unstable_hour_label"]], on="ap_name", how="left")

    return grouped


def calculate_citizen_experience_score(
    operational_mart: pd.DataFrame,
    hourly_metrics: pd.DataFrame | None = None,
    clients: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula una aproximación agregada de experiencia ciudadana por AP y zona."""
    if operational_mart is None or operational_mart.empty:
        return _empty_scores()

    mart = operational_mart.copy()
    if "ap_name" not in mart.columns:
        mart["ap_name"] = mart.index.map(lambda value: f"ap_{value}")
    if "zone_name" not in mart.columns:
        mart["zone_name"] = mart["ap_name"].astype(str).map(extract_zone_name)

    mart["avg_disconnection_rate"] = _safe_numeric_series(mart, "avg_disconnection_rate", default_value=0.0)
    mart["usage_mb_total"] = _safe_numeric_series(mart, "usage_mb_total", default_value=0.0)
    mart["clients_reported"] = _safe_numeric_series(mart, "clients_reported", default_value=0.0)
    mart["total_connections"] = _safe_numeric_series(mart, "total_connections", default_value=0.0)
    mart["active_hours"] = _safe_numeric_series(mart, "active_hours", default_value=0.0)
    mart["evidence_level"] = _safe_numeric_series(mart, "evidence_level", default_value=0.0)

    patterns_df = calculate_hourly_citizen_patterns(hourly_metrics if isinstance(hourly_metrics, pd.DataFrame) else pd.DataFrame())

    activity_proxy = (
        _min_max_scale(mart["usage_mb_total"])
        + _min_max_scale(mart["clients_reported"])
        + _min_max_scale(mart["total_connections"])
    ) / 3.0

    disconnect_penalty = _min_max_scale(mart["avg_disconnection_rate"], invert=False)
    stability_score = (100.0 - disconnect_penalty).clip(0, 100).round(2)
    availability_score = mart.get("status", pd.Series([""] * len(mart))).map(_status_to_availability).fillna(40.0)

    demand_pressure = (_min_max_scale(mart["usage_mb_total"]) + _min_max_scale(mart["clients_reported"])) / 2.0
    perceived_capacity_score = (100.0 - ((demand_pressure * disconnect_penalty) / 100.0)).clip(0, 100).round(2)
    citizen_activity_score = activity_proxy.round(2)

    client_confidence_bonus = 0.0
    if isinstance(clients, pd.DataFrame) and not clients.empty:
        client_confidence_bonus = 8.0
    data_confidence_score = (
        (mart["evidence_level"] * 0.55)
        + _min_max_scale(mart["active_hours"]) * 0.25
        + _min_max_scale(mart["total_connections"]) * 0.20
        + client_confidence_bonus
    ).clip(0, 100).round(2)

    mart["stability_score"] = stability_score
    mart["availability_score"] = availability_score.round(2)
    mart["perceived_capacity_score"] = perceived_capacity_score
    mart["citizen_activity_score"] = citizen_activity_score
    mart["data_confidence_score"] = data_confidence_score
    mart["citizen_experience_score"] = (
        mart["stability_score"] * 0.35
        + mart["availability_score"] * 0.25
        + mart["perceived_capacity_score"] * 0.20
        + mart["citizen_activity_score"] * 0.10
        + mart["data_confidence_score"] * 0.10
    ).round(2)
    mart["citizen_status"] = [
        _citizen_status_from_score(score, confidence)
        for score, confidence in zip(mart["citizen_experience_score"], mart["data_confidence_score"], strict=False)
    ]
    mart["best_hours"] = mart["ap_name"].map(lambda ap_name: _hour_list_from_patterns(patterns_df, str(ap_name), "is_best_hour"))
    mart["avoid_hours"] = mart["ap_name"].map(lambda ap_name: _hour_list_from_patterns(patterns_df, str(ap_name), "is_unstable_hour"))
    mart["explanation"] = (
        "Disponibilidad: "
        + mart["availability_score"].round(1).astype(str)
        + " | Estabilidad: "
        + mart["stability_score"].round(1).astype(str)
        + " | Capacidad percibida: "
        + mart["perceived_capacity_score"].round(1).astype(str)
    )
    mart["limitations"] = (
        "Indicador agregado por AP/zona. No representa experiencias individuales ni rastrea clientes. "
        "Se basa en disponibilidad, desconexiones, actividad y volumen de evidencia."
    )

    result_columns = [column for column in CITIZEN_SCORE_COLUMNS if column in mart.columns]
    return mart[result_columns].sort_values(
        by=["citizen_experience_score", "data_confidence_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_citizen_zone_summary(citizen_scores: pd.DataFrame, operational_mart: pd.DataFrame) -> pd.DataFrame:
    """Resume la experiencia ciudadana por zona usando APs agregados."""
    if citizen_scores is None or citizen_scores.empty:
        return pd.DataFrame(
            columns=["zona", "APs", "score_promedio", "mejor_ap", "estado_ciudadano", "uso_estimado", "recomendacion_ciudadana"]
        )

    scores_df = citizen_scores.copy()
    scores_df["zone_name"] = scores_df.get("zone_name", scores_df.get("ap_name", "Sin zona")).astype(str)
    mart = operational_mart.copy() if isinstance(operational_mart, pd.DataFrame) else pd.DataFrame()
    if not mart.empty and "zone_name" not in mart.columns:
        mart["zone_name"] = mart.get("ap_name", pd.Series(index=mart.index, dtype="object")).astype(str).map(extract_zone_name)

    zone_usage = (
        mart.groupby("zone_name", dropna=False)["usage_mb_total"].sum().to_dict()
        if not mart.empty and "usage_mb_total" in mart.columns and "zone_name" in mart.columns
        else {}
    )

    rows: list[dict[str, Any]] = []
    for zone_name, zone_df in scores_df.groupby("zone_name", dropna=False):
        best_row = zone_df.sort_values("citizen_experience_score", ascending=False).iloc[0]
        score_mean = round(float(zone_df["citizen_experience_score"].mean()), 2)
        confidence_mean = float(pd.to_numeric(zone_df["data_confidence_score"], errors="coerce").fillna(0).mean())
        status = _citizen_status_from_score(score_mean, confidence_mean)

        if status in {"Excelente", "Buena"}:
            recommendation = "Zona recomendable para conexión ciudadana en los horarios sugeridos."
        elif status == "Regular":
            recommendation = "Usar con precaución y priorizar los mejores horarios detectados."
        elif status == "Inestable":
            recommendation = "Conviene evitar esta zona en horas inestables hasta validar operación."
        else:
            recommendation = "Se requiere más evidencia antes de emitir una recomendación sólida."

        rows.append(
            {
                "zona": zone_name,
                "APs": int(zone_df["ap_name"].astype(str).nunique()) if "ap_name" in zone_df.columns else len(zone_df),
                "score_promedio": score_mean,
                "mejor_ap": str(best_row.get("ap_name", "N/A")),
                "estado_ciudadano": status,
                "uso_estimado": round(float(zone_usage.get(zone_name, 0.0)), 2),
                "recomendacion_ciudadana": recommendation,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["score_promedio", "APs"], ascending=[False, False]).reset_index(drop=True)
