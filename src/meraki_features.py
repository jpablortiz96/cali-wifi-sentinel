from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def extract_zone_name(ap_name: object) -> str:
    """Extrae un nombre de zona legible desde el nombre del AP."""
    if ap_name is None or pd.isna(ap_name):
        return "Zona no identificada"

    text = str(ap_name).strip()
    if not text:
        return "Zona no identificada"

    cleaned = re.sub(r"^\d+[_\-\s]*", "", text)
    cleaned = re.sub(r"^(ZW|Zona Wifi)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-\s]*AP\d+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-\s]*AP$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


def parse_connectivity_history(connectivity_history: object, ap_name: str | None = None) -> pd.DataFrame:
    """Convierte el texto de conectividad en transiciones simples de estado."""
    if connectivity_history is None or pd.isna(connectivity_history):
        return pd.DataFrame(columns=["timestamp", "state_code", "ap_name"])

    raw_tokens = [token.strip() for token in str(connectivity_history).split(",") if token.strip()]
    if len(raw_tokens) < 3:
        return pd.DataFrame(columns=["timestamp", "state_code", "ap_name"])

    rows: list[dict[str, object]] = []
    for index in range(0, len(raw_tokens) - 2, 3):
        date_token = raw_tokens[index]
        time_token = raw_tokens[index + 1]
        state_token = raw_tokens[index + 2]
        timestamp = pd.to_datetime(f"{date_token} {time_token}", errors="coerce", dayfirst=True)
        state_code = pd.to_numeric(state_token, errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "state_code": state_code if not pd.isna(state_code) else state_token,
                "ap_name": ap_name,
            }
        )

    connectivity_df = pd.DataFrame(rows)
    if not connectivity_df.empty:
        connectivity_df.attrs["limitation"] = (
            "connectivity_history está disponible como texto exportado; los códigos requieren validación técnica con Meraki."
        )
    return connectivity_df


def build_access_point_dimension(package: dict[str, object]) -> pd.DataFrame:
    """Construye una dimensión base por AP con estado y conectividad."""
    access_points_df = package.get("access_points", pd.DataFrame())
    if not isinstance(access_points_df, pd.DataFrame) or access_points_df.empty:
        return pd.DataFrame()

    dimension_df = access_points_df.copy()
    dimension_df["zone_name"] = dimension_df["ap_name"].map(extract_zone_name)
    dimension_df["ap_number"] = (
        dimension_df["ap_name"].astype(str).str.extract(r"AP[_\- ]?(\d+)$", expand=False).fillna("1")
    )

    transition_counts: list[int] = []
    unique_codes: list[int] = []
    last_connectivity_event: list[pd.Timestamp | None] = []

    for _, row in dimension_df.iterrows():
        history_df = parse_connectivity_history(row.get("connectivity_history"), ap_name=str(row.get("ap_name")))
        transition_counts.append(int(len(history_df)))
        unique_codes.append(int(history_df["state_code"].astype(str).nunique()) if not history_df.empty else 0)
        last_connectivity_event.append(history_df["timestamp"].max() if not history_df.empty else pd.NaT)

    dimension_df["connectivity_transition_count"] = transition_counts
    dimension_df["connectivity_unique_codes"] = unique_codes
    dimension_df["last_connectivity_event"] = last_connectivity_event
    return dimension_df


def build_client_ap_summary(package: dict[str, object]) -> pd.DataFrame:
    """Resume clientes y uso por AP."""
    clients_df = package.get("clients", pd.DataFrame())
    if not isinstance(clients_df, pd.DataFrame) or clients_df.empty:
        return pd.DataFrame()

    summary_df = clients_df.copy()
    summary_df["usage_mb"] = pd.to_numeric(summary_df.get("usage_mb"), errors="coerce").fillna(0.0)
    summary_df["status_normalized"] = summary_df.get("status", "").astype(str).str.strip().str.lower()

    grouped = summary_df.groupby("ap_name", dropna=False).agg(
        clients_reported=("client_id", "nunique"),
        usage_mb_total=("usage_mb", "sum"),
        usage_mb_average=("usage_mb", "mean"),
        clients_online=("status_normalized", lambda values: int(values.eq("online").sum())),
        clients_offline=("status_normalized", lambda values: int(values.eq("offline").sum())),
        last_seen_max=("last_seen", "max"),
    ).reset_index()

    top_device_rows = []
    if "device_type" in summary_df.columns:
        for ap_name, ap_slice in summary_df.groupby("ap_name", dropna=False):
            top_devices = ap_slice["device_type"].astype(str).value_counts().head(3).index.tolist()
            top_device_rows.append(
                {
                    "ap_name": ap_name,
                    "top_device_types": ", ".join(top_devices),
                }
            )

    if top_device_rows:
        grouped = grouped.merge(pd.DataFrame(top_device_rows), on="ap_name", how="left")

    grouped["usage_mb_total"] = grouped["usage_mb_total"].round(2)
    grouped["usage_mb_average"] = grouped["usage_mb_average"].round(2)
    return grouped


def build_event_ap_summary(package: dict[str, object]) -> pd.DataFrame:
    """Resume actividad de eventos por AP."""
    events_df = package.get("events", pd.DataFrame())
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return pd.DataFrame()

    summary_df = events_df.copy()
    summary_df["timestamp"] = pd.to_datetime(summary_df.get("timestamp"), errors="coerce")
    summary_df["event_type_norm"] = summary_df.get("event_type", "").astype(str).str.lower()
    summary_df["event_detail_norm"] = summary_df.get("event_detail", "").astype(str)

    grouped = summary_df.groupby("ap_name", dropna=False).agg(
        total_events=("event_type_norm", "count"),
        associations=("event_type_norm", lambda values: int(values.str.contains("association", na=False).sum())),
        disassociations=("event_type_norm", lambda values: int(values.str.contains("disassociation", na=False).sum())),
        authentications=("event_type_norm", lambda values: int(values.str.contains("authentication", na=False).sum())),
        splash_auth=("event_type_norm", lambda values: int(values.str.contains("splash", na=False).sum())),
        last_seen_event=("timestamp", "max"),
    ).reset_index()

    dominant_rows = []
    for ap_name, ap_slice in summary_df.groupby("ap_name", dropna=False):
        top_event_type = ap_slice["event_type"].astype(str).value_counts().head(1)
        top_event_detail = ap_slice["event_detail"].astype(str).value_counts().head(1)
        dominant_rows.append(
            {
                "ap_name": ap_name,
                "dominant_event_type": top_event_type.index[0] if not top_event_type.empty else "",
                "top_event_detail": top_event_detail.index[0] if not top_event_detail.empty else "",
            }
        )

    return grouped.merge(pd.DataFrame(dominant_rows), on="ap_name", how="left")


def build_hourly_ap_features(package: dict[str, object]) -> pd.DataFrame:
    """Agrega métricas horarias por AP."""
    hourly_df = package.get("hourly_metrics", pd.DataFrame())
    if not isinstance(hourly_df, pd.DataFrame) or hourly_df.empty:
        return pd.DataFrame()

    summary_df = hourly_df.copy()
    summary_df["timestamp_hour"] = pd.to_datetime(summary_df.get("timestamp_hour"), errors="coerce")
    numeric_columns = [
        "total_events",
        "total_connections",
        "total_disconnections",
        "total_auth",
        "unique_clients",
        "disconnection_rate",
    ]
    for column_name in numeric_columns:
        summary_df[column_name] = pd.to_numeric(summary_df.get(column_name), errors="coerce").fillna(0.0)

    grouped = summary_df.groupby("ap_name", dropna=False).agg(
        total_events=("total_events", "sum"),
        total_connections=("total_connections", "sum"),
        total_disconnections=("total_disconnections", "sum"),
        total_auth=("total_auth", "sum"),
        total_unique_client_hours=("unique_clients", "sum"),
        avg_unique_clients=("unique_clients", "mean"),
        max_unique_clients=("unique_clients", "max"),
        avg_disconnection_rate=("disconnection_rate", "mean"),
        max_disconnection_rate=("disconnection_rate", "max"),
        high_disconnection_hours=("disconnection_rate", lambda values: int((values >= 0.5).sum())),
        zero_connection_hours=("total_connections", lambda values: int((values <= 0).sum())),
        active_hours=("timestamp_hour", lambda values: int(pd.Series(values).notna().sum())),
        offline_hours=("status", lambda values: int(pd.Series(values).astype(str).str.lower().eq("offline").sum())),
        dormant_hours=("status", lambda values: int(pd.Series(values).astype(str).str.lower().eq("dormant").sum())),
        status_mode=("status", lambda values: pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else "unknown"),
        first_hour=("timestamp_hour", "min"),
        last_hour=("timestamp_hour", "max"),
    ).reset_index()

    grouped["avg_unique_clients"] = grouped["avg_unique_clients"].round(2)
    grouped["avg_disconnection_rate"] = grouped["avg_disconnection_rate"].round(4)
    grouped["max_disconnection_rate"] = grouped["max_disconnection_rate"].round(4)
    return grouped


def _status_to_risk(status: object) -> float:
    normalized = str(status or "").strip().lower()
    if normalized == "offline":
        return 95.0
    if normalized == "dormant":
        return 70.0
    if normalized == "online":
        return 10.0
    return 35.0


def _series_from_frame(dataframe: pd.DataFrame, column_name: str, default_value: object = None) -> pd.Series:
    """Devuelve una serie existente o una serie por defecto del mismo largo."""
    if column_name in dataframe.columns:
        return dataframe[column_name]
    return pd.Series([default_value] * len(dataframe), index=dataframe.index)


def build_operational_mart(package: dict[str, object]) -> pd.DataFrame:
    """Construye la tabla operativa final por AP para el modo Meraki."""
    access_dim = build_access_point_dimension(package)
    hourly_features = build_hourly_ap_features(package)
    client_summary = build_client_ap_summary(package)
    event_summary = build_event_ap_summary(package)

    if access_dim.empty and hourly_features.empty:
        return pd.DataFrame()

    mart_df = access_dim.copy() if not access_dim.empty else hourly_features.copy()
    for companion_df in [hourly_features, client_summary, event_summary]:
        if companion_df.empty:
            continue
        if "ap_name" not in mart_df.columns:
            mart_df = companion_df.copy()
            continue
        merge_columns = [column_name for column_name in companion_df.columns if column_name != "ap_name"]
        mart_df = mart_df.merge(companion_df[["ap_name"] + merge_columns], on="ap_name", how="outer")

    status_series = _series_from_frame(mart_df, "status")
    if "status_mode" in mart_df.columns:
        status_series = status_series.fillna(mart_df["status_mode"])
    mart_df["status"] = status_series.fillna("unknown").astype(str)
    zone_series = _series_from_frame(mart_df, "zone_name")
    ap_series = _series_from_frame(mart_df, "ap_name", "")
    mart_df["zone_name"] = zone_series.fillna(ap_series).map(extract_zone_name)

    for numeric_col in [
        "total_events",
        "total_connections",
        "total_disconnections",
        "total_auth",
        "total_unique_client_hours",
        "avg_unique_clients",
        "max_unique_clients",
        "clients_reported",
        "usage_mb_total",
        "avg_disconnection_rate",
        "max_disconnection_rate",
        "high_disconnection_hours",
        "zero_connection_hours",
        "active_hours",
        "offline_hours",
        "dormant_hours",
        "connectivity_transition_count",
        "connectivity_unique_codes",
    ]:
        mart_df[numeric_col] = pd.to_numeric(
            _series_from_frame(mart_df, numeric_col, 0.0),
            errors="coerce",
        ).fillna(0.0)

    mart_df["status_risk"] = mart_df["status"].map(_status_to_risk).fillna(35.0)
    mart_df["disconnection_risk"] = (
        mart_df["max_disconnection_rate"].clip(lower=0, upper=1).mul(60)
        + mart_df["high_disconnection_hours"].clip(upper=24).div(24).mul(40)
    ).clip(upper=100)
    mart_df["anomaly_risk"] = (
        mart_df["zero_connection_hours"].clip(upper=24).div(24).mul(50)
        + mart_df["offline_hours"].clip(upper=24).div(24).mul(35)
        + mart_df["dormant_hours"].clip(upper=24).div(24).mul(15)
    ).clip(upper=100)

    if mart_df["max_unique_clients"].max() > 0:
        demand_base = mart_df["max_unique_clients"] / mart_df["max_unique_clients"].max()
    else:
        demand_base = pd.Series(0.0, index=mart_df.index)
    if mart_df["usage_mb_total"].max() > 0:
        usage_base = mart_df["usage_mb_total"] / mart_df["usage_mb_total"].max()
    else:
        usage_base = pd.Series(0.0, index=mart_df.index)
    mart_df["demand_impact"] = ((demand_base * 0.65 + usage_base * 0.35) * 100).round(2)

    mart_df["recurrence_risk"] = (
        mart_df["connectivity_transition_count"].clip(upper=20).div(20).mul(45)
        + mart_df["connectivity_unique_codes"].clip(upper=10).div(10).mul(25)
        + mart_df["total_disconnections"].clip(upper=1000).div(1000).mul(30)
    ).clip(upper=100)

    mart_df["operational_risk_score"] = (
        mart_df["status_risk"] * 0.32
        + mart_df["disconnection_risk"] * 0.28
        + mart_df["anomaly_risk"] * 0.20
        + mart_df["demand_impact"] * 0.12
        + mart_df["recurrence_risk"] * 0.08
    ).round(2)
    mart_df["ap_health_score"] = (100 - mart_df["operational_risk_score"]).clip(lower=0, upper=100).round(2)

    evidence_components = [
        mart_df["total_events"].gt(0).astype(int),
        mart_df["total_connections"].ge(0).astype(int),
        mart_df["clients_reported"].ge(0).astype(int),
        mart_df["status"].ne("unknown").astype(int),
        mart_df["connectivity_transition_count"].ge(0).astype(int),
    ]
    mart_df["evidence_level"] = (sum(evidence_components) / len(evidence_components) * 100).round(2)

    mart_df["risk_classification"] = pd.cut(
        mart_df["operational_risk_score"],
        bins=[-1, 19, 39, 59, 79, 1000],
        labels=["Observacion", "Bajo", "Medio", "Alto", "Critico"],
    ).astype(str)

    mart_df["recommended_action"] = np.select(
        [
            mart_df["status"].str.lower().eq("offline"),
            mart_df["status"].str.lower().eq("dormant"),
            mart_df["max_disconnection_rate"].ge(0.8),
            mart_df["zero_connection_hours"].ge(6),
        ],
        [
            "Validar energia, backhaul y disponibilidad del AP en campo.",
            "Revisar conectividad, uso real y consistencia del estado reportado por Meraki.",
            "Inspeccionar estabilidad de sesión, autenticación y calidad de enlace.",
            "Confirmar si la ausencia de conexiones corresponde a una ventana real de baja actividad o a una degradación.",
        ],
        default="Mantener monitoreo preventivo y revisión periódica.",
    )

    mart_df["limitations"] = (
        "Paquete Meraki curado y anonimizado. Sin coordenadas exactas del AP. "
        "connectivity_history se interpreta como texto exportado y sus códigos requieren validación técnica."
    )

    preferred_columns = [
        "ap_name",
        "zone_name",
        "status",
        "serial",
        "local_ip",
        "total_events",
        "total_connections",
        "total_disconnections",
        "total_auth",
        "total_unique_client_hours",
        "avg_unique_clients",
        "max_unique_clients",
        "clients_reported",
        "usage_mb_total",
        "avg_disconnection_rate",
        "max_disconnection_rate",
        "high_disconnection_hours",
        "zero_connection_hours",
        "active_hours",
        "last_seen_event",
        "dominant_event_type",
        "top_event_detail",
        "status_risk",
        "disconnection_risk",
        "anomaly_risk",
        "demand_impact",
        "recurrence_risk",
        "operational_risk_score",
        "ap_health_score",
        "evidence_level",
        "risk_classification",
        "recommended_action",
        "limitations",
        "connectivity_transition_count",
        "connectivity_unique_codes",
        "offline_hours",
        "dormant_hours",
    ]
    preferred_columns = [column_name for column_name in preferred_columns if column_name in mart_df.columns]
    return mart_df[preferred_columns].sort_values("operational_risk_score", ascending=False).reset_index(drop=True)
