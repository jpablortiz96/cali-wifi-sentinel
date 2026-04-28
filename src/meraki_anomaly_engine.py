from __future__ import annotations

from typing import Any

import pandas as pd

from src.meraki_features import extract_zone_name
from src.utils import get_timestamp


def detect_hourly_anomalies(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Detecta anomalías horarias por AP usando su propio baseline histórico."""
    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame()

    df = hourly_df.copy()
    df["timestamp_hour"] = pd.to_datetime(df.get("timestamp_hour"), errors="coerce")
    for column_name in [
        "total_connections",
        "total_disconnections",
        "total_auth",
        "unique_clients",
        "disconnection_rate",
        "total_events",
    ]:
        df[column_name] = pd.to_numeric(df.get(column_name), errors="coerce").fillna(0.0)

    anomalies: list[dict[str, object]] = []
    for ap_name, ap_slice in df.groupby("ap_name", dropna=False):
        historical = ap_slice.sort_values("timestamp_hour").copy()
        if historical.empty:
            continue

        connections_baseline = float(historical["total_connections"].median())
        connections_p10 = float(historical["total_connections"].quantile(0.10))
        disconnection_p95 = float(historical["disconnection_rate"].quantile(0.95))
        active_reference = bool(historical["total_connections"].gt(0).any())

        for _, row in historical.iterrows():
            timestamp_value = row.get("timestamp_hour")
            status_value = str(row.get("status", "")).strip().lower()
            total_connections = float(row.get("total_connections", 0) or 0)
            disconnection_rate = float(row.get("disconnection_rate", 0) or 0)

            if status_value in {"offline", "dormant"}:
                anomalies.append(
                    {
                        "ap_name": ap_name,
                        "zone_name": extract_zone_name(ap_name),
                        "timestamp_hour": timestamp_value,
                        "anomaly_type": "status_degradado",
                        "severity": "Alta" if status_value == "offline" else "Media",
                        "evidence": f"Estado reportado: {status_value}.",
                        "source": "meraki_package",
                    }
                )

            if active_reference and total_connections == 0:
                anomalies.append(
                    {
                        "ap_name": ap_name,
                        "zone_name": extract_zone_name(ap_name),
                        "timestamp_hour": timestamp_value,
                        "anomaly_type": "sin_conexiones_en_ap_activo",
                        "severity": "Alta",
                        "evidence": (
                            f"Se observaron 0 conexiones en una hora para un AP con actividad histórica. "
                            f"Baseline mediano: {connections_baseline:.2f}."
                        ),
                        "source": "meraki_package",
                    }
                )
            elif connections_baseline > 0 and total_connections <= max(connections_p10, connections_baseline * 0.25):
                anomalies.append(
                    {
                        "ap_name": ap_name,
                        "zone_name": extract_zone_name(ap_name),
                        "timestamp_hour": timestamp_value,
                        "anomaly_type": "conexiones_muy_bajas",
                        "severity": "Media",
                        "evidence": (
                            f"Conexiones por debajo del baseline del AP. Hora: {total_connections:.2f}; "
                            f"baseline mediano: {connections_baseline:.2f}; p10: {connections_p10:.2f}."
                        ),
                        "source": "meraki_package",
                    }
                )

            if disconnection_rate >= max(disconnection_p95, 0.8):
                anomalies.append(
                    {
                        "ap_name": ap_name,
                        "zone_name": extract_zone_name(ap_name),
                        "timestamp_hour": timestamp_value,
                        "anomaly_type": "tasa_desconexion_alta",
                        "severity": "Alta" if disconnection_rate >= 0.95 else "Media",
                        "evidence": (
                            f"Tasa de desconexión de {disconnection_rate:.2f}, por encima del p95 histórico "
                            f"({disconnection_p95:.2f})."
                        ),
                        "source": "meraki_package",
                    }
                )

    anomalies_df = pd.DataFrame(anomalies)
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.sort_values(["ap_name", "timestamp_hour"]).reset_index(drop=True)
    return anomalies_df


def classify_ap_risk(row: pd.Series | dict[str, Any]) -> str:
    """Clasifica el riesgo del AP a partir del score operativo."""
    get_value = row.get if isinstance(row, dict) else row.__getitem__
    score = float(get_value("operational_risk_score") or 0)
    if score >= 80:
        return "Critico"
    if score >= 60:
        return "Alto"
    if score >= 40:
        return "Medio"
    if score >= 20:
        return "Bajo"
    return "Observacion"


def generate_meraki_work_orders(
    operational_mart: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """Genera órdenes de trabajo específicas por AP del paquete Meraki."""
    if operational_mart is None or operational_mart.empty:
        return pd.DataFrame()

    anomalies = anomalies if isinstance(anomalies, pd.DataFrame) else pd.DataFrame()
    grouped_anomalies = (
        anomalies.groupby("ap_name").agg(
            anomaly_count=("anomaly_type", "count"),
            top_anomaly=("anomaly_type", lambda values: values.iloc[0] if len(values) else ""),
            top_evidence=("evidence", lambda values: values.iloc[0] if len(values) else ""),
        )
        if not anomalies.empty
        else pd.DataFrame()
    )

    orders: list[dict[str, object]] = []
    timestamp = get_timestamp()
    for index, row in operational_mart.iterrows():
        ap_name = str(row.get("ap_name", "AP sin identificar"))
        classification = str(row.get("risk_classification") or classify_ap_risk(row))
        status = str(row.get("status", "unknown"))
        score = float(row.get("operational_risk_score", 0) or 0)
        health = float(row.get("ap_health_score", 0) or 0)
        anomaly_count = int(grouped_anomalies.loc[ap_name, "anomaly_count"]) if not grouped_anomalies.empty and ap_name in grouped_anomalies.index else 0
        top_anomaly = grouped_anomalies.loc[ap_name, "top_anomaly"] if not grouped_anomalies.empty and ap_name in grouped_anomalies.index else ""
        top_evidence = grouped_anomalies.loc[ap_name, "top_evidence"] if not grouped_anomalies.empty and ap_name in grouped_anomalies.index else ""

        if classification not in {"Critico", "Alto"} and anomaly_count == 0 and status.lower() == "online":
            continue

        prioridad = "Alta" if classification == "Critico" or status.lower() == "offline" else "Media"
        nivel_confianza = "Alto" if float(row.get("evidence_level", 0) or 0) >= 80 else "Medio"
        if anomaly_count == 0 and status.lower() != "offline":
            prioridad = "Observacion"
            nivel_confianza = "Bajo"

        evidencia = top_evidence or (
            f"AP {ap_name} con score operativo {score:.2f}, health score {health:.2f}, "
            f"estado {status} y {anomaly_count} anomalías horarias observadas."
        )

        orders.append(
            {
                "id": f"WO-MERAKI-{timestamp}-{index + 1:03d}",
                "ap_name": ap_name,
                "zona": ap_name,
                "zone_name": row.get("zone_name", extract_zone_name(ap_name)),
                "tipo_alerta": top_anomaly or "riesgo_operativo_meraki",
                "evidencia": evidencia,
                "prioridad": prioridad,
                "accion_recomendada": row.get("recommended_action", "Validar el AP y revisar métricas horarias."),
                "nivel_confianza": nivel_confianza,
                "campos_usados": "status, total_connections, disconnection_rate, connectivity_history, clients, events",
                "timestamp": timestamp,
                "final_impact_score": score,
                "classification": classification,
                "social_criticality_score": None,
                "decision_passport_id": None,
                "source": "meraki_package",
                "datos_usados": "hourly_metrics, access_points, clients, network_events",
                "limitaciones": (
                    "Sin coordenadas exactas del AP. connectivity_history interpretado como texto exportado."
                ),
            }
        )

    return pd.DataFrame(orders)


def build_meraki_decision_passports(
    operational_mart: pd.DataFrame,
    work_orders: pd.DataFrame,
) -> list[dict[str, object]]:
    """Genera pasaportes específicos por AP en modo Meraki."""
    if operational_mart is None or operational_mart.empty:
        return []

    work_orders = work_orders if isinstance(work_orders, pd.DataFrame) else pd.DataFrame()
    passports: list[dict[str, object]] = []
    for _, row in operational_mart.sort_values("operational_risk_score", ascending=False).head(10).iterrows():
        ap_name = str(row.get("ap_name", "AP sin identificar"))
        order_row = (
            work_orders[work_orders["ap_name"].astype(str) == ap_name].iloc[0].to_dict()
            if not work_orders.empty and "ap_name" in work_orders.columns and work_orders["ap_name"].astype(str).eq(ap_name).any()
            else {}
        )

        passports.append(
            {
                "decision_id": f"DP-MERAKI-{ap_name.replace(' ', '_')}",
                "ap_name": ap_name,
                "zona": row.get("zone_name", extract_zone_name(ap_name)),
                "estado": row.get("status", "unknown"),
                "clasificacion": row.get("risk_classification", "Observacion"),
                "score_final": float(row.get("operational_risk_score", 0) or 0),
                "health_score": float(row.get("ap_health_score", 0) or 0),
                "por_que_importa": (
                    f"El AP {ap_name} concentra evidencia Meraki de riesgo operativo y su estado actual requiere revisión priorizada."
                ),
                "evidencia_tecnica": [
                    f"Estado: {row.get('status', 'unknown')}",
                    f"Conexiones totales: {float(row.get('total_connections', 0) or 0):.0f}",
                    f"Desconexiones totales: {float(row.get('total_disconnections', 0) or 0):.0f}",
                    f"Tasa máxima de desconexión: {float(row.get('max_disconnection_rate', 0) or 0):.2f}",
                ],
                "evidencia_contextual": [
                    f"Zona extraída: {row.get('zone_name', extract_zone_name(ap_name))}",
                    "No se dispone de coordenadas exactas del AP dentro del paquete oficial.",
                ],
                "demanda": {
                    "clients_reported": float(row.get("clients_reported", 0) or 0),
                    "usage_mb_total": float(row.get("usage_mb_total", 0) or 0),
                    "max_unique_clients": float(row.get("max_unique_clients", 0) or 0),
                },
                "recomendacion": row.get("recommended_action", "Validar en campo."),
                "accion_recomendada": row.get("recommended_action", "Validar en campo."),
                "orden_trabajo_asociada": order_row.get("id"),
                "nivel_confianza": "Alta" if float(row.get("evidence_level", 0) or 0) >= 80 else "Media",
                "limitaciones": [
                    "Paquete curado/anonimizado; no representa telemetría en vivo.",
                    "Sin coordenadas exactas del AP.",
                    "connectivity_history está en texto exportado y sus códigos requieren validación técnica con Meraki.",
                ],
                "datos_usados": [
                    "ap_hourly_metrics_curated.csv",
                    "access_points_curated.csv",
                    "clients_curated.csv",
                    "network_events_curated.csv",
                ],
                "datos_faltantes": ["Coordenadas exactas del AP"],
                "trazabilidad": {
                    "source": "meraki_package",
                    "work_order_id": order_row.get("id"),
                },
            }
        )

    return passports
