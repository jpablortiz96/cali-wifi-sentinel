from __future__ import annotations


EVENTS_COLUMNS = [
    "timestamp",
    "ap_name",
    "ssid",
    "client_id",
    "client_description",
    "event_category",
    "event_type",
    "event_detail",
]

CLIENTS_COLUMNS = [
    "client_id",
    "status",
    "client_description",
    "last_seen",
    "usage_mb",
    "device_type",
    "ap_name",
    "policy",
    "onboarding",
]

ACCESS_POINTS_COLUMNS = [
    "ap_name",
    "mac",
    "serial",
    "status",
    "local_ip",
    "connectivity_history",
]

HOURLY_COLUMNS = [
    "timestamp_hour",
    "ap_name",
    "total_events",
    "total_connections",
    "total_disconnections",
    "total_auth",
    "unique_clients",
    "disconnection_rate",
    "status",
]


def build_meraki_schema_mapping() -> dict[str, str | None]:
    """Devuelve un mapeo canónico para el paquete Meraki oficial."""
    return {
        "date_col": "timestamp_hour",
        "zone_col": "ap_name",
        "connections_col": "total_connections",
        "traffic_col": "usage_mb_total",
        "status_col": "status",
        "latitude_col": None,
        "longitude_col": None,
        "territory_col": "zone_name",
        "disconnections_col": "total_disconnections",
        "auth_col": "total_auth",
        "unique_clients_col": "unique_clients",
        "disconnection_rate_col": "disconnection_rate",
        "ap_col": "ap_name",
    }

