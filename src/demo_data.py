from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic_wifi_data(n_zones: int = 30, days: int = 14) -> pd.DataFrame:
    """Genera un dataset sintetico solo para pruebas y demo interna."""
    rng = np.random.default_rng(42)

    n_zones = max(int(n_zones), 5)
    days = max(int(days), 3)

    base_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1)
    dates = [base_date + pd.Timedelta(days=offset) for offset in range(days)]

    communes = [
        "Comuna 1",
        "Comuna 2",
        "Comuna 3",
        "Comuna 4",
        "Comuna 5",
        "Comuna 6",
        "Comuna 7",
        "Comuna 8",
    ]
    barrios = [
        "San Antonio",
        "Granada",
        "Aguablanca",
        "Tequendama",
        "Siloe",
        "Melendez",
        "Versalles",
        "Floralia",
    ]

    critical_zones = set(rng.choice(np.arange(n_zones), size=max(2, n_zones // 8), replace=False).tolist())
    offline_zones = set(rng.choice(np.arange(n_zones), size=max(2, n_zones // 10), replace=False).tolist())
    low_traffic_zones = set(rng.choice(np.arange(n_zones), size=max(3, n_zones // 6), replace=False).tolist())
    degraded_zones = set(rng.choice(np.arange(n_zones), size=max(3, n_zones // 5), replace=False).tolist())

    rows: list[dict[str, Any]] = []
    for zone_index in range(n_zones):
        zone_name = f"Zona WiFi {zone_index + 1:02d}"
        commune = communes[zone_index % len(communes)]
        barrio = barrios[zone_index % len(barrios)]
        lat = 3.4516 + rng.normal(0, 0.035)
        lon = -76.5320 + rng.normal(0, 0.035)

        zone_base_connections = max(20, int(rng.normal(180, 55)))
        zone_base_traffic = max(30, float(rng.normal(650, 180)))

        for day_offset, date_value in enumerate(dates):
            day_factor = 1.0 + 0.15 * np.sin(day_offset / 3)
            connections = max(0, int(zone_base_connections * day_factor + rng.normal(0, 18)))
            traffic = max(0.0, float(zone_base_traffic * day_factor + rng.normal(0, 55)))
            status = "activo"

            if zone_index in degraded_zones and day_offset >= max(2, days // 2):
                connections = max(0, int(connections * rng.uniform(0.35, 0.65)))
                traffic = max(0.0, float(traffic * rng.uniform(0.30, 0.60)))
                status = "degradado"

            if zone_index in low_traffic_zones:
                traffic = max(0.0, float(traffic * rng.uniform(0.08, 0.35)))

            if zone_index in critical_zones and day_offset >= max(1, days - 4):
                connections = max(0, int(connections * rng.uniform(0.05, 0.25)))
                traffic = max(0.0, float(traffic * rng.uniform(0.02, 0.20)))
                status = "critico"

            if zone_index in offline_zones and day_offset >= max(1, days - 3):
                connections = 0
                traffic = 0.0
                status = "offline"

            if rng.random() < 0.03:
                status = "inactivo"

            rows.append(
                {
                    "fecha": date_value.date().isoformat(),
                    "zona": zone_name,
                    "conexiones": int(connections),
                    "trafico_mb": round(float(traffic), 2),
                    "estado_ap": status,
                    "latitud": round(float(lat), 6),
                    "longitud": round(float(lon), 6),
                    "comuna": commune,
                    "barrio": barrio,
                    "tipo_dato": "SINTETICO_NO_OFICIAL",
                }
            )

    synthetic_df = pd.DataFrame(rows)
    synthetic_df = synthetic_df.sort_values(by=["zona", "fecha"]).reset_index(drop=True)
    return synthetic_df
