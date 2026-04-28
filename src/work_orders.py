from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema_mapper import SchemaMapping
from src.utils import get_timestamp, normalize_text


WORK_ORDER_COLUMNS = [
    "id",
    "ap_name",
    "zona",
    "tipo_alerta",
    "evidencia",
    "prioridad",
    "accion_recomendada",
    "nivel_confianza",
    "campos_usados",
    "timestamp",
    "final_impact_score",
    "classification",
    "social_criticality_score",
    "decision_passport_id",
    "source",
    "datos_usados",
    "limitaciones",
]


STATUS_RULES = [
    {
        "keywords": ["critico", "critical", "offline", "down"],
        "tipo_alerta": "Estado critico reportado",
        "prioridad": "Alta",
        "nivel_confianza": "Medio",
        "accion_recomendada": "Validar disponibilidad del punto de acceso y revisar soporte en campo.",
    },
    {
        "keywords": ["falla", "caido", "caida", "inactivo", "error"],
        "tipo_alerta": "Estado con posible falla",
        "prioridad": "Media",
        "nivel_confianza": "Medio",
        "accion_recomendada": "Revisar logs, energia, backhaul y estado operativo del punto.",
    },
]


def _empty_work_orders() -> pd.DataFrame:
    """Devuelve una tabla vacia con el esquema esperado."""
    return pd.DataFrame(columns=WORK_ORDER_COLUMNS)


def _safe_zone_name(value: object) -> str:
    """Normaliza el nombre de la zona para mensajes y ordenes."""
    if value is None or pd.isna(value):
        return "Zona no identificada"
    return str(value)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convierte una serie a numerico de forma tolerante."""
    return pd.to_numeric(series, errors="coerce")


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convierte una serie a fecha de forma tolerante."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _build_order(
    counter: int,
    zona: str,
    tipo_alerta: str,
    evidencia: str,
    prioridad: str,
    accion_recomendada: str,
    nivel_confianza: str,
    campos_usados: list[str],
    timestamp: str,
) -> dict[str, object]:
    """Construye una orden de trabajo con trazabilidad simple."""
    return {
        "id": f"WO-{timestamp}-{counter:03d}",
        "ap_name": None,
        "zona": zona,
        "tipo_alerta": tipo_alerta,
        "evidencia": evidencia,
        "prioridad": prioridad,
        "accion_recomendada": accion_recomendada,
        "nivel_confianza": nivel_confianza,
        "campos_usados": ", ".join(campos_usados),
        "timestamp": timestamp,
        "final_impact_score": None,
        "classification": None,
        "social_criticality_score": None,
        "decision_passport_id": None,
        "source": "generic_dataset",
        "datos_usados": None,
        "limitaciones": None,
    }


def _status_orders(dataframe: pd.DataFrame, schema_mapping: SchemaMapping, timestamp: str) -> list[dict[str, object]]:
    """Detecta posibles incidentes a partir de una columna de estado textual."""
    zone_col = schema_mapping.get("zone_col")
    status_col = schema_mapping.get("status_col")

    if not zone_col or not status_col:
        return []

    orders: list[dict[str, object]] = []
    normalized_status = dataframe[status_col].fillna("").astype(str).map(normalize_text)

    for index, normalized_value in normalized_status.items():
        if not normalized_value:
            continue

        for rule in STATUS_RULES:
            if any(keyword in normalized_value for keyword in rule["keywords"]):
                raw_status = str(dataframe.at[index, status_col])
                zona = _safe_zone_name(dataframe.at[index, zone_col])
                orders.append(
                    _build_order(
                        counter=len(orders) + 1,
                        zona=zona,
                        tipo_alerta=rule["tipo_alerta"],
                        evidencia=(
                            f"Se detecto el valor '{raw_status}' en la columna '{status_col}'. "
                            "La regla es heuristica y requiere validacion operativa."
                        ),
                        prioridad=rule["prioridad"],
                        accion_recomendada=rule["accion_recomendada"],
                        nivel_confianza=rule["nivel_confianza"],
                        campos_usados=[zone_col, status_col],
                        timestamp=timestamp,
                    )
                )
                break

    return orders


def _connection_orders(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    timestamp: str,
) -> list[dict[str, object]]:
    """Detecta zonas con conexiones inusualmente bajas usando reglas simples."""
    zone_col = schema_mapping.get("zone_col")
    connections_col = schema_mapping.get("connections_col")
    date_col = schema_mapping.get("date_col")

    if not zone_col or not connections_col:
        return []

    working_df = pd.DataFrame(
        {
            "zona": dataframe[zone_col].map(_safe_zone_name),
            "connections_value": _to_numeric(dataframe[connections_col]),
        }
    ).dropna(subset=["connections_value"])

    if working_df.empty:
        return []

    orders: list[dict[str, object]] = []
    overall_mean = float(working_df["connections_value"].mean())
    zone_stats = (
        working_df.groupby("zona", dropna=False)["connections_value"]
        .agg(["mean", "count"])
        .reset_index()
    )

    if overall_mean > 0:
        low_threshold = overall_mean * 0.35
        low_usage_zones = zone_stats[zone_stats["mean"] <= low_threshold]

        for _, row in low_usage_zones.iterrows():
            count_value = int(row["count"])
            orders.append(
                _build_order(
                    counter=len(orders) + 1,
                    zona=str(row["zona"]),
                    tipo_alerta="Conectividad baja observada",
                    evidencia=(
                        f"El promedio de '{connections_col}' en la zona fue {row['mean']:.2f}, "
                        f"por debajo del umbral heuristico de {low_threshold:.2f} "
                        f"(35% del promedio general de {overall_mean:.2f})."
                    ),
                    prioridad="Media" if count_value >= 3 else "Observacion",
                    accion_recomendada=(
                        "Validar demanda real, revisar disponibilidad del servicio "
                        "y confirmar si la baja actividad es persistente."
                    ),
                    nivel_confianza="Medio" if count_value >= 3 else "Bajo",
                    campos_usados=[zone_col, connections_col],
                    timestamp=timestamp,
                )
            )

    if date_col:
        dated_df = pd.DataFrame(
            {
                "zona": dataframe[zone_col].map(_safe_zone_name),
                "connections_value": _to_numeric(dataframe[connections_col]),
                "date_value": _to_datetime(dataframe[date_col]),
            }
        ).dropna(subset=["connections_value", "date_value"])

        for zona, group in dated_df.groupby("zona", dropna=False):
            ordered_group = group.sort_values("date_value")
            if len(ordered_group) < 3:
                continue

            historical_mean = float(ordered_group.iloc[:-1]["connections_value"].mean())
            latest_value = float(ordered_group.iloc[-1]["connections_value"])

            if historical_mean > 0 and latest_value <= historical_mean * 0.5:
                orders.append(
                    _build_order(
                        counter=len(orders) + 1,
                        zona=str(zona),
                        tipo_alerta="Caida reciente de conexiones",
                        evidencia=(
                            f"El ultimo valor de '{connections_col}' fue {latest_value:.2f}, "
                            f"por debajo del 50% del promedio historico previo de {historical_mean:.2f}."
                        ),
                        prioridad="Media",
                        accion_recomendada=(
                            "Revisar eventos recientes de la zona y confirmar si existe una degradacion real."
                        ),
                        nivel_confianza="Medio" if len(ordered_group) >= 5 else "Bajo",
                        campos_usados=[zone_col, connections_col, date_col],
                        timestamp=timestamp,
                    )
                )

    return orders


def _traffic_orders(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    timestamp: str,
) -> list[dict[str, object]]:
    """Detecta trafico bajo o nulo sin afirmar causas definitivas."""
    zone_col = schema_mapping.get("zone_col")
    traffic_col = schema_mapping.get("traffic_col")
    date_col = schema_mapping.get("date_col")

    if not zone_col or not traffic_col:
        return []

    working_df = pd.DataFrame(
        {
            "zona": dataframe[zone_col].map(_safe_zone_name),
            "traffic_value": _to_numeric(dataframe[traffic_col]),
        }
    ).dropna(subset=["traffic_value"])

    if working_df.empty:
        return []

    orders: list[dict[str, object]] = []
    overall_mean = float(working_df["traffic_value"].mean())
    zone_stats = (
        working_df.groupby("zona", dropna=False)["traffic_value"]
        .agg(["mean", "count", "min"])
        .reset_index()
    )

    for _, row in zone_stats.iterrows():
        count_value = int(row["count"])

        if float(row["min"]) == 0:
            orders.append(
                _build_order(
                    counter=len(orders) + 1,
                    zona=str(row["zona"]),
                    tipo_alerta="Trafico nulo detectado",
                    evidencia=(
                        f"Se detecto al menos un registro con '{traffic_col}' igual a 0. "
                        "Puede corresponder a inactividad, falla o una ventana normal de bajo uso."
                    ),
                    prioridad="Media" if count_value >= 2 else "Observacion",
                    accion_recomendada=(
                        "Verificar si el trafico nulo coincide con ventanas operativas, mantenimientos o fallas."
                    ),
                    nivel_confianza="Medio" if count_value >= 2 else "Bajo",
                    campos_usados=[zone_col, traffic_col],
                    timestamp=timestamp,
                )
            )

        if overall_mean > 0 and float(row["mean"]) <= overall_mean * 0.30:
            orders.append(
                _build_order(
                    counter=len(orders) + 1,
                    zona=str(row["zona"]),
                    tipo_alerta="Trafico bajo observado",
                    evidencia=(
                        f"El promedio de '{traffic_col}' en la zona fue {row['mean']:.2f}, "
                        f"por debajo del 30% del promedio general de {overall_mean:.2f}."
                    ),
                    prioridad="Observacion",
                    accion_recomendada=(
                        "Comparar con la demanda esperada de la zona y validar si el patron es persistente."
                    ),
                    nivel_confianza="Bajo" if count_value < 3 else "Medio",
                    campos_usados=[zone_col, traffic_col],
                    timestamp=timestamp,
                )
            )

    if date_col:
        dated_df = pd.DataFrame(
            {
                "zona": dataframe[zone_col].map(_safe_zone_name),
                "traffic_value": _to_numeric(dataframe[traffic_col]),
                "date_value": _to_datetime(dataframe[date_col]),
            }
        ).dropna(subset=["traffic_value", "date_value"])

        for zona, group in dated_df.groupby("zona", dropna=False):
            ordered_group = group.sort_values("date_value")
            if len(ordered_group) < 3:
                continue

            historical_mean = float(ordered_group.iloc[:-1]["traffic_value"].mean())
            latest_value = float(ordered_group.iloc[-1]["traffic_value"])

            if historical_mean > 0 and latest_value <= historical_mean * 0.4:
                orders.append(
                    _build_order(
                        counter=len(orders) + 1,
                        zona=str(zona),
                        tipo_alerta="Caida reciente de trafico",
                        evidencia=(
                            f"El ultimo valor de '{traffic_col}' fue {latest_value:.2f}, "
                            f"por debajo del 40% del promedio historico previo de {historical_mean:.2f}."
                        ),
                        prioridad="Observacion",
                        accion_recomendada=(
                            "Revisar si hubo cambios operativos o de demanda antes de escalar el caso."
                        ),
                        nivel_confianza="Bajo" if len(ordered_group) < 5 else "Medio",
                        campos_usados=[zone_col, traffic_col, date_col],
                        timestamp=timestamp,
                    )
                )

    return orders


def _attach_score_metadata(
    work_orders_df: pd.DataFrame,
    impact_scores_df: pd.DataFrame | None = None,
    decision_passports: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Adjunta scores y pasaportes sin romper compatibilidad del Paso 3."""
    if work_orders_df.empty:
        return _empty_work_orders()

    enriched_df = work_orders_df.copy()

    if impact_scores_df is not None and not impact_scores_df.empty:
        score_columns = [
            column_name
            for column_name in [
                "zona",
                "final_impact_score",
                "classification",
                "social_criticality_score",
            ]
            if column_name in impact_scores_df.columns
        ]
        enriched_df = enriched_df.merge(
            impact_scores_df[score_columns].drop_duplicates(subset=["zona"]),
            on="zona",
            how="left",
            suffixes=("", "_score"),
        )
        for column_name in ["final_impact_score", "classification", "social_criticality_score"]:
            score_column_name = f"{column_name}_score"
            if score_column_name in enriched_df.columns:
                enriched_df[column_name] = enriched_df[score_column_name].combine_first(
                    enriched_df[column_name]
                )
                enriched_df = enriched_df.drop(columns=[score_column_name])

    if decision_passports:
        passport_map = {str(passport["zona"]): passport["decision_id"] for passport in decision_passports}
        enriched_df["decision_passport_id"] = enriched_df["zona"].astype(str).map(passport_map)

    for column_name in WORK_ORDER_COLUMNS:
        if column_name not in enriched_df.columns:
            enriched_df[column_name] = None

    return enriched_df[WORK_ORDER_COLUMNS]


def generate_work_orders(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    impact_scores_df: pd.DataFrame | None = None,
    decision_passports: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Genera ordenes preliminares basadas en evidencia heuristica y transparente."""
    if dataframe.empty:
        return _empty_work_orders()

    is_meraki_like = {
        "ap_name",
        "operational_risk_score",
        "ap_health_score",
    }.issubset(set(dataframe.columns))
    if is_meraki_like or str(dataframe.attrs.get("source", "")).strip().lower() == "meraki_package":
        from src.meraki_anomaly_engine import detect_hourly_anomalies, generate_meraki_work_orders

        hourly_like = dataframe.attrs.get("meraki_hourly_metrics")
        if isinstance(hourly_like, pd.DataFrame):
            anomalies_df = detect_hourly_anomalies(hourly_like)
        else:
            anomalies_df = pd.DataFrame()

        meraki_orders = generate_meraki_work_orders(dataframe, anomalies_df)
        return _attach_score_metadata(
            meraki_orders,
            impact_scores_df=impact_scores_df,
            decision_passports=decision_passports,
        )

    timestamp = get_timestamp()
    collected_orders: list[dict[str, object]] = []

    collected_orders.extend(_status_orders(dataframe, schema_mapping, timestamp))
    collected_orders.extend(_connection_orders(dataframe, schema_mapping, timestamp))
    collected_orders.extend(_traffic_orders(dataframe, schema_mapping, timestamp))

    if not collected_orders:
        return _empty_work_orders()

    deduplicated_orders: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for order in collected_orders:
        unique_key = (
            str(order["zona"]),
            str(order["tipo_alerta"]),
            str(order["evidencia"]),
        )
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            deduplicated_orders.append(order)

    priority_order = {"Alta": 0, "Media": 1, "Observacion": 2}
    orders_df = pd.DataFrame(deduplicated_orders)
    orders_df = orders_df.sort_values(
        by=["prioridad", "zona", "tipo_alerta"],
        key=lambda column: column.map(priority_order).fillna(9)
        if column.name == "prioridad"
        else column,
    ).reset_index(drop=True)

    for position in range(len(orders_df)):
        orders_df.at[position, "id"] = f"WO-{timestamp}-{position + 1:03d}"

    orders_df = _attach_score_metadata(
        orders_df,
        impact_scores_df=impact_scores_df,
        decision_passports=decision_passports,
    )

    return orders_df
