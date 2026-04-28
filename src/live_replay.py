from __future__ import annotations

from math import ceil
from typing import Any

import pandas as pd

from src.decision_passport import generate_passports_for_top_zones
from src.impact_scoring import calculate_impact_scores
from src.strategic_recommendations import generate_strategic_recommendations
from src.utils import get_timestamp
from src.work_orders import generate_work_orders
from src.resource_optimizer import optimize_crews


def prepare_replay_events(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Prepara un dataset para reproduccion operativa por lotes."""
    if isinstance(wifi_package, dict) and bool(wifi_package.get("is_official_package")):
        hourly_df = wifi_package.get("hourly_metrics", pd.DataFrame())
        if isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty:
            events_df = hourly_df.copy().reset_index(drop=True)
            events_df["replay_zone"] = events_df["ap_name"].fillna("AP no identificado").astype(str)
            events_df["replay_timestamp"] = pd.to_datetime(
                events_df.get("timestamp_hour"),
                errors="coerce",
            )
            events_df = events_df.sort_values("replay_timestamp", na_position="last").reset_index(drop=True)
            events_df["replay_step"] = range(1, len(events_df) + 1)
            warnings = []
            if events_df["replay_timestamp"].isna().all():
                warnings.append("No se pudo interpretar `timestamp_hour`; se usará el orden original de filas.")
            return {
                "events_df": events_df,
                "has_temporal_data": bool(events_df["replay_timestamp"].notna().any()),
                "warnings": warnings,
                "mode": "meraki_hourly",
            }

    events_df = dataframe.copy().reset_index(drop=True)
    warnings: list[str] = []

    zone_col = schema_mapping.get("zone_col")
    if zone_col and zone_col in events_df.columns:
        events_df["replay_zone"] = events_df[zone_col].astype(object).where(
            pd.notna(events_df[zone_col]),
            None,
        )
        events_df["replay_zone"] = events_df["replay_zone"].fillna("Zona no identificada").astype(str)
    else:
        warnings.append(
            "No hay columna de zona mapeada. La simulacion usa identificadores temporales por registro."
        )
        events_df["replay_zone"] = [f"registro_{index + 1}" for index in range(len(events_df))]

    date_col = schema_mapping.get("date_col")
    has_temporal_data = False
    if date_col and date_col in events_df.columns:
        parsed_dates = pd.to_datetime(events_df[date_col], errors="coerce", dayfirst=True)
        if parsed_dates.notna().any():
            events_df["replay_timestamp"] = parsed_dates
            events_df = events_df.sort_values(
                by=["replay_timestamp"],
                ascending=[True],
                na_position="last",
            ).reset_index(drop=True)
            has_temporal_data = True
        else:
            warnings.append(
                "La columna de fecha no pudo interpretarse de forma util. La simulacion usara el orden de filas."
            )
            events_df["replay_timestamp"] = pd.NaT
    else:
        warnings.append(
            "El dataset no tiene columna de fecha mapeada. La simulacion usa el orden de filas, no temporalidad real."
        )

    events_df["replay_step"] = range(1, len(events_df) + 1)

    return {
        "events_df": events_df,
        "has_temporal_data": has_temporal_data,
        "warnings": warnings,
        "mode": "generic",
    }


def get_replay_batch(events_df: pd.DataFrame, step: int, batch_size: int = 10) -> pd.DataFrame:
    """Devuelve el acumulado procesado hasta el lote actual."""
    if events_df.empty:
        return events_df.copy()

    safe_step = max(int(step), 1)
    safe_batch_size = max(int(batch_size), 1)
    upper_bound = min(len(events_df), safe_step * safe_batch_size)
    return events_df.head(upper_bound).copy().reset_index(drop=True)


def _derive_confidence_level(
    impact_scores: pd.DataFrame,
    work_orders: pd.DataFrame,
) -> str:
    """Resume confianza de la simulacion parcial."""
    if impact_scores.empty:
        return "Baja"

    mean_confidence = (
        float(impact_scores["data_confidence_score"].mean())
        if "data_confidence_score" in impact_scores.columns
        else 0.0
    )
    if mean_confidence >= 75 and not work_orders.empty:
        return "Alta"
    if mean_confidence >= 50:
        return "Media"
    return "Baja"


def run_replay_analysis(
    batch_df: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    available_crews: int = 3,
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Ejecuta analisis parcial sobre el lote acumulado sin romper la app."""
    warnings: list[str] = []

    if isinstance(wifi_package, dict) and bool(wifi_package.get("is_official_package")):
        from src.meraki_anomaly_engine import build_meraki_decision_passports, detect_hourly_anomalies
        from src.meraki_features import build_operational_mart
        from src.meraki_schema import build_meraki_schema_mapping

        max_timestamp = pd.to_datetime(batch_df.get("timestamp_hour"), errors="coerce").max()
        package_copy = dict(wifi_package)
        package_copy["hourly_metrics"] = batch_df.copy()
        if isinstance(max_timestamp, pd.Timestamp) and not pd.isna(max_timestamp):
            if isinstance(package_copy.get("events"), pd.DataFrame) and not package_copy["events"].empty:
                events_df = package_copy["events"].copy()
                events_df["timestamp"] = pd.to_datetime(events_df.get("timestamp"), errors="coerce")
                package_copy["events"] = events_df[events_df["timestamp"] <= max_timestamp].copy()
            if isinstance(package_copy.get("clients"), pd.DataFrame) and not package_copy["clients"].empty:
                clients_df = package_copy["clients"].copy()
                clients_df["last_seen"] = pd.to_datetime(clients_df.get("last_seen"), errors="coerce")
                package_copy["clients"] = clients_df[clients_df["last_seen"] <= max_timestamp].copy()

        try:
            operational_mart = build_operational_mart(package_copy)
            operational_mart.attrs["source"] = "meraki_package"
            operational_mart.attrs["meraki_hourly_metrics"] = batch_df.copy()
        except Exception as error:  # noqa: BLE001
            operational_mart = pd.DataFrame()
            warnings.append(f"No fue posible construir el mart operativo Meraki parcial: {error}")

        try:
            anomalies = detect_hourly_anomalies(batch_df.copy())
        except Exception as error:  # noqa: BLE001
            anomalies = pd.DataFrame()
            warnings.append(f"No fue posible detectar anomalías Meraki: {error}")

        try:
            work_orders = generate_work_orders(operational_mart, build_meraki_schema_mapping())
        except Exception as error:  # noqa: BLE001
            work_orders = pd.DataFrame()
            warnings.append(f"No fue posible generar órdenes Meraki: {error}")

        try:
            impact_scores = calculate_impact_scores(operational_mart, build_meraki_schema_mapping(), work_orders=work_orders)
        except Exception as error:  # noqa: BLE001
            impact_scores = pd.DataFrame()
            warnings.append(f"No fue posible calcular scores Meraki: {error}")

        try:
            recommendations = generate_strategic_recommendations(
                operational_mart,
                build_meraki_schema_mapping(),
                work_orders=work_orders,
                impact_scores_df=impact_scores,
            )
        except Exception as error:  # noqa: BLE001
            recommendations = pd.DataFrame()
            warnings.append(f"No fue posible generar recomendaciones estratégicas Meraki: {error}")

        try:
            crew_plan = optimize_crews(impact_scores, available_crews=available_crews)
        except Exception as error:  # noqa: BLE001
            crew_plan = {
                "recommended_zones": pd.DataFrame(),
                "waiting_zones": pd.DataFrame(),
                "coverage_territorial": "Sin datos",
                "riesgo_no_atencion": "Sin datos",
                "explanation": f"No fue posible optimizar cuadrillas: {error}",
            }
            warnings.append(f"No fue posible optimizar cuadrillas Meraki: {error}")

        try:
            decision_passports = build_meraki_decision_passports(operational_mart, work_orders)
        except Exception as error:  # noqa: BLE001
            decision_passports = []
            warnings.append(f"No fue posible generar pasaportes Meraki: {error}")

        confidence_level = _derive_confidence_level(impact_scores, work_orders)
        return {
            "processed_rows": int(len(batch_df)),
            "work_orders": work_orders,
            "recommendations": recommendations,
            "impact_scores": impact_scores,
            "crew_plan": crew_plan,
            "decision_passports": decision_passports,
            "meraki_anomalies": anomalies,
            "operational_mart": operational_mart,
            "warnings": warnings,
            "confidence_level": confidence_level,
        }

    try:
        work_orders = generate_work_orders(batch_df, schema_mapping)
    except Exception as error:  # noqa: BLE001
        work_orders = pd.DataFrame()
        warnings.append(f"No fue posible generar ordenes de trabajo: {error}")

    try:
        impact_scores = calculate_impact_scores(
            batch_df,
            schema_mapping,
            work_orders=work_orders,
        )
    except Exception as error:  # noqa: BLE001
        impact_scores = pd.DataFrame()
        warnings.append(f"No fue posible calcular impact scores: {error}")

    try:
        recommendations = generate_strategic_recommendations(
            batch_df,
            schema_mapping,
            work_orders=work_orders,
            impact_scores_df=impact_scores,
        )
    except Exception as error:  # noqa: BLE001
        recommendations = pd.DataFrame()
        warnings.append(f"No fue posible generar recomendaciones estrategicas: {error}")

    try:
        crew_plan = optimize_crews(
            impact_scores,
            available_crews=available_crews,
        )
    except Exception as error:  # noqa: BLE001
        crew_plan = {
            "recommended_zones": pd.DataFrame(),
            "waiting_zones": pd.DataFrame(),
            "coverage_territorial": "Sin datos",
            "riesgo_no_atencion": "Sin datos",
            "explanation": f"No fue posible optimizar cuadrillas: {error}",
        }
        warnings.append(f"No fue posible optimizar cuadrillas: {error}")

    try:
        decision_passports = generate_passports_for_top_zones(
            impact_scores,
            work_orders=work_orders,
            recommendations=recommendations,
            top_n=10,
        )
    except Exception as error:  # noqa: BLE001
        decision_passports = []
        warnings.append(f"No fue posible generar pasaportes de decision: {error}")

    confidence_level = _derive_confidence_level(impact_scores, work_orders)

    return {
        "processed_rows": int(len(batch_df)),
        "work_orders": work_orders,
        "recommendations": recommendations,
        "impact_scores": impact_scores,
        "crew_plan": crew_plan,
        "decision_passports": decision_passports,
        "warnings": warnings,
        "confidence_level": confidence_level,
    }


def build_replay_timeline(replay_results_history: list[dict[str, object]]) -> pd.DataFrame:
    """Construye una linea de tiempo de la simulacion por paso."""
    rows: list[dict[str, object]] = []

    for item in replay_results_history:
        results = item.get("results", {})
        impact_scores = results.get("impact_scores", pd.DataFrame())
        work_orders = results.get("work_orders", pd.DataFrame())

        critical_zones = 0
        high_priority_zones = 0
        top_zone = None
        top_score = None
        if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty:
            critical_zones = int(impact_scores["classification"].eq("Critico").sum())
            high_priority_zones = int(
                impact_scores["classification"].isin(["Critico", "Alto"]).sum()
            )
            top_zone = str(impact_scores.iloc[0]["zona"])
            top_score = float(impact_scores.iloc[0]["final_impact_score"])

        total_connections = 0.0
        total_disconnections = 0.0
        critical_aps_count = 0
        operational_mart = results.get("operational_mart", pd.DataFrame())
        if isinstance(operational_mart, pd.DataFrame) and not operational_mart.empty:
            total_connections = float(pd.to_numeric(operational_mart.get("total_connections"), errors="coerce").fillna(0).sum())
            total_disconnections = float(pd.to_numeric(operational_mart.get("total_disconnections"), errors="coerce").fillna(0).sum())
            if "risk_classification" in operational_mart.columns:
                critical_aps_count = int(operational_mart["risk_classification"].astype(str).isin(["Critico", "Alto"]).sum())

        rows.append(
            {
                "step": int(item.get("step", 0)),
                "processed_rows": int(results.get("processed_rows", 0)),
                "work_orders_count": len(work_orders) if isinstance(work_orders, pd.DataFrame) else 0,
                "critical_zones_count": critical_zones,
                "high_priority_zones_count": high_priority_zones,
                "top_zone": top_zone,
                "top_score": top_score,
                "total_connections": total_connections,
                "total_disconnections": total_disconnections,
                "critical_aps_count": critical_aps_count,
                "confidence_level": results.get("confidence_level", "Baja"),
                "timestamp": item.get("timestamp", get_timestamp()),
            }
        )

    return pd.DataFrame(rows)


def detect_replay_changes(
    previous_results: dict[str, object] | None,
    current_results: dict[str, object],
) -> list[str]:
    """Compara dos pasos consecutivos y resume cambios relevantes."""
    if not previous_results:
        return ["Primer estado procesado de la simulacion."]

    changes: list[str] = []
    previous_orders = previous_results.get("work_orders", pd.DataFrame())
    current_orders = current_results.get("work_orders", pd.DataFrame())
    previous_scores = previous_results.get("impact_scores", pd.DataFrame())
    current_scores = current_results.get("impact_scores", pd.DataFrame())

    if isinstance(previous_orders, pd.DataFrame) and isinstance(current_orders, pd.DataFrame):
        previous_ids = set(previous_orders["id"].tolist()) if "id" in previous_orders.columns else set()
        current_ids = set(current_orders["id"].tolist()) if "id" in current_orders.columns else set()
        new_ids = current_ids - previous_ids
        if new_ids and "id" in current_orders.columns:
            new_orders_df = current_orders[current_orders["id"].isin(new_ids)].head(3)
            for _, row in new_orders_df.iterrows():
                changes.append(f"Nueva orden de trabajo detectada en {row['zona']}.")

    if isinstance(previous_scores, pd.DataFrame) and isinstance(current_scores, pd.DataFrame):
        previous_priority_map = (
            previous_scores.set_index("zona")["classification"].to_dict()
            if not previous_scores.empty
            else {}
        )
        current_priority_map = (
            current_scores.set_index("zona")["classification"].to_dict()
            if not current_scores.empty
            else {}
        )

        order_rank = {"Observacion": 0, "Bajo": 1, "Medio": 2, "Alto": 3, "Critico": 4}
        for zone, current_level in current_priority_map.items():
            previous_level = previous_priority_map.get(zone)
            if previous_level is None or previous_level == current_level:
                continue

            if order_rank.get(str(current_level), -1) > order_rank.get(str(previous_level), -1):
                changes.append(f"Zona {zone} subio de prioridad {previous_level} a {current_level}.")
            elif previous_level == "Critico" and current_level != "Critico":
                changes.append(f"Zona {zone} dejo de estar en estado critico.")

        if not previous_scores.empty and not current_scores.empty:
            previous_top = str(previous_scores.iloc[0]["zona"])
            current_top = str(current_scores.iloc[0]["zona"])
            if previous_top != current_top:
                changes.append(
                    f"Cambio la zona mas critica: antes {previous_top}, ahora {current_top}."
                )

    return changes or ["No se detectaron cambios relevantes frente al paso anterior."]


def summarize_replay_state(current_results: dict[str, object]) -> dict[str, object]:
    """Resume el estado actual de la simulacion operativa."""
    impact_scores = current_results.get("impact_scores", pd.DataFrame())
    work_orders = current_results.get("work_orders", pd.DataFrame())

    top_zone = None
    action_suggestion = "Seguir monitoreando el siguiente lote."
    critical_zones = 0
    if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty:
        top_zone = str(impact_scores.iloc[0]["zona"])
        top_classification = str(impact_scores.iloc[0]["classification"])
        critical_zones = int(impact_scores["classification"].eq("Critico").sum())
        if top_classification in {"Critico", "Alto"}:
            action_suggestion = f"Priorizar revision operativa de {top_zone}."

    return {
        "filas_procesadas": int(current_results.get("processed_rows", 0)),
        "numero_ordenes": len(work_orders) if isinstance(work_orders, pd.DataFrame) else 0,
        "numero_zonas_criticas": critical_zones,
        "zona_mas_critica": top_zone,
        "accion_sugerida": action_suggestion,
        "nivel_confianza": current_results.get("confidence_level", "Baja"),
    }


def get_total_replay_steps(events_df: pd.DataFrame, batch_size: int) -> int:
    """Calcula cuantos lotes requiere una simulacion completa."""
    if events_df.empty:
        return 0
    safe_batch_size = max(int(batch_size), 1)
    return int(ceil(len(events_df) / safe_batch_size))
