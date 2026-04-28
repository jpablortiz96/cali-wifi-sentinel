from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import streamlit as st


STANDARD_RESULT_KEYS = [
    "work_orders",
    "impact_scores",
    "crew_plan",
    "decision_passports",
    "recommendations",
    "citizen_experience_scores",
    "citizen_recommendations",
    "citizen_feedback",
    "citizen_feedback_summary",
    "digital_equity_proxy",
    "citizen_insights_markdown",
    "socioeconomic_validation",
    "social_roi_scores",
    "social_roi_recommendations",
    "social_roi_explanation_markdown",
    "quality_gate_report",
    "replay_timeline",
    "human_review_log",
    "audit_log",
    "readiness",
    "weather_context",
    "osm_context",
    "calendar_context",
    "confidence_level",
    "trace_id",
    "operational_mart",
    "meraki_anomalies",
    "wifi_package_summary",
    "is_meraki_mode",
]


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper la app."""
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


def _last_session_value(exact_keys: list[str] | None = None, prefixes: list[str] | None = None) -> object | None:
    """Busca la última coincidencia útil en session_state."""
    exact_keys = exact_keys or []
    prefixes = prefixes or []

    for key in exact_keys:
        if key in st.session_state:
            return st.session_state[key]

    matching_keys = [
        key
        for key in st.session_state.keys()
        if any(str(key).startswith(prefix) for prefix in prefixes)
    ]
    if not matching_keys:
        return None

    return st.session_state[matching_keys[-1]]


def _combine_audit_logs(*logs: list[dict[str, object]] | None) -> list[dict[str, object]]:
    """Combina logs sin duplicar eventos por audit_id."""
    combined: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for log in logs:
        if not log:
            continue
        for event in log:
            if not isinstance(event, dict):
                continue
            audit_id = str(event.get("audit_id", "")).strip()
            if audit_id and audit_id in seen_ids:
                continue
            if audit_id:
                seen_ids.add(audit_id)
            combined.append(event)

    combined.sort(key=lambda item: str(item.get("timestamp", "")))
    return combined


def _pick_first_non_none(*values: object) -> object | None:
    """Devuelve el primer valor no nulo sin evaluar truthiness de DataFrames."""
    for value in values:
        if value is not None:
            return value
    return None


def normalize_results_for_dashboard(results: dict[str, object] | None) -> dict[str, object]:
    """Garantiza una estructura estándar para la Vista Ejecutiva 360."""
    payload = copy.deepcopy(results or {})
    normalized: dict[str, object] = {
        "work_orders": pd.DataFrame(),
        "impact_scores": pd.DataFrame(),
        "crew_plan": {},
        "decision_passports": [],
        "recommendations": pd.DataFrame(),
        "citizen_experience_scores": pd.DataFrame(),
        "citizen_recommendations": pd.DataFrame(),
        "citizen_feedback": pd.DataFrame(),
        "citizen_feedback_summary": {},
        "digital_equity_proxy": pd.DataFrame(),
        "citizen_insights_markdown": "",
        "socioeconomic_validation": {},
        "social_roi_scores": pd.DataFrame(),
        "social_roi_recommendations": pd.DataFrame(),
        "social_roi_explanation_markdown": "",
        "quality_gate_report": {},
        "replay_timeline": pd.DataFrame(),
        "human_review_log": pd.DataFrame(),
        "audit_log": [],
        "readiness": {},
        "weather_context": pd.DataFrame(),
        "osm_context": pd.DataFrame(),
        "calendar_context": pd.DataFrame(),
        "confidence_level": "Baja",
        "trace_id": None,
        "limitations": [],
        "source": payload.get("source"),
        "operational_mart": pd.DataFrame(),
        "meraki_anomalies": pd.DataFrame(),
        "wifi_package_summary": {},
        "is_meraki_mode": False,
    }

    if not payload:
        return normalized

    # Replay state guarda resultados actuales dentro de current_results.
    if isinstance(payload.get("current_results"), dict):
        merged_payload = dict(payload["current_results"])
        merged_payload["trace_id"] = payload.get("trace_id") or merged_payload.get("trace_id")
        merged_payload["replay_timeline"] = payload.get("timeline", pd.DataFrame())
        merged_payload["audit_log"] = payload.get("audit_log", merged_payload.get("audit_log", []))
        merged_payload["audit_summary"] = payload.get("audit_summary", merged_payload.get("audit_summary", {}))
        merged_payload["warnings"] = payload.get("warnings", merged_payload.get("warnings", []))
        payload = merged_payload

    normalized["work_orders"] = _safe_dataframe(payload.get("work_orders"))
    normalized["impact_scores"] = _safe_dataframe(payload.get("impact_scores"))
    normalized["crew_plan"] = payload.get("crew_plan", {}) if isinstance(payload.get("crew_plan"), dict) else {}
    normalized["decision_passports"] = (
        payload.get("decision_passports", [])
        if isinstance(payload.get("decision_passports"), list)
        else []
    )
    normalized["recommendations"] = _safe_dataframe(payload.get("recommendations"))
    normalized["citizen_experience_scores"] = _safe_dataframe(payload.get("citizen_experience_scores"))
    normalized["citizen_recommendations"] = _safe_dataframe(payload.get("citizen_recommendations"))
    normalized["citizen_feedback"] = _safe_dataframe(payload.get("citizen_feedback"))
    normalized["citizen_feedback_summary"] = (
        payload.get("citizen_feedback_summary", {})
        if isinstance(payload.get("citizen_feedback_summary"), dict)
        else {}
    )
    normalized["digital_equity_proxy"] = _safe_dataframe(payload.get("digital_equity_proxy"))
    normalized["citizen_insights_markdown"] = str(payload.get("citizen_insights_markdown", "") or "")
    normalized["socioeconomic_validation"] = (
        payload.get("socioeconomic_validation", {})
        if isinstance(payload.get("socioeconomic_validation"), dict)
        else {}
    )
    normalized["social_roi_scores"] = _safe_dataframe(payload.get("social_roi_scores"))
    normalized["social_roi_recommendations"] = _safe_dataframe(payload.get("social_roi_recommendations"))
    normalized["social_roi_explanation_markdown"] = str(payload.get("social_roi_explanation_markdown", "") or "")
    normalized["quality_gate_report"] = payload.get("quality_gate_report", {}) if isinstance(payload.get("quality_gate_report"), dict) else {}
    replay_timeline_value = _pick_first_non_none(
        payload.get("replay_timeline"),
        payload.get("timeline"),
    )
    normalized["replay_timeline"] = _safe_dataframe(replay_timeline_value)
    normalized["human_review_log"] = _safe_dataframe(payload.get("human_review_log"))
    normalized["audit_log"] = payload.get("operational_audit_log") or payload.get("audit_log") or []
    normalized["readiness"] = payload.get("readiness", {}) if isinstance(payload.get("readiness"), dict) else {}
    normalized["weather_context"] = _safe_dataframe(payload.get("weather_context"))
    normalized["osm_context"] = _safe_dataframe(payload.get("osm_context"))
    normalized["calendar_context"] = _safe_dataframe(payload.get("calendar_context"))
    normalized["confidence_level"] = str(payload.get("confidence_level", "Baja"))
    normalized["trace_id"] = payload.get("trace_id")
    normalized["limitations"] = payload.get("limitations", []) if isinstance(payload.get("limitations"), list) else []
    normalized["source"] = payload.get("source", normalized["source"])
    normalized["operational_mart"] = _safe_dataframe(payload.get("operational_mart"))
    normalized["meraki_anomalies"] = _safe_dataframe(payload.get("meraki_anomalies"))
    normalized["wifi_package_summary"] = payload.get("wifi_package_summary", {}) if isinstance(payload.get("wifi_package_summary"), dict) else {}
    normalized["is_meraki_mode"] = bool(payload.get("is_meraki_mode"))

    return normalized


def get_latest_operational_results() -> dict[str, object]:
    """Busca resultados recientes del sistema sin depender de una sola clave."""
    available_keys = [str(key) for key in st.session_state.keys()]
    current_dataset_signature = st.session_state.get("current_dataset_signature")

    latest_saved = st.session_state.get("latest_operational_results")
    if isinstance(latest_saved, dict):
        saved_signature = latest_saved.get("dataset_signature")
        if not current_dataset_signature or not saved_signature or saved_signature == current_dataset_signature:
            normalized = normalize_results_for_dashboard(latest_saved.get("results"))
            normalized["source"] = latest_saved.get("source", normalized.get("source"))
            return {
                "source": latest_saved.get("source", "mixed"),
                "results": normalized,
                "has_results": any(
                    not _safe_dataframe(normalized.get(key)).empty
                    if key in {"work_orders", "impact_scores", "recommendations", "replay_timeline", "human_review_log"}
                    else bool(normalized.get(key))
                    for key in STANDARD_RESULT_KEYS
                ),
                "available_keys": available_keys,
            }

    cycle_results = _last_session_value(
        exact_keys=[
            "autonomous_cycle_results",
            "mission_control_results",
            "orchestrator_results",
            "cycle_results",
        ],
        prefixes=["cycle_results::"],
    )
    replay_results = _last_session_value(
        exact_keys=["replay_results", "latest_replay_results"],
        prefixes=["replay_state::"],
    )

    source = "none"
    assembled: dict[str, object] = {}
    if isinstance(cycle_results, dict):
        assembled.update(cycle_results)
        source = "mission_control"
    if isinstance(replay_results, dict):
        replay_payload = normalize_results_for_dashboard(replay_results)
        assembled.update(replay_payload)
        source = "replay" if source == "none" else "mixed"

    partial_keys = {
        "work_orders": _last_session_value(exact_keys=["work_orders"]),
        "impact_scores": _last_session_value(exact_keys=["impact_scores"]),
        "decision_passports": _last_session_value(exact_keys=["decision_passports"]),
        "crew_plan": _last_session_value(exact_keys=["crew_plan"]),
        "recommendations": _last_session_value(exact_keys=["recommendations"]),
        "quality_gate_report": _last_session_value(
            exact_keys=["quality_gate_report"],
            prefixes=["quality_gate::"],
        ),
        "operational_audit_log": _last_session_value(
            exact_keys=["operational_audit_log"],
            prefixes=["manual_audit::"],
        ),
    }
    for key, value in partial_keys.items():
        if value is not None and key not in assembled:
            assembled[key] = value
            source = "mixed" if source != "none" else "mixed"

    normalized = normalize_results_for_dashboard(assembled)
    has_results = any(
        [
            not normalized["work_orders"].empty,
            not normalized["impact_scores"].empty,
            bool(normalized["decision_passports"]),
            bool(normalized["crew_plan"]),
            not normalized["recommendations"].empty,
            not normalized["replay_timeline"].empty,
        ]
    )

    return {
        "source": source,
        "results": normalized,
        "has_results": has_results,
        "available_keys": available_keys,
    }


def save_latest_operational_results(results: dict[str, object], source: str = "mission_control") -> None:
    """Guarda un snapshot unificado para la Vista Ejecutiva 360."""
    st.session_state["latest_operational_results"] = {
        "source": source,
        "results": normalize_results_for_dashboard(results),
        "dataset_signature": st.session_state.get("current_dataset_signature"),
    }


def get_dashboard_context(
    df: pd.DataFrame | None = None,
    schema_mapping: dict[str, str | None] | None = None,
) -> dict[str, object]:
    """Une dataset, resultados, revisión humana y auditoría en un contexto único."""
    latest = get_latest_operational_results()
    normalized = normalize_results_for_dashboard(latest.get("results"))

    review_queue = _last_session_value(
        exact_keys=["human_review_log"],
        prefixes=["review_queue::"],
    )
    manual_audit = _last_session_value(
        exact_keys=["operational_audit_log"],
        prefixes=["manual_audit::"],
    )
    quality_gate = _last_session_value(
        exact_keys=["quality_gate_report"],
        prefixes=["quality_gate::"],
    )
    current_dataset_signature = st.session_state.get("current_dataset_signature")
    strategic_payload = st.session_state.get("gemini_strategic_recommendations_payload", {})
    manual_weather_payload = st.session_state.get("latest_manual_weather_context", {})
    manual_osm_payload = st.session_state.get("latest_manual_osm_context", {})

    if review_queue is not None:
        normalized["human_review_log"] = _safe_dataframe(review_queue)
    if isinstance(quality_gate, dict) and quality_gate:
        normalized["quality_gate_report"] = quality_gate
    if isinstance(strategic_payload, dict) and strategic_payload.get("dataset_signature") == current_dataset_signature:
        normalized["recommendations"] = _safe_dataframe(strategic_payload.get("data"))
    if isinstance(manual_weather_payload, dict) and manual_weather_payload.get("dataset_signature") == current_dataset_signature:
        normalized["weather_context"] = _safe_dataframe(manual_weather_payload.get("data"))
    if isinstance(manual_osm_payload, dict) and manual_osm_payload.get("dataset_signature") == current_dataset_signature:
        normalized["osm_context"] = _safe_dataframe(manual_osm_payload.get("data"))
    normalized["audit_log"] = _combine_audit_logs(
        normalized.get("audit_log", []),
        manual_audit if isinstance(manual_audit, list) else [],
    )

    has_coordinates = False
    if isinstance(df, pd.DataFrame) and schema_mapping:
        lat_col = schema_mapping.get("latitude_col")
        lon_col = schema_mapping.get("longitude_col")
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            lat_values = pd.to_numeric(df[lat_col], errors="coerce")
            lon_values = pd.to_numeric(df[lon_col], errors="coerce")
            has_coordinates = bool((lat_values.notna() & lon_values.notna()).any())

    return {
        "source": latest.get("source", "none"),
        "results": normalized,
        "has_results": latest.get("has_results", False),
        "available_keys": latest.get("available_keys", []),
        "dataframe": df,
        "schema_mapping": schema_mapping or {},
        "has_coordinates": has_coordinates,
    }
