from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent_orchestrator import build_executive_summary, run_autonomous_cycle
from src.app_state import (
    get_dashboard_context,
    normalize_results_for_dashboard,
    save_latest_operational_results,
)
from src.config import DATA_EXTERNAL_CACHE_DIR, PROJECT_ROOT, PROMPTS_DIR
from src.calendar_public_context import enrich_hourly_with_public_calendar, summarize_calendar_impact
from src.citizen_feedback import load_citizen_feedback, save_citizen_feedback, summarize_citizen_feedback
from src.citizen_insights_agent import (
    build_citizen_insights_context,
    fallback_citizen_insights,
    generate_citizen_insights_with_gemini,
)
from src.citizen_metrics import (
    build_citizen_zone_summary,
    calculate_citizen_experience_score,
    calculate_hourly_citizen_patterns,
)
from src.citizen_recommender import build_citizen_alerts, recommend_best_wifi_zones
from src.data_loader import (
    DataLoaderError,
    list_local_datasets,
    load_local_dataset,
    load_tabular_file,
    save_uploaded_file,
)
from src.data_quality import (
    build_dataset_profile,
    build_quality_summary,
    get_column_types,
    get_dataframe_summary,
    get_null_counts,
    profile_to_text,
)
from src.dashboard_insights import (
    build_executive_dashboard_summary,
    build_next_best_actions,
    build_risk_alerts,
    build_top_findings,
)
from src.dashboard_visuals import (
    create_calendar_heatmap,
    create_classification_donut,
    create_impact_scatter,
    create_kpi_cards_data,
    create_priority_bar_chart,
    create_recommendations_treemap,
    create_replay_timeline_chart,
    create_score_component_radar,
    create_territory_heatmap,
    create_work_order_status_chart,
)
from src.decision_passport import generate_passports_for_top_zones
from src.demo_data import generate_synthetic_wifi_data
from src.digital_equity import calculate_digital_equity_proxy
from src.evidence_pack import (
    build_evidence_summary_table,
    build_readable_evidence_report,
    create_evidence_files,
    dataframe_to_csv_bytes,
    dict_to_json_bytes,
)
from src.evidence_formatters import (
    format_audit_log_for_display,
    format_crew_plan_for_display,
    format_impact_scores_for_display,
    format_passports_for_display,
    format_quality_gate_for_display,
    format_recommendations_for_display,
    format_work_orders_for_display,
    safe_to_dataframe,
)
from src.executive_insights_agent import get_or_generate_dashboard_insights
from src.geo_visuals import create_cali_priority_map_pro
from src.gemini_client import generate_gemini_text, is_gemini_configured, load_gemini_config
from src.human_in_the_loop import (
    bulk_update_work_orders,
    create_review_queue,
    export_review_log,
    summarize_human_review,
    update_work_order_status,
)
from src.impact_scoring import calculate_impact_scores
from src.live_replay import (
    build_replay_timeline,
    detect_replay_changes,
    get_replay_batch,
    get_total_replay_steps,
    prepare_replay_events,
    run_replay_analysis,
    summarize_replay_state,
)
from src.meraki_features import build_operational_mart
from src.meraki_schema import build_meraki_schema_mapping
from src.operational_audit import (
    append_audit_event,
    audit_log_to_dataframe,
    build_operational_audit_summary,
    create_audit_event,
)
from src.osm_context import enrich_osm_context
from src.platform_agent import answer_platform_question, build_platform_agent_context
from src.profile_storage import save_profile_json
from src.readiness_score import calculate_data_readiness
from src.resource_optimizer import optimize_crews
from src.schema_mapper import (
    MAPPING_FIELD_CONFIG,
    build_schema_mapping,
    get_module_readiness,
    suggest_schema_mapping,
)
from src.social_roi import (
    calculate_social_roi_score,
    generate_social_infrastructure_recommendations,
    prepare_social_roi_inputs,
)
from src.social_roi_agent import (
    build_social_roi_context,
    fallback_social_roi_explanation,
    generate_social_roi_explanation_with_gemini,
)
from src.socioeconomic_connectors import (
    fetch_socrata_dataset,
    get_dane_ipm_metadata,
    get_dane_nbi_metadata,
    get_sisben_open_data_metadata,
)
from src.socioeconomic_sources import (
    detect_socioeconomic_geo_level,
    load_socioeconomic_file,
    normalize_socioeconomic_columns,
    validate_socioeconomic_dataset,
)
from src.strategic_recommendation_agent import get_or_generate_strategic_recommendations
from src.strategic_recommendations import generate_strategic_recommendations
from src.technical_chat import answer_technical_question, build_orchestrated_context
from src.ui_components import (
    inject_back_to_top_button,
    inject_premium_css,
    render_dataframe_clean,
    render_dashboard_empty_state,
    render_download_buttons_for_evidence,
    render_empty_state,
    render_action_card,
    render_compact_badge,
    render_floating_chat_widget,
    render_insight_card,
    render_json_advanced,
    render_metric_row,
    render_premium_kpi_card,
    render_section_header,
    render_status_badge,
)
from src.utils import get_timestamp
from src.validation_suite import build_quality_gate_report
from src.weather_context import enrich_weather_context
from src.wifi_package_loader import (
    detect_official_wifi_package,
    get_package_summary,
    load_wifi_package_from_folder,
    load_wifi_package_from_github,
)
from src.work_orders import generate_work_orders
from src.user_portals import (
    CITIZEN_PROFILE,
    TECHNICAL_PROFILE,
    get_citizen_sections,
    get_default_section_for_profile,
    get_technical_sections,
    render_profile_selector,
)


st.set_page_config(page_title="Cali WiFi Sentinel 360", layout="wide")
inject_premium_css()
st.markdown("<div id='top-anchor'></div>", unsafe_allow_html=True)
inject_back_to_top_button()


APP_TITLE = "Cali WiFi Sentinel 360"
APP_SUBTITLE = "Centro de comando ejecutivo para operación, priorización y evidencia de la red WiFi pública"
INSPECTION_DESCRIPTION = (
    "Inspecciona, mapea, prioriza y monitorea la red WiFi pública con una experiencia operativa, ejecutiva y auditable."
)
SYNTHETIC_WARNING_TITLE = "DATOS SINTÉTICOS / NO OFICIALES"
SYNTHETIC_WARNING_BODY = (
    "Estos datos son sintéticos y solo sirven para probar la app. "
    "No representan datos oficiales de la Alcaldía ni del evento."
)
REPLAY_NOTE = (
    "Simulación Operativa no significa monitoreo en tiempo real real. "
    "Es una reproducción controlada del dataset cargado para demostrar cómo el sistema "
    "procesaría registros entrantes en un entorno operativo."
)

RECOMMENDED_FLOW_TEXT = (
    "Flujo recomendado: 1) Carga datos -> 2) Mapea columnas -> 3) Ejecuta Mission Control o Simulacion "
    "-> 4) Revisa Vista Ejecutiva 360 -> 5) Exporta evidencia."
)


def show_initial_instructions() -> None:
    """Guía inicial cuando aún no existe un dataset activo."""
    st.info(
        "Carga un archivo CSV, TXT o Excel, selecciona un dataset local de `data/raw/` o activa el paquete oficial "
        "Zonas WiFi Inteligentes para probar el flujo completo."
    )
    st.markdown(
        """
        **Ruta recomendada**
        1. Carga o selecciona un dataset.
        2. Si usas el paquete oficial, cárgalo y construye el mart operativo.
        3. Revisa la estructura y el perfil del archivo.
        4. Mapea manualmente las columnas reales o aplica el esquema Meraki.
        5. Ejecuta Mission Control o la Simulación Operativa.
        6. Revisa órdenes, impacto, cuadrillas, validación humana y auditoría.
        """
    )


def build_tab_visibility_css(all_sections: list[str], visible_sections: list[str]) -> str:
    """Oculta pestañas no relevantes según el perfil activo sin romper la lógica existente."""
    hidden_indexes = [
        index + 1
        for index, section_name in enumerate(all_sections)
        if section_name not in visible_sections
    ]
    if not hidden_indexes:
        return ""

    rules = []
    for index in hidden_indexes:
        rules.append(
            f".stTabs [data-baseweb=\"tab-list\"] button:nth-child({index})"
            "{display:none !important;}"
        )

    return "<style>" + "".join(rules) + "</style>"


def is_synthetic_dataset(dataframe: pd.DataFrame | None) -> bool:
    """Detecta si el dataset activo corresponde a la demo sintética."""
    if dataframe is None or dataframe.empty or "tipo_dato" not in dataframe.columns:
        return False
    return bool(dataframe["tipo_dato"].astype(str).eq("SINTETICO_NO_OFICIAL").all())


def build_file_signature(file_bytes: bytes) -> str:
    """Genera una huella estable para aislar estado por dataset."""
    return hashlib.md5(file_bytes).hexdigest()


def build_mapping_signature(schema_mapping: dict[str, str | None]) -> str:
    """Genera una huella del mapeo actual."""
    payload = json.dumps(schema_mapping, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_state_key(prefix: str, file_signature: str, mapping_signature: str | None = None) -> str:
    """Crea claves únicas de session_state por dataset y mapeo."""
    if mapping_signature:
        return f"{prefix}::{file_signature}::{mapping_signature}"
    return f"{prefix}::{file_signature}"


def save_file_once_per_session(file_name: str, file_bytes: bytes, file_signature: str) -> Path:
    """Evita guardar varias veces el mismo upload manual."""
    last_signature = st.session_state.get("last_uploaded_signature")
    if last_signature != file_signature:
        saved_path = save_uploaded_file(file_name, file_bytes)
        st.session_state["last_uploaded_signature"] = file_signature
        st.session_state["last_saved_path"] = str(saved_path)

    return Path(st.session_state["last_saved_path"])


def save_profile_once_per_session(
    profile: dict[str, object],
    original_filename: str,
    file_signature: str,
) -> Path:
    """Guarda el perfil JSON una sola vez por dataset."""
    profile_signature = f"profile::{file_signature}"
    if st.session_state.get("last_profile_signature") != profile_signature:
        saved_path = save_profile_json(profile, original_filename)
        st.session_state["last_profile_signature"] = profile_signature
        st.session_state["last_profile_path"] = str(saved_path)

    return Path(st.session_state["last_profile_path"])


def build_candidate_table(profile: dict[str, object]) -> pd.DataFrame:
    """Convierte sugerencias heurísticas del perfil a tabla legible."""
    labels = {
        "fecha": "Fecha",
        "zona": "Zona",
        "uso_conectividad": "Uso/conectividad",
        "ubicacion_geografica": "Ubicación geográfica",
        "territorio": "Comuna/barrio/territorio",
    }
    candidates = profile.get("candidate_columns", {})
    rows = []
    for key, label in labels.items():
        values = candidates.get(key, []) if isinstance(candidates, dict) else []
        rows.append(
            {
                "categoria": label,
                "columnas_candidatas": ", ".join(values) if values else "Sin sugerencias",
            }
        )
    return pd.DataFrame(rows)


def load_data_inspector_prompt() -> str:
    """Lee el prompt base para análisis estructural."""
    prompt_path = PROMPTS_DIR / "data_inspector.md"
    return prompt_path.read_text(encoding="utf-8")


def build_final_gemini_prompt(profile: dict[str, object]) -> str:
    """Combina prompt base y perfil resumido del dataset."""
    return (
        f"{load_data_inspector_prompt()}\n\n"
        "Analiza únicamente el siguiente perfil estructural resumido. "
        "No se envió la base completa.\n\n"
        f"{profile_to_text(profile)}"
    )


def initialize_mapping_state(
    file_signature: str,
    suggested_mapping: dict[str, str | None],
) -> None:
    """Inicializa selectores de mapeo cuando cambia el dataset."""
    signature_key = "mapping_state_file_signature"
    if st.session_state.get(signature_key) == file_signature:
        return

    for field_key in MAPPING_FIELD_CONFIG:
        widget_key = f"mapping_widget_{field_key}"
        st.session_state[widget_key] = suggested_mapping.get(field_key) or "Sin seleccionar"

    st.session_state[signature_key] = file_signature


def build_mapping_summary_table(schema_mapping: dict[str, str | None]) -> pd.DataFrame:
    """Resume el mapeo seleccionado."""
    rows = []
    for field_key, config in MAPPING_FIELD_CONFIG.items():
        rows.append(
            {
                "campo_logico": field_key,
                "etiqueta": config["label"],
                "columna_seleccionada": schema_mapping.get(field_key) or "Sin seleccionar",
            }
        )
    return pd.DataFrame(rows)


def build_readiness_table(readiness: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Resume el estado de cada agente oficial."""
    return pd.DataFrame(
        [
            {
                "modulo": "Agente Operativo",
                "estado": "Listo" if readiness["operational"]["ready"] else "Incompleto",
                "detalle": ", ".join(readiness["operational"]["missing"]) or "Cumple mínimo",
            },
            {
                "modulo": "Agente Conversacional",
                "estado": "Listo" if readiness["conversational"]["ready"] else "Incompleto",
                "detalle": ", ".join(readiness["conversational"]["missing"]) or "Cumple mínimo",
            },
            {
                "modulo": "Agente Estratégico",
                "estado": "Listo" if readiness["strategic"]["ready"] else "Incompleto",
                "detalle": ", ".join(readiness["strategic"]["missing"]) or "Cumple mínimo",
            },
        ]
    )


def build_readiness_checks_table(readiness_result: dict[str, Any]) -> pd.DataFrame:
    """Convierte checks del readiness score en tabla."""
    return pd.DataFrame(readiness_result.get("checks", []))


def build_readiness_alignment_table(readiness_result: dict[str, Any]) -> pd.DataFrame:
    """Resume la alineación con los tres agentes oficiales."""
    alignment = readiness_result.get("official_challenge_alignment", {})
    return pd.DataFrame(
        [
            {"agente": "Agente Operativo", "estado": alignment.get("agente_operativo", "Sin evaluar")},
            {"agente": "Agente Conversacional", "estado": alignment.get("agente_conversacional", "Sin evaluar")},
            {"agente": "Agente Estratégico", "estado": alignment.get("agente_estrategico", "Sin evaluar")},
        ]
    )


def build_anomalies_table(work_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae una vista compacta de órdenes y anomalías."""
    if work_orders_df is None or work_orders_df.empty:
        return pd.DataFrame(
            columns=[
                "ap_name",
                "zona",
                "zone_name",
                "tipo_alerta",
                "evidencia",
                "prioridad",
                "nivel_confianza",
                "classification",
                "final_impact_score",
            ]
        )

    selected_columns = [
        column_name
        for column_name in [
            "ap_name",
            "zona",
            "zone_name",
            "tipo_alerta",
            "evidencia",
            "prioridad",
            "nivel_confianza",
            "classification",
            "final_impact_score",
            "decision_passport_id",
        ]
        if column_name in work_orders_df.columns
    ]
    return work_orders_df[selected_columns].copy()


def build_map_dataframe(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    impact_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepara una tabla simple para st.map."""
    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    zone_col = schema_mapping.get("zone_col")

    if not latitude_col or not longitude_col:
        return pd.DataFrame(columns=["lat", "lon"])

    map_df = pd.DataFrame(
        {
            "lat": pd.to_numeric(dataframe[latitude_col], errors="coerce"),
            "lon": pd.to_numeric(dataframe[longitude_col], errors="coerce"),
        }
    )

    if zone_col:
        map_df["zona"] = dataframe[zone_col].astype(str)
        map_df = (
            map_df.dropna(subset=["lat", "lon"])
            .groupby("zona", dropna=False)[["lat", "lon"]]
            .mean()
            .reset_index()
        )
    else:
        map_df = map_df.dropna(subset=["lat", "lon"])

    if impact_scores_df is not None and not impact_scores_df.empty and "zona" in map_df.columns:
        score_columns = [
            column_name
            for column_name in ["zona", "final_impact_score", "classification", "territorio"]
            if column_name in impact_scores_df.columns
        ]
        map_df = map_df.merge(
            impact_scores_df[score_columns].drop_duplicates(subset=["zona"]),
            on="zona",
            how="left",
        )

    return map_df


def build_territory_ranking(
    impact_scores_df: pd.DataFrame,
    work_orders_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye un ranking por territorio cuando no hay coordenadas."""
    if impact_scores_df is None or impact_scores_df.empty or "territorio" not in impact_scores_df.columns:
        return pd.DataFrame()

    territory_df = impact_scores_df.dropna(subset=["territorio"]).copy()
    if territory_df.empty:
        return pd.DataFrame()

    ranking_df = territory_df.groupby("territorio", dropna=False).agg(
        zonas=("zona", "nunique"),
        score_promedio=("final_impact_score", "mean"),
        score_maximo=("final_impact_score", "max"),
        zonas_criticas=("classification", lambda values: int(values.isin(["Critico", "Alto"]).sum())),
    )

    if work_orders_df is not None and not work_orders_df.empty:
        order_counts = work_orders_df.merge(
            territory_df[["zona", "territorio"]].drop_duplicates(),
            on="zona",
            how="left",
        )["territorio"].value_counts()
        ranking_df["ordenes_preliminares"] = ranking_df.index.map(order_counts).fillna(0).astype(int)

    ranking_df = ranking_df.reset_index()
    ranking_df["score_promedio"] = ranking_df["score_promedio"].round(2)
    ranking_df["score_maximo"] = ranking_df["score_maximo"].round(2)
    ranking_df = ranking_df.sort_values(
        by=["zonas_criticas", "score_promedio", "zonas"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranking_df


def _mode_or_first(series: pd.Series) -> str:
    """Obtiene la moda simple o el primer valor no nulo."""
    clean_series = series.dropna().astype(str)
    if clean_series.empty:
        return ""
    mode_values = clean_series.mode()
    if not mode_values.empty:
        return str(mode_values.iloc[0])
    return str(clean_series.iloc[0])


def build_generic_citizen_operational_mart(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    active_results: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Crea un proxy de mart operativo para la capa ciudadana en modo genérico."""
    zone_col = schema_mapping.get("zone_col")
    if dataframe is None or dataframe.empty or not zone_col or zone_col not in dataframe.columns:
        return pd.DataFrame()

    working_df = pd.DataFrame(
        {
            "ap_name": dataframe[zone_col].astype(str),
            "zone_name": dataframe[schema_mapping.get("territory_col")].astype(str)
            if schema_mapping.get("territory_col") and schema_mapping.get("territory_col") in dataframe.columns
            else dataframe[zone_col].astype(str),
        }
    )

    if schema_mapping.get("status_col") and schema_mapping["status_col"] in dataframe.columns:
        working_df["status"] = dataframe[schema_mapping["status_col"]].astype(str)
    else:
        working_df["status"] = "unknown"

    if schema_mapping.get("connections_col") and schema_mapping["connections_col"] in dataframe.columns:
        working_df["connections_value"] = pd.to_numeric(dataframe[schema_mapping["connections_col"]], errors="coerce").fillna(0)
    else:
        working_df["connections_value"] = 0.0

    if schema_mapping.get("traffic_col") and schema_mapping["traffic_col"] in dataframe.columns:
        working_df["traffic_value"] = pd.to_numeric(dataframe[schema_mapping["traffic_col"]], errors="coerce").fillna(0)
    else:
        working_df["traffic_value"] = 0.0

    if schema_mapping.get("date_col") and schema_mapping["date_col"] in dataframe.columns:
        parsed_dates = pd.to_datetime(dataframe[schema_mapping["date_col"]], errors="coerce")
        working_df["date_value"] = parsed_dates
    else:
        working_df["date_value"] = pd.NaT

    aggregated = (
        working_df.groupby(["ap_name", "zone_name"], dropna=False)
        .agg(
            status=("status", _mode_or_first),
            total_connections=("connections_value", "sum"),
            usage_mb_total=("traffic_value", "sum"),
            clients_reported=("connections_value", "mean"),
            active_hours=("date_value", lambda values: int(pd.Series(values).dropna().nunique()) if not pd.Series(values).dropna().empty else len(values)),
            observations=("ap_name", "count"),
        )
        .reset_index()
    )
    aggregated["clients_reported"] = pd.to_numeric(aggregated["clients_reported"], errors="coerce").fillna(0).round(2)
    aggregated["avg_disconnection_rate"] = 0.0
    aggregated["evidence_level"] = (
        pd.to_numeric(aggregated["observations"], errors="coerce").fillna(0).clip(lower=1).rank(pct=True) * 100
    ).round(2)

    impact_scores = (
        active_results.get("impact_scores", pd.DataFrame())
        if isinstance(active_results, dict)
        else pd.DataFrame()
    )
    if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty and "zona" in impact_scores.columns:
        merge_columns = [column for column in ["zona", "final_impact_score"] if column in impact_scores.columns]
        aggregated = aggregated.merge(
            impact_scores[merge_columns].drop_duplicates(subset=["zona"]),
            left_on="ap_name",
            right_on="zona",
            how="left",
        )
        aggregated["operational_risk_score"] = pd.to_numeric(aggregated.get("final_impact_score"), errors="coerce").fillna(0)
        aggregated = aggregated.drop(columns=["zona"], errors="ignore")
    else:
        aggregated["operational_risk_score"] = 0.0

    aggregated["ap_health_score"] = (100 - aggregated["operational_risk_score"]).clip(lower=0, upper=100)
    aggregated["limitations"] = (
        "Modo genérico: el mart ciudadano se deriva de columnas mapeadas manualmente. "
        "No representa telemetría horaria nativa tipo Meraki."
    )
    return aggregated


def build_citizen_bundle(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    active_results: dict[str, object],
    wifi_package: dict[str, object] | None = None,
    calendar_context_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Calcula artefactos de experiencia ciudadana a partir de resultados actuales."""
    meraki_mart = active_results.get("operational_mart", pd.DataFrame()) if isinstance(active_results, dict) else pd.DataFrame()
    operational_mart = meraki_mart if isinstance(meraki_mart, pd.DataFrame) and not meraki_mart.empty else build_generic_citizen_operational_mart(
        dataframe,
        schema_mapping,
        active_results=active_results,
    )
    is_meraki_mode = isinstance(active_results, dict) and bool(active_results.get("is_meraki_mode"))
    hourly_metrics = (
        wifi_package.get("hourly_metrics", pd.DataFrame())
        if is_meraki_mode and isinstance(wifi_package, dict)
        else pd.DataFrame()
    )
    clients_df = (
        wifi_package.get("clients", pd.DataFrame())
        if is_meraki_mode and isinstance(wifi_package, dict)
        else pd.DataFrame()
    )
    osm_context_df = active_results.get("osm_context", pd.DataFrame()) if isinstance(active_results, dict) else pd.DataFrame()
    citizen_scores = calculate_citizen_experience_score(operational_mart, hourly_metrics=hourly_metrics, clients=clients_df)
    hourly_patterns = calculate_hourly_citizen_patterns(hourly_metrics)
    zone_summary = build_citizen_zone_summary(citizen_scores, operational_mart)
    citizen_recommendations = recommend_best_wifi_zones(citizen_scores, top_n=10)
    citizen_alerts = build_citizen_alerts(citizen_scores)
    feedback_df = load_citizen_feedback()
    feedback_summary = summarize_citizen_feedback(feedback_df)
    digital_equity_df = calculate_digital_equity_proxy(
        operational_mart,
        citizen_scores,
        osm_context=osm_context_df if isinstance(osm_context_df, pd.DataFrame) else pd.DataFrame(),
        feedback_summary=feedback_summary,
    )
    calendar_summary = summarize_calendar_impact(calendar_context_df if isinstance(calendar_context_df, pd.DataFrame) else pd.DataFrame())

    return {
        "operational_mart": operational_mart,
        "citizen_scores": citizen_scores,
        "hourly_patterns": hourly_patterns,
        "zone_summary": zone_summary,
        "recommendations": citizen_recommendations,
        "alerts": citizen_alerts,
        "feedback_df": feedback_df,
        "feedback_summary": feedback_summary,
        "calendar_context": calendar_context_df if isinstance(calendar_context_df, pd.DataFrame) else pd.DataFrame(),
        "calendar_summary": calendar_summary,
        "digital_equity": digital_equity_df,
        "is_meraki_mode": is_meraki_mode,
    }


def build_social_roi_bundle(
    active_results: dict[str, object],
    citizen_bundle: dict[str, object],
    socioeconomic_df: pd.DataFrame | None = None,
    socioeconomic_validation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Construye artefactos de retorno social a partir de red + experiencia + datos socioeconómicos agregados."""
    validation = socioeconomic_validation if isinstance(socioeconomic_validation, dict) else {}
    socio_df = socioeconomic_df if isinstance(socioeconomic_df, pd.DataFrame) else pd.DataFrame()
    merged_inputs = prepare_social_roi_inputs(
        operational_mart=active_results.get("operational_mart", pd.DataFrame()) if isinstance(active_results, dict) else pd.DataFrame(),
        citizen_scores=citizen_bundle.get("citizen_scores", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame(),
        digital_equity_df=citizen_bundle.get("digital_equity", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame(),
        socioeconomic_df=socio_df,
        osm_context=active_results.get("osm_context", pd.DataFrame()) if isinstance(active_results, dict) else pd.DataFrame(),
    )
    social_roi_scores = calculate_social_roi_score(merged_inputs)
    recommendations_df = generate_social_infrastructure_recommendations(social_roi_scores)

    limitations = list(validation.get("warnings", [])) if isinstance(validation.get("warnings"), list) else []
    if isinstance(validation.get("privacy_warnings"), list):
        limitations.extend(validation.get("privacy_warnings", []))
    if socio_df.empty:
        limitations.append("No hay dataset socioeconómico cargado; el score de retorno social no puede priorizar vulnerabilidad agregada.")

    return {
        "merged_inputs": merged_inputs,
        "social_roi_scores": social_roi_scores,
        "recommendations": recommendations_df,
        "validation": validation,
        "limitations": limitations,
    }


def flatten_passports(passports: list[dict[str, object]]) -> pd.DataFrame:
    """Convierte pasaportes a tabla para revisión y descarga."""
    if not passports:
        return pd.DataFrame()

    rows = []
    for passport in passports:
        row = passport.copy()
        for field_name in [
            "evidencia_tecnica",
            "evidencia_contextual",
            "limitaciones",
            "datos_usados",
            "datos_faltantes",
        ]:
            value = row.get(field_name, [])
            if isinstance(value, list):
                row[field_name] = " | ".join(str(item) for item in value)
        rows.append(row)

    return pd.DataFrame(rows)


def build_local_results(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    available_crews: int = 3,
    synthetic_flag: bool = False,
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Construye un bundle local sin APIs externas y sin ejecutar el orquestador completo."""
    readiness = calculate_data_readiness(dataframe, schema_mapping)
    is_meraki_mode = isinstance(wifi_package, dict) and bool(wifi_package.get("is_official_package"))
    meraki_anomalies = pd.DataFrame()
    if is_meraki_mode:
        from src.meraki_anomaly_engine import build_meraki_decision_passports, detect_hourly_anomalies

        dataframe.attrs["source"] = "meraki_package"
        hourly_metrics = wifi_package.get("hourly_metrics", pd.DataFrame())
        if isinstance(hourly_metrics, pd.DataFrame):
            dataframe.attrs["meraki_hourly_metrics"] = hourly_metrics
            meraki_anomalies = detect_hourly_anomalies(hourly_metrics)

    work_orders = generate_work_orders(dataframe, schema_mapping)
    impact_scores = calculate_impact_scores(dataframe, schema_mapping, work_orders=work_orders)
    recommendations = generate_strategic_recommendations(
        dataframe,
        schema_mapping,
        work_orders=work_orders,
        impact_scores_df=impact_scores,
    )
    if is_meraki_mode:
        decision_passports = build_meraki_decision_passports(dataframe, work_orders)
    else:
        decision_passports = generate_passports_for_top_zones(
            impact_scores,
            work_orders=work_orders,
            recommendations=recommendations,
            top_n=10,
        )
    work_orders = generate_work_orders(
        dataframe,
        schema_mapping,
        impact_scores_df=impact_scores,
        decision_passports=decision_passports,
    )
    crew_plan = optimize_crews(impact_scores, available_crews=available_crews)

    results: dict[str, object] = {
        "trace_id": None,
        "readiness": readiness,
        "work_orders": work_orders,
        "recommendations": recommendations,
        "impact_scores": impact_scores,
        "crew_plan": crew_plan,
        "decision_passports": decision_passports,
        "operational_mart": dataframe.copy() if is_meraki_mode else pd.DataFrame(),
        "meraki_anomalies": meraki_anomalies if is_meraki_mode else pd.DataFrame(),
        "calendar_context": pd.DataFrame(),
        "weather_context": pd.DataFrame(),
        "osm_context": pd.DataFrame(),
        "limitations": [],
        "confidence_level": "Media" if readiness.get("score", 0) >= 60 else "Baja",
        "agent_event_log": [],
        "audit_log": [],
        "audit_summary": {},
        "is_synthetic_data": synthetic_flag,
        "is_meraki_mode": is_meraki_mode,
        "wifi_package_summary": get_package_summary(wifi_package) if is_meraki_mode else {},
    }
    results["quality_gate_report"] = build_quality_gate_report(dataframe, schema_mapping, results=results)
    results["executive_summary"] = build_executive_summary(results)
    return results


def get_cached_base_results(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    file_signature: str,
    mapping_signature: str,
    synthetic_flag: bool,
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Cachea resultados locales para no recalcular en cada pestaña."""
    state_key = build_state_key("base_results", file_signature, mapping_signature)
    if state_key not in st.session_state:
        st.session_state[state_key] = build_local_results(
            dataframe,
            schema_mapping,
            available_crews=3,
            synthetic_flag=synthetic_flag,
            wifi_package=wifi_package,
        )
    return st.session_state[state_key]


def combine_audit_logs(*logs: list[dict[str, object]] | None) -> list[dict[str, object]]:
    """Combina varias bitácoras sin duplicar eventos."""
    combined: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for log in logs:
        if not log:
            continue
        for event in log:
            audit_id = str(event.get("audit_id", ""))
            if audit_id and audit_id in seen_ids:
                continue
            if audit_id:
                seen_ids.add(audit_id)
            combined.append(event)

    combined.sort(key=lambda item: str(item.get("timestamp", "")))
    return combined


def get_active_results(
    base_results: dict[str, object],
    cycle_results: dict[str, object] | None,
    replay_state: dict[str, object] | None,
    synthetic_flag: bool,
) -> dict[str, object]:
    """Prioriza resultados de simulación, luego Mission Control y luego cálculo local."""
    current_dataset_signature = st.session_state.get("current_dataset_signature")

    def apply_session_overlays(payload: dict[str, object]) -> dict[str, object]:
        """Superpone resultados manuales recientes sin tocar los cálculos base."""
        manual_weather_payload = st.session_state.get("latest_manual_weather_context", {})
        if isinstance(manual_weather_payload, dict) and manual_weather_payload.get("dataset_signature") == current_dataset_signature:
            payload["weather_context"] = manual_weather_payload.get("data", pd.DataFrame())

        manual_osm_payload = st.session_state.get("latest_manual_osm_context", {})
        if isinstance(manual_osm_payload, dict) and manual_osm_payload.get("dataset_signature") == current_dataset_signature:
            payload["osm_context"] = manual_osm_payload.get("data", pd.DataFrame())

        recommendations_payload = st.session_state.get("gemini_strategic_recommendations_payload", {})
        if isinstance(recommendations_payload, dict) and recommendations_payload.get("dataset_signature") == current_dataset_signature:
            payload["recommendations"] = recommendations_payload.get("data", pd.DataFrame())
            payload["recommendations_summary"] = recommendations_payload.get("summary", "")
            payload["recommendations_source"] = recommendations_payload.get("source", "fallback")

        return payload

    if replay_state and replay_state.get("current_results"):
        merged_results = dict(base_results)
        merged_results.update(replay_state["current_results"])
        merged_results["trace_id"] = replay_state.get("trace_id")
        merged_results["replay_timeline"] = replay_state.get("timeline", pd.DataFrame())
        merged_results["audit_log"] = replay_state.get("audit_log", [])
        merged_results["audit_summary"] = build_operational_audit_summary(merged_results["audit_log"])
        merged_results["is_synthetic_data"] = synthetic_flag
        return apply_session_overlays(merged_results)

    if cycle_results:
        cycle_payload = dict(cycle_results)
        cycle_payload["is_synthetic_data"] = synthetic_flag
        return apply_session_overlays(cycle_payload)

    fallback_payload = dict(base_results)
    fallback_payload["is_synthetic_data"] = synthetic_flag
    return apply_session_overlays(fallback_payload)


def build_export_payload(
    active_results: dict[str, object],
    replay_state: dict[str, object] | None,
    human_review_log: pd.DataFrame | None,
    quality_gate_report: dict[str, object] | None,
    operational_audit_log: list[dict[str, object]] | None,
    synthetic_flag: bool,
    citizen_bundle: dict[str, object] | None = None,
    citizen_insights_markdown: str | None = None,
    social_roi_bundle: dict[str, object] | None = None,
    social_roi_explanation_markdown: str | None = None,
) -> dict[str, object]:
    """Enriquece resultados para paquete de evidencia."""
    payload = dict(active_results)
    payload["is_synthetic_data"] = synthetic_flag
    payload["replay_timeline"] = replay_state.get("timeline", pd.DataFrame()) if replay_state else pd.DataFrame()
    payload["human_review_log"] = human_review_log if human_review_log is not None else pd.DataFrame()
    payload["quality_gate_report"] = quality_gate_report or payload.get("quality_gate_report", {})
    payload["operational_audit_log"] = operational_audit_log or payload.get("audit_log", [])
    payload["operational_audit_summary"] = build_operational_audit_summary(payload["operational_audit_log"])
    if isinstance(citizen_bundle, dict):
        payload["citizen_experience_scores"] = citizen_bundle.get("citizen_scores", pd.DataFrame())
        payload["citizen_recommendations"] = citizen_bundle.get("recommendations", pd.DataFrame())
        payload["citizen_feedback"] = citizen_bundle.get("feedback_df", pd.DataFrame())
        payload["citizen_feedback_summary"] = citizen_bundle.get("feedback_summary", {})
        payload["digital_equity_proxy"] = citizen_bundle.get("digital_equity", pd.DataFrame())
        payload["citizen_zone_summary"] = citizen_bundle.get("zone_summary", pd.DataFrame())
        payload["citizen_alerts"] = citizen_bundle.get("alerts", pd.DataFrame())
        payload["calendar_public_context"] = citizen_bundle.get("calendar_context", pd.DataFrame())
        payload["calendar_public_summary"] = citizen_bundle.get("calendar_summary", {})
    if citizen_insights_markdown:
        payload["citizen_insights_markdown"] = citizen_insights_markdown
    if isinstance(social_roi_bundle, dict):
        payload["socioeconomic_validation"] = social_roi_bundle.get("validation", {})
        payload["social_roi_scores"] = social_roi_bundle.get("social_roi_scores", pd.DataFrame())
        payload["social_roi_recommendations"] = social_roi_bundle.get("recommendations", pd.DataFrame())
    if social_roi_explanation_markdown:
        payload["social_roi_explanation_markdown"] = social_roi_explanation_markdown
    return payload


def display_list(title: str, items: list[str], empty_message: str) -> None:
    """Muestra una lista sencilla y tolerante a vacíos."""
    st.markdown(f"**{title}**")
    if not items:
        st.write(empty_message)
        return
    for item in items:
        st.write(f"- {item}")


def label_local_dataset(path: Path) -> str:
    """Genera una etiqueta clara para datasets locales."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_default_wifi_package_folder() -> str:
    """Propone la carpeta más probable para el paquete oficial."""
    preferred_dir = PROJECT_ROOT / "data" / "raw" / "Zonas-WiFi-Inteligentes"
    if preferred_dir.exists():
        return str(preferred_dir)
    fallback_dir = PROJECT_ROOT / "data" / "raw"
    return str(fallback_dir)


def is_meraki_package_active() -> bool:
    """Indica si hay un paquete oficial cargado en la sesión."""
    package = st.session_state.get("wifi_package")
    return isinstance(package, dict) and bool(package.get("is_official_package"))


def advance_replay_state(
    replay_state: dict[str, object],
    schema_mapping: dict[str, str | None],
    available_crews: int,
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Avanza un lote de la simulación operativa."""
    current_step = int(replay_state.get("current_step", 0))
    total_steps = int(replay_state.get("total_steps", 0))
    if current_step >= total_steps:
        return replay_state

    next_step = current_step + 1
    events_df = replay_state["events_df"]
    batch_size = int(replay_state["batch_size"])
    batch_df = get_replay_batch(events_df, step=next_step, batch_size=batch_size)
    previous_results = replay_state.get("current_results")
    current_results = run_replay_analysis(
        batch_df,
        schema_mapping,
        available_crews=available_crews,
        wifi_package=wifi_package,
    )
    changes = detect_replay_changes(previous_results, current_results)

    history = list(replay_state.get("history", []))
    history.append(
        {
            "step": next_step,
            "results": current_results,
            "changes": changes,
            "timestamp": get_timestamp(),
        }
    )

    audit_log = append_audit_event(
        replay_state.get("audit_log", []),
        create_audit_event(
            module="Simulacion Operativa",
            action="Procesar lote",
            status="ok" if not current_results.get("warnings") else "warning",
            message=(
                f"Paso {next_step}/{total_steps}: {current_results.get('processed_rows', 0)} filas acumuladas, "
                f"{len(current_results.get('work_orders', pd.DataFrame()))} ordenes."
            ),
            metadata={
                "trace_id": replay_state.get("trace_id"),
                "step": next_step,
                "processed_rows": int(current_results.get("processed_rows", 0)),
                "warnings": current_results.get("warnings", []),
            },
        ),
    )

    replay_state["current_step"] = next_step
    replay_state["current_results"] = current_results
    replay_state["last_changes"] = changes
    replay_state["history"] = history
    replay_state["timeline"] = build_replay_timeline(history)
    replay_state["audit_log"] = audit_log
    replay_state["audit_summary"] = build_operational_audit_summary(audit_log)
    return replay_state


st.markdown(
    (
        "<div class='cw-header-block'>"
        f"<h1 class='cw-header-title'>{APP_TITLE}</h1>"
        f"<div class='cw-header-subtitle'>{APP_SUBTITLE}</div>"
        f"<div class='cw-header-subtitle' style='margin-top:0.6rem;'>{INSPECTION_DESCRIPTION}</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)
st.markdown(f"<div class='cw-flow-banner'>{RECOMMENDED_FLOW_TEXT}</div>", unsafe_allow_html=True)
selected_profile = render_profile_selector()
visible_sections = get_citizen_sections() if selected_profile == CITIZEN_PROFILE else get_technical_sections()

with st.sidebar:
    st.markdown("### Navegación rápida")
    st.caption("Sigue este flujo para usar la plataforma de forma estable.")
    st.markdown(
        "\n".join(
            [
                "1. Carga e Inspección",
                "2. Mapeo de Columnas",
                "3. Mission Control",
                "4. Simulación Operativa",
                "5. Vista Ejecutiva 360",
                "6. Paquete de Evidencia",
            ]
        )
    )
    st.caption(f"Portal activo: {selected_profile}")
    if selected_profile == TECHNICAL_PROFILE:
        st.info("Perfil técnico: se priorizan módulos de operación, auditoría y evidencia.")
    else:
        st.info("Perfil ciudadano: se priorizan módulos de experiencia, equidad y retorno social.")
    st.markdown("Secciones visibles:")
    st.markdown("\n".join([f"- {section}" for section in visible_sections]))

st.markdown("### Cargar paquete oficial Zonas WiFi Inteligentes")
package_control_col1, package_control_col2, package_control_col3 = st.columns([1.2, 1.2, 1.6])
with package_control_col1:
    package_source_mode = st.radio(
        "Origen del paquete Meraki",
        options=["Carpeta local", "GitHub oficial"],
        horizontal=True,
        key="wifi_package_source_mode",
    )
with package_control_col2:
    if package_source_mode == "Carpeta local":
        package_folder_input = st.text_input(
            "Ruta base del paquete",
            value=st.session_state.get("wifi_package_folder", get_default_wifi_package_folder()),
            key="wifi_package_folder",
        )
    else:
        package_repo_url = st.text_input(
            "Repo URL",
            value=st.session_state.get(
                "wifi_package_repo_url",
                "https://github.com/AlejandroTenorioT/Zonas-WiFi-Inteligentes",
            ),
            key="wifi_package_repo_url",
        )
with package_control_col3:
    st.caption(
        "Modo multitabla para el paquete Cisco Meraki anonimizando de Zonas WiFi Inteligentes. "
        "No reemplaza la carga CSV/Excel individual."
    )
    if st.button("Cargar paquete Meraki", use_container_width=True):
        with st.spinner("Cargando paquete oficial Meraki..."):
            try:
                if package_source_mode == "Carpeta local":
                    loaded_package = load_wifi_package_from_folder(package_folder_input)
                else:
                    loaded_package = load_wifi_package_from_github(package_repo_url)
                st.session_state["wifi_package"] = loaded_package
                st.session_state["wifi_package_summary"] = get_package_summary(loaded_package)
                if loaded_package.get("is_official_package"):
                    st.success("Paquete Meraki cargado. Ahora puedes construir el mart operativo.")
                else:
                    st.warning("Se cargó un paquete parcial o con advertencias. Revisa las tablas detectadas.")
            except Exception as error:  # noqa: BLE001
                st.error(f"No fue posible cargar el paquete Meraki: {error}")

uploader_col, local_col = st.columns([1.7, 1.3])
with uploader_col:
    uploaded_file = st.file_uploader(
        "Subir dataset manual",
        type=["csv", "xlsx", "xls", "txt"],
        help="Acepta CSV, TXT y Excel sin asumir columnas específicas.",
    )

local_datasets = list_local_datasets()
local_dataset_map = {label_local_dataset(path): path for path in local_datasets}

with local_col:
    selected_local_label = st.selectbox(
        "O usar dataset local",
        options=["Sin seleccionar"] + list(local_dataset_map.keys()),
        help="Busca archivos en data/raw/ y en la raíz del proyecto sin depender de un nombre fijo.",
    )
    if st.button("Cargar dataset local seleccionado", use_container_width=True):
        if selected_local_label == "Sin seleccionar":
            st.warning("Selecciona primero un archivo local.")
        else:
            st.session_state["active_local_dataset_path"] = str(local_dataset_map[selected_local_label])
            st.session_state["use_synthetic_data"] = False

if st.session_state.get("use_synthetic_data") and not st.session_state.get("_dev_allow_synthetic_mode", False):
    st.session_state["use_synthetic_data"] = False

if False:
    if st.button("Usar datos sintéticos de demostración", use_container_width=True):
        st.session_state["use_synthetic_data"] = True
        st.session_state["active_local_dataset_path"] = None

if uploaded_file is not None:
    st.session_state["use_synthetic_data"] = False
    st.session_state["active_local_dataset_path"] = None

dataframe: pd.DataFrame | None = None
dataset_name = ""
dataset_source = ""
dataset_source_path = ""
file_signature = "sin_dataset"
saved_copy_path: Path | None = None
saved_profile_path: Path | None = None
data_loading_error = ""
wifi_package = st.session_state.get("wifi_package")
wifi_package_summary = (
    st.session_state.get("wifi_package_summary")
    if isinstance(st.session_state.get("wifi_package_summary"), dict)
    else {}
)
meraki_operational_mart = st.session_state.get("meraki_operational_mart")

try:
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        file_signature = build_file_signature(uploaded_bytes)
        dataframe = load_tabular_file(uploaded_file.name, uploaded_bytes)
        dataset_name = uploaded_file.name
        dataset_source = "Upload manual"
        saved_copy_path = save_file_once_per_session(uploaded_file.name, uploaded_bytes, file_signature)
    elif st.session_state.get("active_local_dataset_path"):
        local_path = Path(st.session_state["active_local_dataset_path"]).resolve()
        if not local_path.exists():
            st.session_state["active_local_dataset_path"] = None
            raise DataLoaderError("El dataset local seleccionado ya no existe.")
        local_bytes = local_path.read_bytes()
        file_signature = build_file_signature(local_bytes)
        dataframe = load_local_dataset(local_path)
        dataset_name = local_path.name
        dataset_source = "Archivo local"
        dataset_source_path = str(local_path)
    elif isinstance(meraki_operational_mart, pd.DataFrame) and not meraki_operational_mart.empty and is_meraki_package_active():
        dataframe = meraki_operational_mart.copy()
        dataframe.attrs["source"] = "meraki_package"
        if isinstance(wifi_package, dict) and isinstance(wifi_package.get("hourly_metrics"), pd.DataFrame):
            dataframe.attrs["meraki_hourly_metrics"] = wifi_package["hourly_metrics"]
        dataset_name = "meraki_operational_mart.csv"
        dataset_source = "Paquete oficial Zonas WiFi Inteligentes"
        mart_bytes = dataframe.to_csv(index=False).encode("utf-8")
        file_signature = build_file_signature(mart_bytes)
    elif st.session_state.get("use_synthetic_data"):
        dataframe = generate_synthetic_wifi_data()
        synthetic_bytes = dataframe.to_csv(index=False).encode("utf-8")
        file_signature = build_file_signature(synthetic_bytes)
        dataset_name = "synthetic_wifi_demo.csv"
        dataset_source = "Demo sintética"
except (DataLoaderError, OSError) as error:
    data_loading_error = str(error)

synthetic_flag = is_synthetic_dataset(dataframe)
st.session_state["current_dataset_signature"] = file_signature
if synthetic_flag:
    st.error(f"{SYNTHETIC_WARNING_TITLE}: {SYNTHETIC_WARNING_BODY}")

if data_loading_error:
    st.error(data_loading_error)

profile: dict[str, object] = {}
suggested_mapping: dict[str, str | None] = {field_key: None for field_key in MAPPING_FIELD_CONFIG}
if dataframe is not None and not dataframe.empty:
    profile = build_dataset_profile(dataframe)
    saved_profile_path = save_profile_once_per_session(profile, dataset_name, file_signature)
    suggested_mapping = suggest_schema_mapping(dataframe)
    initialize_mapping_state(file_signature, suggested_mapping)

all_sections = [
    "Carga e Inspeccion",
    "Mapeo de Columnas",
    "Mission Control",
    "Simulacion Operativa",
    "Vista Ejecutiva 360",
    "Portal Ciudadano",
    "Experiencia Ciudadana",
    "Recomendador de Zonas WiFi",
    "Buzon Ciudadano",
    "Equidad Digital",
    "Retorno Social de Conectividad",
    "Agente Operativo",
    "Impacto Ciudadano",
    "Cuadrillas",
    "Pasaporte de Decision",
    "Agente Estrategico",
    "Agente Conversacional",
    "Agente Ciudadano",
    "Vista Publica de Calidad",
    "Validacion Humana",
    "Blindaje Tecnico",
    "Auditoria Operativa",
    "Paquete de Evidencia",
]
tabs = st.tabs(all_sections)
tab_visibility_css = build_tab_visibility_css(all_sections, visible_sections)
if tab_visibility_css:
    st.markdown(tab_visibility_css, unsafe_allow_html=True)

schema_mapping: dict[str, str | None] = {field_key: None for field_key in MAPPING_FIELD_CONFIG}
mapping_signature = build_mapping_signature(schema_mapping)
base_results: dict[str, object] | None = None
cycle_results: dict[str, object] | None = None
replay_state: dict[str, object] | None = None
review_queue: pd.DataFrame | None = None
quality_gate_report: dict[str, object] | None = None
manual_audit_log: list[dict[str, object]] = []
active_results: dict[str, object] | None = None


with tabs[0]:
    if is_meraki_package_active():
        summary_payload = wifi_package_summary or get_package_summary(wifi_package)
        st.success("Modo Meraki / Zonas WiFi Inteligentes activo.")
        st.write(f"- Fuente del paquete: `{summary_payload.get('source', 'No disponible')}`")
        display_list(
            "Limitaciones del paquete",
            summary_payload.get("warnings", []),
            "Sin advertencias adicionales del paquete.",
        )
        package_table_summary = summary_payload.get("table_summary", pd.DataFrame())
        if isinstance(package_table_summary, pd.DataFrame) and not package_table_summary.empty:
            st.markdown("**Tablas detectadas del paquete oficial**")
            st.dataframe(package_table_summary, use_container_width=True, height=230)

        build_mart_col1, build_mart_col2 = st.columns([1, 2])
        with build_mart_col1:
            if st.button("Construir mart operativo", use_container_width=True, key="build_meraki_mart_button"):
                try:
                    mart_df = build_operational_mart(wifi_package)
                    mart_df.attrs["source"] = "meraki_package"
                    if isinstance(wifi_package.get("hourly_metrics"), pd.DataFrame):
                        mart_df.attrs["meraki_hourly_metrics"] = wifi_package["hourly_metrics"]
                    st.session_state["meraki_operational_mart"] = mart_df
                    st.success("Mart operativo Meraki construido. La app usará este dataset como base analítica.")
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(f"No fue posible construir el mart operativo: {error}")
        with build_mart_col2:
            current_mart = st.session_state.get("meraki_operational_mart")
            if isinstance(current_mart, pd.DataFrame) and not current_mart.empty:
                st.caption(
                    "El mart operativo ya está disponible y alimenta Mission Control, Replay, Impacto, Pasaportes y Evidencia."
                )
    st.subheader("Módulo 1 y 2: carga, inspección y análisis estructural")
    if dataframe is None or dataframe.empty:
        show_initial_instructions()
        if local_datasets:
            st.caption("Datasets locales detectados:")
            for dataset_path in local_datasets[:10]:
                st.write(f"- {label_local_dataset(dataset_path)}")
        else:
            st.caption("No se detectaron datasets locales en `data/raw/` ni en la raíz del proyecto.")
    else:
        summary = get_dataframe_summary(dataframe)
        st.success("Dataset activo listo para inspección.")
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Filas", summary["rows"])
        info_col2.metric("Columnas", summary["columns"])
        info_col3.metric("Duplicados", int(profile.get("duplicated_rows", 0)))

        st.markdown("**Metadatos del dataset**")
        st.write(f"- Archivo activo: `{dataset_name}`")
        st.write(f"- Fuente: {dataset_source}")
        if dataset_source_path:
            st.write(f"- Ruta local: `{dataset_source_path}`")
        if saved_copy_path is not None:
            st.write(f"- Copia guardada en `data/raw/`: `{saved_copy_path}`")
        if saved_profile_path is not None:
            st.write(f"- Perfil JSON guardado en `data/processed/`: `{saved_profile_path}`")

        st.warning(
            "Todavía no se ha validado qué columnas sirven para detección de fallas, "
            "priorización o análisis urbano."
        )

        st.markdown("**Vista previa de datos**")
        st.dataframe(dataframe.head(20), use_container_width=True)

        preview_col, dtypes_col = st.columns(2)
        with preview_col:
            st.markdown("**Columnas y tipos**")
            st.dataframe(get_column_types(dataframe), use_container_width=True, height=320)
        with dtypes_col:
            st.markdown("**Valores nulos por columna**")
            st.dataframe(get_null_counts(dataframe), use_container_width=True, height=320)

        st.markdown("**Resumen básico de calidad**")
        st.dataframe(build_quality_summary(dataframe), use_container_width=True)

        st.markdown("**Columnas candidatas detectadas por heurística**")
        st.dataframe(build_candidate_table(profile), use_container_width=True)

        st.divider()
        st.subheader("Análisis estructural asistido con Gemini")
        gemini_config = load_gemini_config()
        gemini_state = "Configurado" if is_gemini_configured() else "No configurado"
        st.write(f"- Estado Gemini: **{gemini_state}**")
        st.write(f"- Modelo: `{gemini_config['model']}`")
        st.caption("Gemini solo recibe el perfil estructural resumido, no la base completa.")

        structural_key = build_state_key("structural_gemini", file_signature)
        if is_gemini_configured():
            if st.button("Generar análisis estructural con Gemini"):
                with st.spinner("Generando análisis estructural..."):
                    st.session_state[structural_key] = generate_gemini_text(
                        build_final_gemini_prompt(profile)
                    )
            if st.session_state.get(structural_key):
                st.markdown(st.session_state[structural_key])
        else:
            st.info(
                "Para activar Gemini crea un archivo `.env` en la raíz con:\n"
                "`GEMINI_API_KEY=tu_clave`\n`GEMINI_MODEL=gemini-2.5-flash`"
            )


with tabs[1]:
    st.subheader("Mapeo flexible de columnas")
    if dataframe is None or dataframe.empty:
        st.info("Carga o selecciona primero un dataset.")
    elif is_meraki_package_active() and isinstance(st.session_state.get("meraki_operational_mart"), pd.DataFrame):
        schema_mapping = build_meraki_schema_mapping()
        mapping_signature = build_mapping_signature(schema_mapping)
        module_readiness = get_module_readiness(dataframe, schema_mapping)
        readiness_result = calculate_data_readiness(dataframe, schema_mapping)

        st.success("Mapeo automático aplicado para paquete Meraki.")
        st.caption("El modo Meraki usa el esquema canónico del paquete oficial. No requiere selección manual.")
        st.dataframe(build_mapping_summary_table(schema_mapping), use_container_width=True)
        st.dataframe(build_readiness_table(module_readiness), use_container_width=True)
        st.dataframe(build_readiness_checks_table(readiness_result), use_container_width=True)
        st.dataframe(build_readiness_alignment_table(readiness_result), use_container_width=True)
    else:
        st.write(
            "Selecciona manualmente qué columna representa cada variable lógica. "
            "El sistema sugiere candidatos, pero no asume verdades definitivas."
        )

        available_columns = ["Sin seleccionar"] + [str(column) for column in dataframe.columns]
        selector_cols = st.columns(2)
        raw_mapping: dict[str, str | None] = {}

        for index, (field_key, config) in enumerate(MAPPING_FIELD_CONFIG.items()):
            widget_key = f"mapping_widget_{field_key}"
            if st.session_state.get(widget_key) not in available_columns:
                st.session_state[widget_key] = "Sin seleccionar"
            default_value = st.session_state.get(widget_key, "Sin seleccionar")
            with selector_cols[index % 2]:
                raw_mapping[field_key] = st.selectbox(
                    config["label"],
                    options=available_columns,
                    index=available_columns.index(default_value),
                    key=widget_key,
                    help=f"Sugerido: {suggested_mapping.get(field_key) or 'sin sugerencia'}",
                )

        schema_mapping = build_schema_mapping(raw_mapping)
        mapping_signature = build_mapping_signature(schema_mapping)
        module_readiness = get_module_readiness(dataframe, schema_mapping)
        readiness_result = calculate_data_readiness(dataframe, schema_mapping)

        st.markdown("**Resumen del mapeo**")
        st.dataframe(build_mapping_summary_table(schema_mapping), use_container_width=True)

        st.markdown("**Estado mínimo por agente**")
        st.dataframe(build_readiness_table(module_readiness), use_container_width=True)
        st.warning(
            "Estas columnas son candidatas detectadas por heurística. Aún deben validarse manualmente "
            "antes de construir anomalías, agentes o dashboard final."
        )

        score_col1, score_col2 = st.columns(2)
        score_col1.metric("Data Readiness Score", readiness_result["score"])
        score_col2.metric("Clasificación", readiness_result["classification"])

        st.markdown("**Checks del readiness**")
        st.dataframe(build_readiness_checks_table(readiness_result), use_container_width=True)

        st.markdown("**Alineación con agentes oficiales**")
        st.dataframe(build_readiness_alignment_table(readiness_result), use_container_width=True)

        display_list("Fortalezas", readiness_result.get("strengths", []), "Sin fortalezas registradas.")
        display_list("Brechas", readiness_result.get("gaps", []), "Sin brechas registradas.")
        display_list(
            "Próximas acciones recomendadas",
            readiness_result.get("recommended_next_actions", []),
            "No hay acciones adicionales registradas.",
        )


if dataframe is not None and not dataframe.empty:
    base_results = get_cached_base_results(
        dataframe,
        schema_mapping,
        file_signature,
        mapping_signature,
        synthetic_flag,
        wifi_package=wifi_package if is_meraki_package_active() else None,
    )
    cycle_state_key = build_state_key("cycle_results", file_signature, mapping_signature)
    replay_state_key = build_state_key("replay_state", file_signature, mapping_signature)
    review_queue_key = build_state_key("review_queue", file_signature, mapping_signature)
    quality_gate_key = build_state_key("quality_gate", file_signature, mapping_signature)
    manual_audit_key = build_state_key("manual_audit", file_signature, mapping_signature)
    manual_weather_context_key = "latest_manual_weather_context"
    manual_osm_context_key = "latest_manual_osm_context"
    manual_public_calendar_context_key = "latest_public_calendar_context"
    gemini_recommendations_key = "gemini_strategic_recommendations_payload"
    citizen_insights_key = "citizen_insights_payload"

    cycle_results = st.session_state.get(cycle_state_key)
    replay_state = st.session_state.get(replay_state_key)
    review_queue = st.session_state.get(review_queue_key)
    quality_gate_report = st.session_state.get(quality_gate_key) or (
        cycle_results.get("quality_gate_report") if cycle_results else base_results.get("quality_gate_report")
    )
    manual_audit_log = st.session_state.get(manual_audit_key, [])
    active_results = get_active_results(base_results, cycle_results, replay_state, synthetic_flag)

    def sync_latest_operational_snapshot(source_hint: str | None = None) -> None:
        """Sincroniza el snapshot ejecutivo con overlays manuales y evidencia reciente."""
        snapshot = dict(get_active_results(base_results, cycle_results, replay_state, synthetic_flag))
        snapshot["human_review_log"] = st.session_state.get(review_queue_key, pd.DataFrame())
        snapshot["quality_gate_report"] = st.session_state.get(quality_gate_key) or snapshot.get("quality_gate_report", {})
        snapshot["audit_log"] = combine_audit_logs(
            cycle_results.get("audit_log", []) if cycle_results else [],
            replay_state.get("audit_log", []) if replay_state else [],
            st.session_state.get(manual_audit_key, []),
        )

        manual_weather_payload = st.session_state.get(manual_weather_context_key, {})
        if isinstance(manual_weather_payload, dict) and manual_weather_payload.get("dataset_signature") == file_signature:
            snapshot["weather_context"] = manual_weather_payload.get("data", pd.DataFrame())

        manual_osm_payload = st.session_state.get(manual_osm_context_key, {})
        if isinstance(manual_osm_payload, dict) and manual_osm_payload.get("dataset_signature") == file_signature:
            snapshot["osm_context"] = manual_osm_payload.get("data", pd.DataFrame())

        public_calendar_payload = st.session_state.get(manual_public_calendar_context_key, {})
        if isinstance(public_calendar_payload, dict) and public_calendar_payload.get("dataset_signature") == file_signature:
            snapshot["calendar_public_context"] = public_calendar_payload.get("data", pd.DataFrame())

        recommendations_payload = st.session_state.get(gemini_recommendations_key, {})
        if isinstance(recommendations_payload, dict) and recommendations_payload.get("dataset_signature") == file_signature:
            snapshot["recommendations"] = recommendations_payload.get("data", pd.DataFrame())

        socioeconomic_payload = st.session_state.get("socioeconomic_dataset_payload", {})
        if isinstance(socioeconomic_payload, dict) and socioeconomic_payload.get("dataset_signature") == file_signature:
            snapshot["socioeconomic_validation"] = socioeconomic_payload.get("validation", {})

        social_roi_payload = st.session_state.get("social_roi_explanation_payload", {})
        if isinstance(social_roi_payload, dict) and social_roi_payload.get("dataset_signature") == file_signature:
            snapshot["social_roi_explanation_markdown"] = social_roi_payload.get("markdown", "")

        resolved_source = source_hint
        if not resolved_source:
            if replay_state and replay_state.get("current_results"):
                resolved_source = "replay"
            elif cycle_results:
                resolved_source = "mission_control"
            else:
                resolved_source = "mixed"
        save_latest_operational_results(snapshot, source=resolved_source)
else:
    module_readiness = None
    readiness_result = None

citizen_bundle: dict[str, object] = {}
citizen_calendar_context_df = pd.DataFrame()
citizen_insights_markdown = ""
social_roi_bundle: dict[str, object] = {}
socioeconomic_df = pd.DataFrame()
socioeconomic_validation: dict[str, object] = {}
social_roi_explanation_markdown = ""
if dataframe is not None and not dataframe.empty and isinstance(active_results, dict):
    public_calendar_payload = st.session_state.get("latest_public_calendar_context", {})
    if isinstance(public_calendar_payload, dict) and public_calendar_payload.get("dataset_signature") == file_signature:
        citizen_calendar_context_df = public_calendar_payload.get("data", pd.DataFrame())

    citizen_bundle = build_citizen_bundle(
        dataframe,
        schema_mapping,
        active_results,
        wifi_package=wifi_package if is_meraki_package_active() else None,
        calendar_context_df=citizen_calendar_context_df,
    )
    active_results["citizen_experience_scores"] = citizen_bundle.get("citizen_scores", pd.DataFrame())
    active_results["citizen_recommendations"] = citizen_bundle.get("recommendations", pd.DataFrame())
    active_results["citizen_feedback"] = citizen_bundle.get("feedback_df", pd.DataFrame())
    active_results["citizen_feedback_summary"] = citizen_bundle.get("feedback_summary", {})
    active_results["digital_equity_proxy"] = citizen_bundle.get("digital_equity", pd.DataFrame())
    citizen_payload = st.session_state.get("citizen_insights_payload", {})
    if isinstance(citizen_payload, dict) and citizen_payload.get("dataset_signature") == file_signature:
        citizen_insights_markdown = str(citizen_payload.get("markdown", ""))
        active_results["citizen_insights_markdown"] = citizen_insights_markdown

    socioeconomic_payload = st.session_state.get("socioeconomic_dataset_payload", {})
    if isinstance(socioeconomic_payload, dict) and socioeconomic_payload.get("dataset_signature") == file_signature:
        socioeconomic_df = socioeconomic_payload.get("data", pd.DataFrame())
        socioeconomic_validation = socioeconomic_payload.get("validation", {})

    social_roi_bundle = build_social_roi_bundle(
        active_results,
        citizen_bundle,
        socioeconomic_df=socioeconomic_df,
        socioeconomic_validation=socioeconomic_validation,
    )
    active_results["socioeconomic_validation"] = socioeconomic_validation
    active_results["social_roi_scores"] = social_roi_bundle.get("social_roi_scores", pd.DataFrame())
    active_results["social_roi_recommendations"] = social_roi_bundle.get("recommendations", pd.DataFrame())
    social_roi_payload = st.session_state.get("social_roi_explanation_payload", {})
    if isinstance(social_roi_payload, dict) and social_roi_payload.get("dataset_signature") == file_signature:
        social_roi_explanation_markdown = str(social_roi_payload.get("markdown", ""))
        active_results["social_roi_explanation_markdown"] = social_roi_explanation_markdown


with tabs[4]:
    render_section_header(
        "Vista Ejecutiva 360",
        "Centro de comando para monitorear criticidad, impacto, órdenes de trabajo, cuadrillas y evidencia operativa de la red WiFi pública.",
    )
    if synthetic_flag:
        st.error(f"{SYNTHETIC_WARNING_TITLE}: {SYNTHETIC_WARNING_BODY}")

    if dataframe is None or dataframe.empty:
        render_dashboard_empty_state()
    else:
        dashboard_context = get_dashboard_context(dataframe, schema_mapping)
        dashboard_results = normalize_results_for_dashboard(dashboard_context.get("results"))
        dashboard_results["is_synthetic_data"] = synthetic_flag
        dashboard_results["gemini_configured"] = is_gemini_configured()
        if not dashboard_context.get("has_results"):
            render_dashboard_empty_state()
            quick_cols = st.columns(4)
            with quick_cols[0]:
                render_action_card("1. Cargar datos", "Usa el uploader o selecciona un archivo local en data/raw/.", "media")
            with quick_cols[1]:
                render_action_card("2. Mapear columnas", "Completa zona, métricas y geografía si están disponibles.", "media")
            with quick_cols[2]:
                render_action_card("3. Ejecutar Mission Control", "Lanza el ciclo autónomo para generar órdenes, impacto y cuadrillas.", "alta")
            with quick_cols[3]:
                render_action_card("4. Volver aquí", "Abre de nuevo Vista Ejecutiva 360 para revisar KPIs, mapas y alertas.", "baja")
        else:
            source_labels = {
                "mission_control": "Mission Control",
                "replay": "Simulación Operativa",
                "mixed": "Mixto",
                "none": "Sin fuente",
            }
            source_label = source_labels.get(str(dashboard_context.get("source", "none")), "Mixto")
            header_col1, header_col2 = st.columns([3, 2])
            with header_col1:
                render_compact_badge(f"Fuente de resultados: {source_label}", status="ok")
            with header_col2:
                render_compact_badge(
                    f"Trace ID: {dashboard_results.get('trace_id') or 'Sin trace'}",
                    status="info",
                )

            impact_scores_df = dashboard_results.get("impact_scores", pd.DataFrame())
            impact_scores_df = impact_scores_df if isinstance(impact_scores_df, pd.DataFrame) else pd.DataFrame()
            work_orders_df = dashboard_results.get("work_orders", pd.DataFrame())
            work_orders_df = work_orders_df if isinstance(work_orders_df, pd.DataFrame) else pd.DataFrame()
            recommendations_df = dashboard_results.get("recommendations", pd.DataFrame())
            recommendations_df = recommendations_df if isinstance(recommendations_df, pd.DataFrame) else pd.DataFrame()
            replay_timeline_df = dashboard_results.get("replay_timeline", pd.DataFrame())
            replay_timeline_df = replay_timeline_df if isinstance(replay_timeline_df, pd.DataFrame) else pd.DataFrame()
            review_df = dashboard_results.get("human_review_log", pd.DataFrame())
            review_df = review_df if isinstance(review_df, pd.DataFrame) else pd.DataFrame()
            weather_context_df = dashboard_results.get("weather_context", pd.DataFrame())
            weather_context_df = weather_context_df if isinstance(weather_context_df, pd.DataFrame) else pd.DataFrame()
            osm_context_df = dashboard_results.get("osm_context", pd.DataFrame())
            osm_context_df = osm_context_df if isinstance(osm_context_df, pd.DataFrame) else pd.DataFrame()
            operational_mart_df = dashboard_results.get("operational_mart", pd.DataFrame())
            operational_mart_df = operational_mart_df if isinstance(operational_mart_df, pd.DataFrame) else pd.DataFrame()

            api_col1, api_col2, api_col3 = st.columns(3)
            with api_col1:
                render_compact_badge(
                    f"Open-Meteo: {'Activo' if not weather_context_df.empty else ('Sin coordenadas' if not dashboard_context.get('has_coordinates') else 'No disponible')}",
                    status="ok" if not weather_context_df.empty else ("warning" if dashboard_context.get("has_coordinates") else "neutral"),
                )
            with api_col2:
                render_compact_badge(
                    f"OSM Overpass: {'Activo' if not osm_context_df.empty else ('Sin coordenadas' if not dashboard_context.get('has_coordinates') else 'No disponible')}",
                    status="ok" if not osm_context_df.empty else ("warning" if dashboard_context.get("has_coordinates") else "neutral"),
                )
            with api_col3:
                external_cache_used = any(DATA_EXTERNAL_CACHE_DIR.glob("*.json"))
                render_compact_badge(
                    f"Cache externo: {'Usado' if external_cache_used else 'No detectado'}",
                    status="info" if external_cache_used else "neutral",
                )

            kpi_data = create_kpi_cards_data(dashboard_results, df=dataframe)
            kpi_row_one = st.columns(4)
            with kpi_row_one[0]:
                render_premium_kpi_card("Registros procesados", kpi_data["total_records"], "Base activa", "neutral", "📶")
            with kpi_row_one[1]:
                render_premium_kpi_card("Zonas analizadas", kpi_data["total_zones"], "Con score o seguimiento", "neutral", "📍")
            with kpi_row_one[2]:
                render_premium_kpi_card("Órdenes de trabajo", kpi_data["work_orders_count"], "Alertas accionables", "warning", "🛠️")
            with kpi_row_one[3]:
                render_premium_kpi_card("Zonas críticas", kpi_data["critical_zones_count"], "Prioridad máxima", "critical", "🚨")

            kpi_row_two = st.columns(4)
            with kpi_row_two[0]:
                render_premium_kpi_card("Readiness score", kpi_data["readiness_score"], "Preparación del dataset", "info", "🧭")
            with kpi_row_two[1]:
                render_premium_kpi_card("Quality Gate", kpi_data["quality_gate"], "Estado operativo", "ok", "🛡️")
            with kpi_row_two[2]:
                render_premium_kpi_card("Nivel de confianza", kpi_data["confidence_level"], "Resultado actual", "info", "🧠")
            with kpi_row_two[3]:
                render_premium_kpi_card("Órdenes pendientes", kpi_data["pending_orders"], "Pendientes de revisión", "high", "👥")

            if dashboard_results.get("is_meraki_mode") and not operational_mart_df.empty:
                ap_total = int(operational_mart_df["ap_name"].astype(str).nunique()) if "ap_name" in operational_mart_df.columns else len(operational_mart_df)
                ap_online = int(operational_mart_df["status"].astype(str).str.lower().eq("online").sum()) if "status" in operational_mart_df.columns else 0
                ap_offline = int(operational_mart_df["status"].astype(str).str.lower().eq("offline").sum()) if "status" in operational_mart_df.columns else 0
                ap_dormant = int(operational_mart_df["status"].astype(str).str.lower().eq("dormant").sum()) if "status" in operational_mart_df.columns else 0
                total_events = int(pd.to_numeric(operational_mart_df.get("total_events"), errors="coerce").fillna(0).sum()) if "total_events" in operational_mart_df.columns else 0
                total_clients = int(pd.to_numeric(operational_mart_df.get("clients_reported"), errors="coerce").fillna(0).sum()) if "clients_reported" in operational_mart_df.columns else 0
                total_connections = int(pd.to_numeric(operational_mart_df.get("total_connections"), errors="coerce").fillna(0).sum()) if "total_connections" in operational_mart_df.columns else 0
                avg_disconnection = round(float(pd.to_numeric(operational_mart_df.get("avg_disconnection_rate"), errors="coerce").fillna(0).mean()), 3) if "avg_disconnection_rate" in operational_mart_df.columns else 0.0
                top_ap = str(operational_mart_df.iloc[0]["ap_name"]) if "ap_name" in operational_mart_df.columns else "N/A"
                top_zone = str(operational_mart_df.iloc[0]["zone_name"]) if "zone_name" in operational_mart_df.columns else "N/A"
                meraki_row = st.columns(5)
                with meraki_row[0]:
                    render_premium_kpi_card("APs totales", ap_total, "Paquete Meraki", "neutral", "📡")
                with meraki_row[1]:
                    render_premium_kpi_card("APs online", ap_online, "Disponibles", "ok", "🟢")
                with meraki_row[2]:
                    render_premium_kpi_card("APs offline", ap_offline, "Riesgo alto", "critical", "🔴")
                with meraki_row[3]:
                    render_premium_kpi_card("APs dormant", ap_dormant, "Actividad degradada", "warning", "🟠")
                with meraki_row[4]:
                    render_premium_kpi_card("Tasa desconexión prom.", avg_disconnection, "Baseline horario", "info", "📉")
                meraki_row_two = st.columns(5)
                with meraki_row_two[0]:
                    render_premium_kpi_card("Eventos totales", total_events, "network_events + hourly", "neutral", "📚")
                with meraki_row_two[1]:
                    render_premium_kpi_card("Clientes", total_clients, "clients_curated", "neutral", "👥")
                with meraki_row_two[2]:
                    render_premium_kpi_card("Conexiones", total_connections, "hourly_metrics", "neutral", "🔗")
                with meraki_row_two[3]:
                    render_premium_kpi_card("AP más crítico", top_ap, "Mayor riesgo operativo", "critical", "⚠️")
                with meraki_row_two[4]:
                    render_premium_kpi_card("Zona más crítica", top_zone, "Mayor concentración de riesgo", "warning", "📍")

            summary_text = build_executive_dashboard_summary(dashboard_results, df=dataframe)
            render_insight_card("Resumen ejecutivo automático", summary_text, status="info")

            citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame()
            citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
            citizen_recommendations_df = citizen_bundle.get("recommendations", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame()
            citizen_recommendations_df = citizen_recommendations_df if isinstance(citizen_recommendations_df, pd.DataFrame) else pd.DataFrame()
            citizen_alerts_df = citizen_bundle.get("alerts", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame()
            citizen_alerts_df = citizen_alerts_df if isinstance(citizen_alerts_df, pd.DataFrame) else pd.DataFrame()
            citizen_feedback_summary = citizen_bundle.get("feedback_summary", {}) if isinstance(citizen_bundle, dict) else {}
            digital_equity_df = citizen_bundle.get("digital_equity", pd.DataFrame()) if isinstance(citizen_bundle, dict) else pd.DataFrame()
            digital_equity_df = digital_equity_df if isinstance(digital_equity_df, pd.DataFrame) else pd.DataFrame()
            social_roi_scores_df = social_roi_bundle.get("social_roi_scores", pd.DataFrame()) if isinstance(social_roi_bundle, dict) else pd.DataFrame()
            social_roi_scores_df = social_roi_scores_df if isinstance(social_roi_scores_df, pd.DataFrame) else pd.DataFrame()
            social_roi_recommendations_df = social_roi_bundle.get("recommendations", pd.DataFrame()) if isinstance(social_roi_bundle, dict) else pd.DataFrame()
            social_roi_recommendations_df = social_roi_recommendations_df if isinstance(social_roi_recommendations_df, pd.DataFrame) else pd.DataFrame()

            render_section_header(
                "Bloque ciudadano",
                "Indicadores agregados para orientar a usuarios finales y dar visibilidad ejecutiva sobre experiencia estimada, alertas y señales de equidad digital.",
            )
            if citizen_scores_df.empty:
                render_empty_state(
                    "Sin datos ciudadanos aún",
                    "Ejecuta el módulo de Experiencia Ciudadana para alimentar esta vista.",
                )
            else:
                citizen_avg_score = round(
                    float(pd.to_numeric(citizen_scores_df.get("citizen_experience_score"), errors="coerce").dropna().mean()),
                    2,
                )
                recommended_zones_count = int(
                    citizen_scores_df["citizen_status"].astype(str).isin(["Excelente", "Buena"]).sum()
                ) if "citizen_status" in citizen_scores_df.columns else 0
                alert_zones_count = int(
                    citizen_alerts_df["tipo_alerta"].astype(str).isin(["Zona inestable", "Disponibilidad baja", "Desconexiones altas"]).sum()
                ) if not citizen_alerts_df.empty and "tipo_alerta" in citizen_alerts_df.columns else 0
                equity_avg = round(
                    float(pd.to_numeric(digital_equity_df.get("digital_equity_proxy"), errors="coerce").dropna().mean()),
                    2,
                ) if not digital_equity_df.empty else 0.0

                citizen_kpis = st.columns(5)
                with citizen_kpis[0]:
                    render_premium_kpi_card("Citizen Experience", citizen_avg_score, "Promedio agregado", "info", "📶")
                with citizen_kpis[1]:
                    render_premium_kpi_card("Zonas recomendadas", recommended_zones_count, "Excelente o buena", "ok", "✅")
                with citizen_kpis[2]:
                    render_premium_kpi_card("Alertas ciudadanas", alert_zones_count, "Inestables o con baja disponibilidad", "warning", "⚠️")
                with citizen_kpis[3]:
                    render_premium_kpi_card("Reportes ciudadanos", citizen_feedback_summary.get("total_reportes", 0), "Buzón anónimo", "neutral", "📝")
                with citizen_kpis[4]:
                    render_premium_kpi_card("Equidad digital proxy", equity_avg, "Promedio relativo", "info", "🌐")

                citizen_cols = st.columns(2)
                with citizen_cols[0]:
                    render_dataframe_clean(
                        citizen_recommendations_df.head(5),
                        title="Zonas recomendadas para conectarse",
                        height=220,
                    )
                with citizen_cols[1]:
                    render_dataframe_clean(
                        citizen_alerts_df.head(5),
                        title="Alertas ciudadanas agregadas",
                        height=220,
                    )

                render_action_card(
                    "Portal Ciudadano",
                    "Ve a Portal Ciudadano para consultar recomendaciones por zona y horario, revisar alertas y enviar feedback anónimo.",
                    "media",
                )

            render_section_header(
                "Retorno Social de Conectividad",
                "Cruce entre desempeño WiFi, experiencia ciudadana y datos socioeconómicos agregados para priorizar mejoras con mayor impacto público esperado.",
            )
            if social_roi_scores_df.empty:
                render_empty_state(
                    "Sin retorno social calculado",
                    "Carga y valida un dataset socioeconómico agregado para activar esta vista.",
                )
            else:
                top_social_roi_row = social_roi_scores_df.sort_values("social_roi_score", ascending=False).iloc[0]
                social_roi_avg = round(
                    float(pd.to_numeric(social_roi_scores_df.get("social_roi_score"), errors="coerce").dropna().mean()),
                    2,
                )
                high_social_roi = int(
                    social_roi_scores_df["social_roi_label"].astype(str).isin(["Muy alto retorno social", "Alto retorno social"]).sum()
                ) if "social_roi_label" in social_roi_scores_df.columns else 0
                social_roi_cols = st.columns(4)
                with social_roi_cols[0]:
                    render_premium_kpi_card("Social ROI promedio", social_roi_avg, "Retorno social estimado", "info", "🌍")
                with social_roi_cols[1]:
                    render_premium_kpi_card("Zona top ROI", str(top_social_roi_row.get("zone_name", "N/A")), "Mayor prioridad social", "warning", "🎯")
                with social_roi_cols[2]:
                    render_premium_kpi_card("Zonas alto ROI", high_social_roi, "Alto o muy alto retorno", "ok", "✅")
                with social_roi_cols[3]:
                    render_premium_kpi_card(
                        "Nivel socioeconómico",
                        socioeconomic_validation.get("level", "Sin validar"),
                        "Nivel geográfico cargado",
                        "neutral",
                        "📊",
                    )
                social_roi_preview_cols = st.columns(2)
                with social_roi_preview_cols[0]:
                    render_dataframe_clean(
                        social_roi_scores_df.sort_values("social_roi_score", ascending=False).head(5),
                        title="Zonas con mayor retorno social esperado",
                        height=220,
                    )
                with social_roi_preview_cols[1]:
                    render_dataframe_clean(
                        social_roi_recommendations_df.head(5),
                        title="Recomendaciones de infraestructura y acompañamiento",
                        height=220,
                    )

            selected_zone = None
            if not impact_scores_df.empty and "zona" in impact_scores_df.columns:
                selected_zone = st.selectbox(
                    "Zona destacada para radar ejecutivo",
                    impact_scores_df["zona"].astype(str).head(20).tolist(),
                    key="executive_radar_zone",
                )

            row_1_col_1, row_1_col_2 = st.columns(2)
            with row_1_col_1:
                st.plotly_chart(
                    create_classification_donut(impact_scores_df),
                    use_container_width=True,
                    key="executive_classification_donut",
                )
            with row_1_col_2:
                st.plotly_chart(
                    create_priority_bar_chart(impact_scores_df),
                    use_container_width=True,
                    key="executive_priority_bar",
                )

            row_2_col_1, row_2_col_2 = st.columns(2)
            with row_2_col_1:
                st.plotly_chart(
                    create_impact_scatter(impact_scores_df),
                    use_container_width=True,
                    key="executive_impact_scatter",
                )
            with row_2_col_2:
                st.plotly_chart(
                    create_score_component_radar(impact_scores_df, selected_zone=selected_zone),
                    use_container_width=True,
                    key="executive_score_radar",
                )

            row_3_col_1, row_3_col_2 = st.columns(2)
            timeline_fig = create_replay_timeline_chart(replay_timeline_df)
            with row_3_col_1:
                if timeline_fig is not None:
                    st.plotly_chart(
                        timeline_fig,
                        use_container_width=True,
                        key="executive_replay_timeline",
                    )
                else:
                    render_empty_state(
                        "Timeline no disponible",
                        "Ejecuta Simulación Operativa para ver evolución por lotes.",
                    )
            with row_3_col_2:
                st.plotly_chart(
                    create_work_order_status_chart(work_orders_df=work_orders_df, review_queue_df=review_df),
                    use_container_width=True,
                    key="executive_work_order_status",
                )

            row_4_col_1, row_4_col_2 = st.columns(2)
            recommendations_treemap = create_recommendations_treemap(recommendations_df)
            territory_heatmap = create_territory_heatmap(impact_scores_df, territory_col=schema_mapping.get("territory_col"))
            with row_4_col_1:
                if recommendations_df.empty:
                    if st.button(
                        "Generar recomendaciones para activar Treemap",
                        use_container_width=True,
                        key="executive_generate_recommendations_button",
                    ):
                        recommendation_payload = get_or_generate_strategic_recommendations(
                            dashboard_results,
                            df=dataframe,
                            schema_mapping=schema_mapping,
                            force_refresh=True,
                        )
                        recommendations_df = recommendation_payload["recommendations_df"]
                        dashboard_results["recommendations"] = recommendations_df
                        st.session_state[gemini_recommendations_key] = {
                            "dataset_signature": file_signature,
                            "data": recommendations_df,
                            "summary": recommendation_payload.get("summary", ""),
                            "limitations": recommendation_payload.get("limitations", []),
                            "source": recommendation_payload.get("source", "fallback"),
                        }
                        active_results["recommendations"] = recommendations_df
                        sync_latest_operational_snapshot(source_hint=str(dashboard_context.get("source", "mixed")))
                        recommendations_treemap = create_recommendations_treemap(recommendations_df)
                if recommendations_treemap is not None:
                    st.plotly_chart(
                        recommendations_treemap,
                        use_container_width=True,
                        key="executive_recommendations_treemap",
                    )
                    render_dataframe_clean(
                        format_recommendations_for_display(recommendations_df).head(8),
                        title="Recomendaciones estratégicas activas",
                        height=240,
                    )
                else:
                    render_empty_state(
                        "Sin árbol de recomendaciones",
                        "Genera recomendaciones estratégicas con suficiente contexto para activar esta vista.",
                    )
            with row_4_col_2:
                if territory_heatmap is not None:
                    st.plotly_chart(
                        territory_heatmap,
                        use_container_width=True,
                        key="executive_territory_heatmap",
                    )
                else:
                    render_empty_state(
                        "Sin heatmap territorial",
                        "Mapea territorio o comuna para activar esta vista comparativa.",
                    )

            render_section_header("Mapa geográfico de prioridad")
            geo_fig = create_cali_priority_map_pro(
                dataframe,
                schema_mapping,
                impact_scores_df=impact_scores_df,
                work_orders_df=work_orders_df,
                recommendations_df=recommendations_df,
                height=760,
            )
            if geo_fig is not None:
                st.plotly_chart(
                    geo_fig,
                    use_container_width=True,
                    key="executive_geo_priority_map",
                )
                st.caption(
                    "El mapa usa coordenadas reales del dataset cargado, etiquetas con criticidad y un límite de Cali de referencia. "
                    "La cartografía base puede requerir internet."
                )
            else:
                render_empty_state(
                    "Mapa ejecutivo no disponible",
                    "El dataset no contiene latitud/longitud mapeadas. Para activar el mapa ejecutivo, mapea columnas geográficas reales.",
                )

            calendar_fig = create_calendar_heatmap(dataframe, schema_mapping, impact_scores_df=impact_scores_df)
            if calendar_fig is not None:
                st.plotly_chart(
                    calendar_fig,
                    use_container_width=True,
                    key="executive_calendar_heatmap",
                )

            findings = build_top_findings(dashboard_results)
            alerts = build_risk_alerts(dashboard_results)
            actions = build_next_best_actions(dashboard_results)

            render_section_header("Análisis ejecutivo asistido")
            st.caption(
                "Este análisis se basa en resultados generados por Mission Control, Simulación Operativa, órdenes de trabajo, scoring, pasaportes, validación y auditoría."
            )
            insight_button_col1, insight_button_col2 = st.columns([1, 2])
            with insight_button_col1:
                generate_with_gemini = st.button(
                    "Analizar hallazgos con Gemini",
                    use_container_width=True,
                    key="executive_insights_gemini_button",
                )
            with insight_button_col2:
                if not is_gemini_configured():
                    st.caption("Gemini no está configurado. Se mostrará fallback determinístico.")
            insights_payload = get_or_generate_dashboard_insights(
                dashboard_results,
                df=dataframe,
                schema_mapping=schema_mapping,
                force_refresh=generate_with_gemini,
            )
            st.markdown(insights_payload.get("markdown", "No hay análisis disponible."))

            render_section_header("Hallazgos y alertas")
            findings_col, alerts_col, actions_col = st.columns(3)
            with findings_col:
                for finding in findings[:5]:
                    render_insight_card("Hallazgo", finding, status="success")
            with alerts_col:
                for alert in alerts[:5]:
                    render_insight_card("Alerta", alert, status="warning")
            with actions_col:
                for action in actions[:5]:
                    render_action_card("Próxima acción", action, priority="media")

            render_section_header("Acciones rápidas")
            quick_action_cols = st.columns(5)
            with quick_action_cols[0]:
                render_action_card("Mission Control", "Ve a la pestaña Mission Control para recalcular el ciclo.", "alta")
            with quick_action_cols[1]:
                render_action_card("Simulación", "Ve a Simulación Operativa para reproducir el dataset por lotes.", "media")
            with quick_action_cols[2]:
                render_action_card("Pasaportes", "Ve a Pasaporte de Decisión para revisar zonas priorizadas.", "media")
            with quick_action_cols[3]:
                render_action_card("Validación", "Ve a Validación Humana para cerrar órdenes pendientes.", "media")
            with quick_action_cols[4]:
                render_action_card("Evidencia", "Ve a Paquete de Evidencia para exportar resultados y reportes.", "baja")

            with st.expander("Diagnóstico de fuentes de datos", expanded=False):
                st.write(f"- Fuente detectada: **{dashboard_context.get('source', 'none')}**")
                st.write(f"- Claves disponibles en session_state: **{len(dashboard_context.get('available_keys', []))}**")
                st.write(f"- work_orders disponibles: **{'Sí' if not work_orders_df.empty else 'No'}**")
                st.write(f"- impact_scores disponibles: **{'Sí' if not impact_scores_df.empty else 'No'}**")
                st.write(f"- replay_timeline disponible: **{'Sí' if not replay_timeline_df.empty else 'No'}**")
                st.write(f"- Coordenadas disponibles: **{'Sí' if dashboard_context.get('has_coordinates') else 'No'}**")
                st.caption(", ".join(dashboard_context.get("available_keys", [])[:25]))


with tabs[2]:
    st.subheader("Mission Control / Ciclo Autónomo")
    if dataframe is None or dataframe.empty:
        st.info("Carga un dataset y define el mapeo antes de ejecutar el ciclo autónomo.")
    else:
        readiness = build_quality_gate_report(dataframe, schema_mapping, results=base_results)
        state_col1, state_col2, state_col3 = st.columns(3)
        state_col1.metric("Quality gate base", readiness.get("quality_gate", "Sin evaluar"))
        state_col2.metric("Readiness operativo", readiness["operational_readiness"]["operational_status"])
        state_col3.metric("Score operativo", readiness["operational_readiness"]["score"])
        if is_meraki_package_active():
            st.info(
                "Modo Meraki activo: Mission Control usará el mart operativo por AP y las métricas horarias del paquete oficial."
            )
        st.caption(
            "Open-Meteo y OpenStreetMap/Overpass se usarán para enriquecer contexto territorial y climático. "
            "Se aplican límites y caché para evitar llamadas excesivas."
        )

        control_col1, control_col2, control_col3 = st.columns(3)
        available_crews_cycle = control_col1.number_input("Cuadrillas disponibles", min_value=1, value=3, step=1)
        use_weather_context = control_col2.toggle("Usar clima contextual", value=bool(schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col")))
        use_osm_context = control_col3.toggle("Usar contexto urbano OSM", value=bool(schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col")))
        max_external_points = st.slider("Máximo de puntos externos", min_value=1, max_value=50, value=20)
        if max_external_points > 20:
            st.warning("Superar 20 puntos externos puede aumentar tiempo de espera y consumo de APIs. Usa este rango solo si realmente lo necesitas.")

        if st.button("Ejecutar ciclo autónomo", use_container_width=True):
            with st.spinner("Ejecutando ciclo autónomo..."):
                cycle_results = run_autonomous_cycle(
                    dataframe,
                    schema_mapping,
                    available_crews=int(available_crews_cycle),
                    use_weather_context=use_weather_context,
                    use_osm_context=use_osm_context,
                    max_external_points=int(max_external_points),
                    wifi_package=wifi_package if is_meraki_package_active() else None,
                )
                cycle_results["is_synthetic_data"] = synthetic_flag
                st.session_state[cycle_state_key] = cycle_results
                save_latest_operational_results(cycle_results, source="mission_control")
                st.session_state[quality_gate_key] = cycle_results.get("quality_gate_report", {})
                st.session_state[manual_audit_key] = append_audit_event(
                    manual_audit_log,
                    create_audit_event(
                        module="Mission Control",
                        action="Ejecutar ciclo autónomo desde UI",
                        status="ok",
                        message="El usuario ejecutó un nuevo ciclo autónomo.",
                        metadata={"trace_id": cycle_results.get("trace_id")},
                    ),
                )
                cycle_results = st.session_state[cycle_state_key]
                manual_audit_log = st.session_state[manual_audit_key]
                active_results = get_active_results(base_results, cycle_results, replay_state, synthetic_flag)

        if cycle_results:
            st.success("Ciclo autónomo disponible.")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Trace ID", cycle_results.get("trace_id") or "Sin trace")
            mc2.metric("Órdenes", len(cycle_results.get("work_orders", pd.DataFrame())))
            impact_df = cycle_results.get("impact_scores", pd.DataFrame())
            mc3.metric(
                "Top zona crítica",
                str(impact_df.iloc[0]["zona"]) if isinstance(impact_df, pd.DataFrame) and not impact_df.empty else "N/A",
            )
            mc4.metric("Confianza", cycle_results.get("confidence_level", "Baja"))

            st.markdown("**Quality gate resumido**")
            st.write(f"- Quality gate: **{cycle_results.get('quality_gate_report', {}).get('quality_gate', 'Sin evaluar')}**")
            st.write(f"- Demo readiness: **{cycle_results.get('quality_gate_report', {}).get('demo_readiness', 'Sin evaluar')}**")

            st.markdown("**Resumen ejecutivo base**")
            st.markdown(cycle_results.get("executive_summary", "Sin resumen disponible."))

            audit_summary = cycle_results.get("audit_summary", {})
            st.markdown("**Resumen de auditoría del ciclo**")
            au1, au2, au3, au4 = st.columns(4)
            au1.metric("Eventos", audit_summary.get("eventos_totales", 0))
            au2.metric("OK", audit_summary.get("eventos_ok", 0))
            au3.metric("Warnings", audit_summary.get("advertencias", 0))
            au4.metric("Errores", audit_summary.get("errores", 0))

            st.markdown("**Top zonas críticas**")
            if isinstance(impact_df, pd.DataFrame) and not impact_df.empty:
                st.dataframe(impact_df.head(10), use_container_width=True)
            else:
                st.info("No se generaron zonas críticas con el dataset actual.")

            st.markdown("**Plan de cuadrillas**")
            recommended_df = cycle_results.get("crew_plan", {}).get("recommended_zones", pd.DataFrame())
            if isinstance(recommended_df, pd.DataFrame) and not recommended_df.empty:
                st.dataframe(recommended_df, use_container_width=True)
            st.write(cycle_results.get("crew_plan", {}).get("explanation", ""))

            display_list("Limitaciones del ciclo", cycle_results.get("limitations", []), "Sin limitaciones registradas.")

            st.markdown("**Event log de agentes**")
            st.dataframe(pd.DataFrame(cycle_results.get("agent_event_log", [])), use_container_width=True)

            cycle_audit_df = audit_log_to_dataframe(cycle_results.get("audit_log", []))
            if not cycle_audit_df.empty:
                st.download_button(
                    "Descargar audit log del ciclo (CSV)",
                    data=dataframe_to_csv_bytes(cycle_audit_df),
                    file_name="operational_audit_log_cycle.csv",
                    mime="text/csv",
                    key="download_cycle_audit_log_csv",
                )


with tabs[3]:
    render_section_header(
        "Simulación Operativa con Dataset Cargado",
        "Este módulo procesa el dataset cargado en lotes, como si los registros llegaran progresivamente a un centro de monitoreo. No inventa datos; usa la base cargada y el mapeo de columnas definido por el usuario.",
    )
    st.caption(REPLAY_NOTE)

    if dataframe is None or dataframe.empty:
        st.info("Carga primero un dataset.")
    else:
        readiness = build_quality_gate_report(dataframe, schema_mapping, results=active_results)
        req1, req2, req3 = st.columns(3)
        req1.metric("Dataset cargado", "Sí")
        req2.metric("Zona mapeada", "Sí" if schema_mapping.get("zone_col") else "No")
        req3.metric("Readiness operativo", readiness["operational_readiness"]["operational_status"])

        sim_col1, sim_col2 = st.columns(2)
        batch_size = sim_col1.number_input("Tamaño de lote", min_value=1, value=10, step=1)
        replay_crews = sim_col2.number_input("Cuadrillas para la simulación", min_value=1, value=3, step=1)

        action_col1, action_col2, action_col3, action_col4 = st.columns(4)

        if action_col1.button("Preparar simulación", use_container_width=True):
            replay_package = prepare_replay_events(
                dataframe,
                schema_mapping,
                wifi_package=wifi_package if is_meraki_package_active() else None,
            )
            replay_trace_id = f"REPLAY-{get_timestamp()}"
            replay_state = {
                "trace_id": replay_trace_id,
                "events_df": replay_package["events_df"],
                "has_temporal_data": replay_package["has_temporal_data"],
                "warnings": replay_package["warnings"],
                "batch_size": int(batch_size),
                "available_crews": int(replay_crews),
                "current_step": 0,
                "total_steps": get_total_replay_steps(replay_package["events_df"], int(batch_size)),
                "history": [],
                "current_results": None,
                "last_changes": [],
                "timeline": pd.DataFrame(),
                "audit_log": [
                    create_audit_event(
                        module="Simulación Operativa",
                        action="Preparar simulación",
                        status="warning" if replay_package["warnings"] else "ok",
                        message="La simulación quedó lista para procesar lotes.",
                        metadata={
                            "trace_id": replay_trace_id,
                            "has_temporal_data": replay_package["has_temporal_data"],
                            "warnings": replay_package["warnings"],
                        },
                    )
                ],
                "audit_summary": {},
            }
            replay_state["audit_summary"] = build_operational_audit_summary(replay_state["audit_log"])
            st.session_state[replay_state_key] = replay_state
            replay_state = st.session_state[replay_state_key]

        if action_col2.button(
            "Avanzar un lote",
            use_container_width=True,
            disabled=replay_state is None or replay_state.get("current_step", 0) >= replay_state.get("total_steps", 0),
        ):
            replay_state = advance_replay_state(
                replay_state,
                schema_mapping,
                int(replay_crews),
                wifi_package=wifi_package if is_meraki_package_active() else None,
            )
            st.session_state[replay_state_key] = replay_state
            if replay_state.get("current_results"):
                save_latest_operational_results(replay_state, source="replay")

        if action_col3.button(
            "Ejecutar simulación completa",
            use_container_width=True,
            disabled=replay_state is None or replay_state.get("current_step", 0) >= replay_state.get("total_steps", 0),
        ):
            while replay_state and replay_state.get("current_step", 0) < replay_state.get("total_steps", 0):
                replay_state = advance_replay_state(
                    replay_state,
                    schema_mapping,
                    int(replay_crews),
                    wifi_package=wifi_package if is_meraki_package_active() else None,
                )
            st.session_state[replay_state_key] = replay_state
            if replay_state and replay_state.get("current_results"):
                save_latest_operational_results(replay_state, source="replay")

        if action_col4.button("Reiniciar simulación", use_container_width=True):
            st.session_state.pop(replay_state_key, None)
            replay_state = None

        replay_state = st.session_state.get(replay_state_key)
        active_results = get_active_results(base_results, cycle_results, replay_state, synthetic_flag)
        if replay_state:
            if not replay_state.get("has_temporal_data"):
                st.warning(
                    "El dataset no tiene columna de fecha mapeada. La simulación usa el orden de filas, no temporalidad real."
                )

            display_list("Advertencias de preparación", replay_state.get("warnings", []), "Sin advertencias de preparación.")

            total_rows = len(replay_state["events_df"])
            current_results = replay_state.get("current_results")
            current_summary = summarize_replay_state(current_results) if current_results else {}
            processed_rows = int(current_results.get("processed_rows", 0)) if current_results else 0
            processed_percentage = round((processed_rows / total_rows) * 100, 2) if total_rows else 0.0

            metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)
            metric1.metric("Filas procesadas", processed_rows)
            metric2.metric("Porcentaje procesado", f"{processed_percentage}%")
            metric3.metric("Órdenes generadas", current_summary.get("numero_ordenes", 0))
            metric4.metric("Zonas críticas", current_summary.get("numero_zonas_criticas", 0))
            metric5.metric("Zona más crítica", current_summary.get("zona_mas_critica") or "N/A")
            metric6.metric("Confianza", current_summary.get("nivel_confianza", "Baja"))

            st.write(f"- Trace de simulación: `{replay_state.get('trace_id')}`")
            st.write(f"- Lote actual: {replay_state.get('current_step', 0)} / {replay_state.get('total_steps', 0)}")
            st.write(f"- Acción sugerida: {current_summary.get('accion_sugerida', 'Preparar y avanzar la simulación.')}")

            display_list(
                "Cambios detectados frente al paso anterior",
                replay_state.get("last_changes", []),
                "Aún no hay cambios detectados.",
            )

            if current_results and current_results.get("warnings"):
                display_list("Warnings del análisis parcial", current_results.get("warnings", []), "Sin warnings.")

            timeline_df = replay_state.get("timeline", pd.DataFrame())
            if not timeline_df.empty:
                st.markdown("**Timeline de la simulación**")
                st.dataframe(timeline_df, use_container_width=True)

                timeline_fig = create_replay_timeline_chart(timeline_df)
                if timeline_fig is not None:
                    st.plotly_chart(
                        timeline_fig,
                        use_container_width=True,
                        key="replay_timeline_chart",
                    )

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(
                        create_work_order_status_chart(
                            work_orders_df=current_results.get("work_orders", pd.DataFrame()) if current_results else pd.DataFrame(),
                            review_queue_df=None,
                        ),
                        use_container_width=True,
                        key="replay_work_order_status_chart",
                    )
                with chart_col2:
                    latest_impact_df = current_results.get("impact_scores", pd.DataFrame()) if current_results else pd.DataFrame()
                    latest_impact_df = latest_impact_df if isinstance(latest_impact_df, pd.DataFrame) else pd.DataFrame()
                    st.plotly_chart(
                        create_priority_bar_chart(latest_impact_df),
                        use_container_width=True,
                        key="replay_priority_bar_chart",
                    )

            if replay_state.get("last_changes"):
                change_cols = st.columns(min(3, len(replay_state.get("last_changes", []))))
                for index, change in enumerate(replay_state.get("last_changes", [])[:3]):
                    with change_cols[index]:
                        render_insight_card("Cambio detectado", str(change), status="warning")


with tabs[5]:
    render_section_header(
        "Portal Ciudadano",
        "Consulta el estado estimado de las zonas WiFi publicas y recibe recomendaciones de conexion basadas en datos agregados.",
    )
    st.caption("Estos resultados son estimaciones agregadas por AP, zona y hora. No rastrean personas ni exponen identificadores individuales.")

    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin portal ciudadano disponible", "Carga un dataset y genera resultados operativos primero.")
    else:
        citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame())
        citizen_zone_summary_df = citizen_bundle.get("zone_summary", pd.DataFrame())
        citizen_recommendations_df = citizen_bundle.get("recommendations", pd.DataFrame())
        citizen_alerts_df = citizen_bundle.get("alerts", pd.DataFrame())
        hourly_patterns_df = citizen_bundle.get("hourly_patterns", pd.DataFrame())
        citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
        citizen_zone_summary_df = citizen_zone_summary_df if isinstance(citizen_zone_summary_df, pd.DataFrame) else pd.DataFrame()
        citizen_recommendations_df = citizen_recommendations_df if isinstance(citizen_recommendations_df, pd.DataFrame) else pd.DataFrame()
        citizen_alerts_df = citizen_alerts_df if isinstance(citizen_alerts_df, pd.DataFrame) else pd.DataFrame()
        hourly_patterns_df = hourly_patterns_df if isinstance(hourly_patterns_df, pd.DataFrame) else pd.DataFrame()

        if citizen_scores_df.empty:
            render_empty_state("Sin experiencia ciudadana calculada", "El dataset actual no tiene suficiente evidencia agregada para recomendaciones de usuario.")
        else:
            summary_cols = st.columns(3)
            with summary_cols[0]:
                render_dataframe_clean(citizen_recommendations_df.head(5), title="Mejores zonas", height=220)
            with summary_cols[1]:
                if not citizen_alerts_df.empty and "tipo_alerta" in citizen_alerts_df.columns:
                    alert_filter = citizen_alerts_df["tipo_alerta"].astype(str).isin(["Zona inestable", "Disponibilidad baja", "Desconexiones altas"])
                    render_dataframe_clean(citizen_alerts_df[alert_filter].head(5), title="Zonas en observacion", height=220)
                else:
                    render_empty_state("Sin alertas ciudadanas", "No se registran alertas agregadas por ahora.")
            with summary_cols[2]:
                unstable_df = citizen_scores_df[citizen_scores_df["citizen_status"].astype(str).eq("Inestable")].head(5) if "citizen_status" in citizen_scores_df.columns else pd.DataFrame()
                render_dataframe_clean(unstable_df, title="Zonas inestables", height=220)

            st.markdown("### Donde conectarme")
            zone_options = ["Todas"] + sorted(citizen_zone_summary_df["zona"].astype(str).unique().tolist()) if not citizen_zone_summary_df.empty and "zona" in citizen_zone_summary_df.columns else ["Todas"]
            time_options = ["Sin preferencia"]
            if not hourly_patterns_df.empty and "hour" in hourly_patterns_df.columns:
                time_options += [f"{int(hour):02d}:00" for hour in sorted(hourly_patterns_df["hour"].dropna().unique().tolist())]
            portal_col1, portal_col2 = st.columns(2)
            with portal_col1:
                preferred_zone = st.selectbox("Zona de interes", zone_options, key="citizen_portal_zone")
            with portal_col2:
                preferred_time = st.selectbox("Horario deseado", time_options, key="citizen_portal_time")

            citizen_recommend_key = build_state_key("citizen_recommendations", file_signature, mapping_signature)
            if st.button("Recomendar zonas", use_container_width=True, key="citizen_recommend_button"):
                st.session_state[citizen_recommend_key] = recommend_best_wifi_zones(
                    citizen_scores_df,
                    user_zone=None if preferred_zone == "Todas" else preferred_zone,
                    time_preference=None if preferred_time == "Sin preferencia" else preferred_time,
                    top_n=5,
                )

            selected_recommendations_df = st.session_state.get(citizen_recommend_key)
            if not isinstance(selected_recommendations_df, pd.DataFrame) or selected_recommendations_df.empty:
                selected_recommendations_df = citizen_recommendations_df.head(5)
            render_dataframe_clean(selected_recommendations_df, title="Ranking recomendado para usuarios", height=260)

            st.markdown("### Mejores horarios")
            if not hourly_patterns_df.empty:
                if "hour" in hourly_patterns_df.columns:
                    hourly_summary = hourly_patterns_df.groupby("hour", dropna=False)[["avg_connections", "avg_disconnection_rate"]].mean().reset_index()
                    st.plotly_chart(
                        px.line(
                            hourly_summary,
                            x="hour",
                            y=["avg_connections", "avg_disconnection_rate"],
                            title="Actividad y estabilidad promedio por hora",
                        ),
                        use_container_width=True,
                        key="citizen_hourly_line",
                    )
                if {"day_name", "hour", "avg_connections"}.issubset(hourly_patterns_df.columns):
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    heatmap_df = hourly_patterns_df.groupby(["day_name", "hour"], dropna=False)["avg_connections"].mean().reset_index()
                    heatmap_df["day_name"] = pd.Categorical(heatmap_df["day_name"], categories=day_order, ordered=True)
                    heatmap_pivot = heatmap_df.pivot(index="day_name", columns="hour", values="avg_connections").fillna(0)
                    st.plotly_chart(
                        px.imshow(
                            heatmap_pivot,
                            aspect="auto",
                            title="Uso promedio por dia y hora",
                            labels={"x": "Hora", "y": "Dia", "color": "Conexiones prom."},
                        ),
                        use_container_width=True,
                        key="citizen_hourly_heatmap",
                    )
            else:
                render_empty_state("Sin patron horario", "No hay suficiente evidencia horaria para mostrar mejores franjas de conexion.")

            st.markdown("### Alertas ciudadanas")
            render_dataframe_clean(citizen_alerts_df, height=260)
            st.info("Estos resultados son estimaciones con datos agregados. No representan personas individuales ni monitoreo en vivo real.")


with tabs[6]:
    render_section_header(
        "Experiencia Ciudadana",
        "Score agregado para aproximar estabilidad, disponibilidad, capacidad percibida y actividad observada por AP o zona.",
    )
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin experiencia ciudadana", "Carga un dataset y genera resultados operativos primero.")
    else:
        citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame())
        citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
        hourly_patterns_df = citizen_bundle.get("hourly_patterns", pd.DataFrame())
        hourly_patterns_df = hourly_patterns_df if isinstance(hourly_patterns_df, pd.DataFrame) else pd.DataFrame()
        zone_summary_df = citizen_bundle.get("zone_summary", pd.DataFrame())
        zone_summary_df = zone_summary_df if isinstance(zone_summary_df, pd.DataFrame) else pd.DataFrame()

        if citizen_scores_df.empty:
            render_empty_state("Sin scores ciudadanos", "No fue posible calcular Citizen Experience Score con la evidencia actual.")
        else:
            avg_citizen_score = round(float(pd.to_numeric(citizen_scores_df["citizen_experience_score"], errors="coerce").mean()), 2)
            best_ap = str(citizen_scores_df.iloc[0]["ap_name"]) if "ap_name" in citizen_scores_df.columns else "N/A"
            unstable_count = int(citizen_scores_df["citizen_status"].astype(str).eq("Inestable").sum()) if "citizen_status" in citizen_scores_df.columns else 0
            state_cols = st.columns(4)
            state_cols[0].metric("Citizen Experience promedio", avg_citizen_score)
            state_cols[1].metric("AP mejor valorado", best_ap)
            state_cols[2].metric("AP / zonas inestables", unstable_count)
            state_cols[3].metric("Registros evaluados", len(citizen_scores_df))

            chart_row_1 = st.columns(2)
            best_scores_df = citizen_scores_df.sort_values("citizen_experience_score", ascending=False).head(10)
            worst_scores_df = citizen_scores_df.sort_values("citizen_experience_score", ascending=True).head(10)
            with chart_row_1[0]:
                st.plotly_chart(
                    px.bar(best_scores_df, x="citizen_experience_score", y="ap_name", color="citizen_status", orientation="h", title="Top AP / zona con mejor experiencia"),
                    use_container_width=True,
                    key="citizen_best_bar",
                )
            with chart_row_1[1]:
                st.plotly_chart(
                    px.bar(worst_scores_df, x="citizen_experience_score", y="ap_name", color="citizen_status", orientation="h", title="AP / zona con experiencia inestable"),
                    use_container_width=True,
                    key="citizen_worst_bar",
                )

            chart_row_2 = st.columns(2)
            with chart_row_2[0]:
                status_df = citizen_scores_df["citizen_status"].fillna("Sin evidencia suficiente").value_counts().reset_index()
                status_df.columns = ["citizen_status", "count"]
                st.plotly_chart(
                    px.pie(status_df, values="count", names="citizen_status", hole=0.55, title="Distribucion de estados ciudadanos"),
                    use_container_width=True,
                    key="citizen_status_donut",
                )
            with chart_row_2[1]:
                if {"day_name", "hour", "avg_disconnection_rate"}.issubset(hourly_patterns_df.columns):
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    instability_df = hourly_patterns_df.groupby(["day_name", "hour"], dropna=False)["avg_disconnection_rate"].mean().reset_index()
                    instability_df["day_name"] = pd.Categorical(instability_df["day_name"], categories=day_order, ordered=True)
                    instability_pivot = instability_df.pivot(index="day_name", columns="hour", values="avg_disconnection_rate").fillna(0)
                    st.plotly_chart(
                        px.imshow(
                            instability_pivot,
                            aspect="auto",
                            title="Heatmap de inestabilidad por dia y hora",
                            labels={"x": "Hora", "y": "Dia", "color": "Desconexion prom."},
                        ),
                        use_container_width=True,
                        key="citizen_instability_heatmap",
                    )
                else:
                    render_empty_state("Sin heatmap horario", "No hay suficiente historial horario para la matriz de dia y hora.")

            if is_meraki_package_active() and isinstance(wifi_package, dict) and isinstance(wifi_package.get("hourly_metrics"), pd.DataFrame):
                st.markdown("### Calendario ciudadano")
                st.caption("Usa Nager.Date con cache para comparar festivos y fines de semana. Solo se consulta al presionar el boton.")
                if st.button("Activar calendario ciudadano", use_container_width=True, key="citizen_calendar_button"):
                    with st.spinner("Consultando calendario publico de Colombia..."):
                        calendar_context = enrich_hourly_with_public_calendar(wifi_package["hourly_metrics"])
                        st.session_state[manual_public_calendar_context_key] = {
                            "dataset_signature": file_signature,
                            "data": calendar_context,
                        }
                        st.session_state[manual_audit_key] = append_audit_event(
                            manual_audit_log,
                            create_audit_event(
                                module="Experiencia Ciudadana",
                                action="Enriquecer calendario publico",
                                status="ok" if not calendar_context.empty else "warning",
                                message="Se cargo calendario publico para analisis ciudadano.",
                                metadata={"rows": int(len(calendar_context))},
                            ),
                        )
                        st.success("Calendario ciudadano actualizado.")
                        st.rerun()

                if isinstance(citizen_calendar_context_df, pd.DataFrame) and not citizen_calendar_context_df.empty:
                    calendar_summary = citizen_bundle.get("calendar_summary", {})
                    cal_cols = st.columns(3)
                    cal_cols[0].metric("Conexiones prom. en festivos", calendar_summary.get("avg_connections_holidays", 0.0))
                    cal_cols[1].metric("Conexiones prom. no festivos", calendar_summary.get("avg_connections_non_holidays", 0.0))
                    cal_cols[2].metric("Desconexion prom. en festivos", calendar_summary.get("avg_disconnection_holidays", 0.0))
                    render_dataframe_clean(citizen_calendar_context_df.head(20), title="Muestra enriquecida con calendario publico", height=260)

            render_dataframe_clean(zone_summary_df, title="Resumen por zona", height=240)
            render_dataframe_clean(citizen_scores_df, title="Tabla completa de Citizen Experience Score", height=320)


with tabs[7]:
    render_section_header(
        "Recomendador de Zonas WiFi",
        "Sugiere dónde y cuándo conectarse usando experiencia agregada, estabilidad y patrones horarios.",
    )
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin recomendador disponible", "Carga un dataset y genera resultados operativos primero.")
    else:
        citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame())
        citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
        citizen_zone_summary_df = citizen_bundle.get("zone_summary", pd.DataFrame())
        citizen_zone_summary_df = citizen_zone_summary_df if isinstance(citizen_zone_summary_df, pd.DataFrame) else pd.DataFrame()
        hourly_patterns_df = citizen_bundle.get("hourly_patterns", pd.DataFrame())
        hourly_patterns_df = hourly_patterns_df if isinstance(hourly_patterns_df, pd.DataFrame) else pd.DataFrame()

        if citizen_scores_df.empty:
            render_empty_state("Sin evidencia suficiente", "No hay Citizen Experience Score disponible para recomendar zonas.")
        else:
            zone_options = ["Top general"]
            if not citizen_zone_summary_df.empty and "zona" in citizen_zone_summary_df.columns:
                zone_options += sorted(citizen_zone_summary_df["zona"].astype(str).dropna().unique().tolist())
            time_options = ["Sin preferencia"]
            if not hourly_patterns_df.empty and "hour" in hourly_patterns_df.columns:
                time_options += [f"{int(hour):02d}:00" for hour in sorted(hourly_patterns_df["hour"].dropna().unique().tolist())]

            recommender_col1, recommender_col2 = st.columns(2)
            with recommender_col1:
                user_zone_preference = st.selectbox("Zona de referencia", zone_options, key="citizen_recommender_zone")
            with recommender_col2:
                user_time_preference = st.selectbox("Preferencia horaria", time_options, key="citizen_recommender_time")

            recommender_state_key = build_state_key("citizen_wifi_recommender", file_signature, mapping_signature)
            if st.button("Calcular ranking ciudadano", use_container_width=True, key="citizen_recommender_button"):
                st.session_state[recommender_state_key] = recommend_best_wifi_zones(
                    citizen_scores_df,
                    user_zone=None if user_zone_preference == "Top general" else user_zone_preference,
                    time_preference=None if user_time_preference == "Sin preferencia" else user_time_preference,
                    top_n=8,
                )

            recommender_df = st.session_state.get(recommender_state_key)
            if not isinstance(recommender_df, pd.DataFrame) or recommender_df.empty:
                recommender_df = recommend_best_wifi_zones(citizen_scores_df, top_n=8)

            render_dataframe_clean(recommender_df, title="Ranking recomendado para conectarse", height=320)
            if not recommender_df.empty and {"ap_name", "score"}.issubset(recommender_df.columns):
                st.plotly_chart(
                    px.bar(
                        recommender_df.head(8),
                        x="score",
                        y="ap_name",
                        color="estado" if "estado" in recommender_df.columns else None,
                        orientation="h",
                        title="Top zonas/AP recomendados",
                    ),
                    use_container_width=True,
                    key="citizen_recommender_bar",
                )
            st.caption("La recomendación usa datos agregados. No estima experiencia individual ni rastrea personas.")


with tabs[8]:
    render_section_header("Buzon Ciudadano", "Recibe reportes anonimos para mejorar el servicio sin recolectar datos personales.")
    st.warning("No ingreses datos personales. Este reporte es anonimo y se usa solo para mejorar el servicio.")

    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin buzon disponible", "Carga un dataset y genera resultados primero para seleccionar zonas o APs.")
    else:
        citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame())
        citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
        selectable_targets = sorted(
            set(citizen_scores_df.get("ap_name", pd.Series(dtype="object")).astype(str).tolist())
            | set(citizen_scores_df.get("zone_name", pd.Series(dtype="object")).astype(str).tolist())
        )
        selectable_targets = [item for item in selectable_targets if item] or ["Sin referencia disponible"]

        with st.form("citizen_feedback_form", clear_on_submit=True):
            feedback_zone = st.selectbox("Zona o AP", selectable_targets, key="citizen_feedback_zone")
            feedback_rating = st.slider("Calificacion del servicio", min_value=1, max_value=5, value=3, key="citizen_feedback_rating")
            feedback_issue = st.selectbox(
                "Problema principal",
                ["sin_categoria", "no_conecta", "conexion_inestable", "lentitud", "cobertura", "experiencia_buena", "otro"],
                key="citizen_feedback_issue",
            )
            feedback_comment = st.text_area("Comentario opcional", key="citizen_feedback_comment", placeholder="Describe brevemente tu experiencia sin incluir datos personales.")
            submitted_feedback = st.form_submit_button("Enviar reporte", use_container_width=True)

        if submitted_feedback:
            updated_feedback_df = save_citizen_feedback(
                zone_or_ap=feedback_zone,
                rating=int(feedback_rating),
                issue_type=feedback_issue,
                comment=feedback_comment,
                source="citizen_portal",
            )
            st.session_state[manual_audit_key] = append_audit_event(
                manual_audit_log,
                create_audit_event(
                    module="Buzon Ciudadano",
                    action="Registrar feedback anonimo",
                    status="ok",
                    message="Se registro un reporte ciudadano anonimo.",
                    metadata={"zone_or_ap": feedback_zone, "issue_type": feedback_issue},
                ),
            )
            st.success("Reporte anonimo guardado correctamente.")
            citizen_bundle["feedback_df"] = updated_feedback_df
            citizen_bundle["feedback_summary"] = summarize_citizen_feedback(updated_feedback_df)
            st.rerun()

        feedback_df = citizen_bundle.get("feedback_df", pd.DataFrame())
        feedback_df = feedback_df if isinstance(feedback_df, pd.DataFrame) else pd.DataFrame()
        feedback_summary = citizen_bundle.get("feedback_summary", {})
        feedback_metrics = st.columns(4)
        feedback_metrics[0].metric("Total reportes", feedback_summary.get("total_reportes", 0))
        feedback_metrics[1].metric("Rating promedio", feedback_summary.get("rating_promedio", 0.0))
        feedback_metrics[2].metric("Sentimiento general", feedback_summary.get("sentimiento_general", "Sin reportes"))
        feedback_metrics[3].metric("Zonas/AP con mas reportes", len(feedback_summary.get("zonas_con_mas_reportes", [])))

        top_issue_df = pd.DataFrame(feedback_summary.get("problemas_mas_frecuentes", []))
        top_zone_feedback_df = pd.DataFrame(feedback_summary.get("zonas_con_mas_reportes", []))
        latest_comments_df = pd.DataFrame(feedback_summary.get("ultimos_comentarios_anonimos", []))
        feedback_cols = st.columns(3)
        with feedback_cols[0]:
            render_dataframe_clean(top_issue_df, title="Problemas mas frecuentes", height=220)
        with feedback_cols[1]:
            render_dataframe_clean(top_zone_feedback_df, title="Zonas/AP con mas reportes", height=220)
        with feedback_cols[2]:
            render_dataframe_clean(latest_comments_df, title="Ultimos comentarios anonimos", height=220)

        render_dataframe_clean(feedback_df.tail(20), title="Historial reciente del buzon", height=260)


with tabs[9]:
    render_section_header("Equidad Digital", "Proxy responsable para señalar donde podria existir necesidad de mejora en acceso, calidad o evidencia.")
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin proxy de equidad", "Carga un dataset y genera resultados operativos primero.")
    else:
        digital_equity_df = citizen_bundle.get("digital_equity", pd.DataFrame())
        digital_equity_df = digital_equity_df if isinstance(digital_equity_df, pd.DataFrame) else pd.DataFrame()
        if digital_equity_df.empty:
            render_empty_state("Sin proxy disponible", "No hay suficiente evidencia agregada para calcular Digital Equity Proxy.")
        else:
            need_improvement_df = digital_equity_df[digital_equity_df["equity_label"].astype(str).eq("Alta necesidad de mejora")].head(10)
            risk_demand_df = digital_equity_df[digital_equity_df["equity_label"].astype(str).eq("Alta demanda con riesgo operativo")].head(10)
            low_evidence_df = digital_equity_df[digital_equity_df["equity_label"].astype(str).eq("Baja evidencia / requiere validación")].head(10)
            eq_cols = st.columns(3)
            with eq_cols[0]:
                render_dataframe_clean(need_improvement_df, title="Alta necesidad de mejora", height=240)
            with eq_cols[1]:
                render_dataframe_clean(risk_demand_df, title="Alta demanda con riesgo", height=240)
            with eq_cols[2]:
                render_dataframe_clean(low_evidence_df, title="Baja evidencia", height=240)

            st.plotly_chart(
                px.bar(digital_equity_df.head(15), x="zone_name", y="digital_equity_proxy", color="equity_label", title="Digital Equity Proxy por zona"),
                use_container_width=True,
                key="digital_equity_bar",
            )
            render_dataframe_clean(digital_equity_df, title="Tabla completa de equidad digital proxy", height=320)
            st.info("Este indicador es un proxy de equidad digital. No usa poblacion real ni confirma brechas estructurales por si solo.")


with tabs[10]:
    render_section_header(
        "Retorno Social de Conectividad",
        "Cruza desempeño de la red WiFi con indicadores socioeconómicos agregados para priorizar mejoras donde la conectividad puede generar mayor impacto público.",
    )
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin módulo social disponible", "Carga un dataset y genera resultados operativos primero.")
    else:
        st.caption(
            "Este módulo acepta datos socioeconómicos agregados por zona, comuna, barrio, corregimiento, manzana o municipio. "
            "No procesa datos personales ni fichas individuales de SISBÉN."
        )

        socio_local_candidates = [
            path for path in list_local_datasets()
            if Path(path).suffix.lower() in {".csv", ".xlsx", ".xls", ".txt"}
        ]
        socio_upload = st.file_uploader(
            "Cargar archivo socioeconómico (CSV/XLSX)",
            type=["csv", "xlsx", "xls", "txt"],
            key="socioeconomic_file_uploader",
        )
        socio_source_mode = st.radio(
            "Origen del dataset socioeconómico",
            ["Archivo cargado", "Archivo local", "URL pública", "Socrata / datos.gov.co"],
            horizontal=True,
            key="socioeconomic_source_mode",
        )

        socio_input_value: object = socio_upload
        if socio_source_mode == "Archivo local":
            socio_selected_path = st.selectbox(
                "Archivo local disponible",
                options=socio_local_candidates or ["Sin archivos disponibles"],
                key="socioeconomic_local_selector",
            )
            socio_input_value = None if socio_selected_path == "Sin archivos disponibles" else socio_selected_path
        elif socio_source_mode == "URL pública":
            socio_input_value = st.text_input(
                "URL pública del archivo socioeconómico",
                key="socioeconomic_public_url",
                placeholder="https://...",
            ).strip()
        elif socio_source_mode == "Socrata / datos.gov.co":
            socrata_col1, socrata_col2 = st.columns(2)
            with socrata_col1:
                socrata_domain = st.text_input(
                    "Dominio Socrata",
                    value="www.datos.gov.co",
                    key="socioeconomic_socrata_domain",
                ).strip()
            with socrata_col2:
                socrata_dataset_id = st.text_input(
                    "Dataset ID",
                    key="socioeconomic_socrata_id",
                    placeholder="abcd-1234",
                ).strip()
            socio_input_value = {"domain": socrata_domain, "dataset_id": socrata_dataset_id}

        metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
        with metadata_col1:
            st.caption("Fuentes sugeridas")
            st.markdown(f"- [DANE IPM]({get_dane_ipm_metadata().get('url')})")
        with metadata_col2:
            st.markdown(f"- [DANE NBI]({get_dane_nbi_metadata().get('url')})")
        with metadata_col3:
            st.markdown(f"- [SISBÉN abierto]({get_sisben_open_data_metadata().get('url')})")

        if st.button("Cargar y validar dataset socioeconómico", use_container_width=True, key="socioeconomic_validate_button"):
            try:
                if socio_source_mode == "Socrata / datos.gov.co":
                    if not isinstance(socio_input_value, dict) or not socio_input_value.get("dataset_id"):
                        st.warning("Ingresa un dataset_id de Socrata para continuar.")
                    else:
                        socio_raw_df = fetch_socrata_dataset(
                            socio_input_value.get("domain", "www.datos.gov.co"),
                            socio_input_value.get("dataset_id", ""),
                            limit=5000,
                        )
                        socioeconomic_df = normalize_socioeconomic_columns(socio_raw_df)
                        socioeconomic_validation = validate_socioeconomic_dataset(socioeconomic_df)
                else:
                    socioeconomic_df = normalize_socioeconomic_columns(load_socioeconomic_file(socio_input_value))
                    socioeconomic_validation = validate_socioeconomic_dataset(socioeconomic_df)

                if not socioeconomic_df.empty:
                    st.session_state["socioeconomic_dataset_payload"] = {
                        "dataset_signature": file_signature,
                        "data": socioeconomic_df,
                        "validation": socioeconomic_validation,
                    }
                    st.success("Dataset socioeconómico cargado y validado.")
                    st.rerun()
            except Exception as exc:
                st.warning(f"No fue posible cargar el dataset socioeconómico: {exc}")

        validation_cols = st.columns(4)
        validation_cols[0].metric("Nivel geográfico", socioeconomic_validation.get("level", "Sin validar"))
        validation_cols[1].metric("Indicadores", len(socioeconomic_validation.get("available_indicators", [])))
        validation_cols[2].metric("Advertencias", len(socioeconomic_validation.get("warnings", [])))
        validation_cols[3].metric("Privacidad", len(socioeconomic_validation.get("privacy_warnings", [])))
        render_dataframe_clean(socioeconomic_df.head(15), title="Muestra socioeconómica cargada", height=240)
        render_dataframe_clean(
            pd.DataFrame({"indicador": socioeconomic_validation.get("available_indicators", [])}),
            title="Indicadores disponibles",
            height=160,
        )
        render_dataframe_clean(
            pd.DataFrame({"advertencia": socioeconomic_validation.get("warnings", [])}),
            title="Advertencias de calidad",
            height=160,
        )
        render_dataframe_clean(
            pd.DataFrame({"privacidad": socioeconomic_validation.get("privacy_warnings", [])}),
            title="Advertencias de privacidad",
            height=160,
        )

        if st.button("Calcular Retorno Social de Conectividad", use_container_width=True, key="social_roi_calculate_button"):
            social_roi_bundle = build_social_roi_bundle(
                active_results,
                citizen_bundle,
                socioeconomic_df=socioeconomic_df,
                socioeconomic_validation=socioeconomic_validation,
            )
            active_results["socioeconomic_validation"] = socioeconomic_validation
            active_results["social_roi_scores"] = social_roi_bundle.get("social_roi_scores", pd.DataFrame())
            active_results["social_roi_recommendations"] = social_roi_bundle.get("recommendations", pd.DataFrame())
            sync_latest_operational_snapshot(source_hint="mixed")
            st.success("Retorno Social de Conectividad actualizado.")

        social_roi_scores_df = social_roi_bundle.get("social_roi_scores", pd.DataFrame()) if isinstance(social_roi_bundle, dict) else pd.DataFrame()
        social_roi_scores_df = social_roi_scores_df if isinstance(social_roi_scores_df, pd.DataFrame) else pd.DataFrame()
        social_roi_recommendations_df = social_roi_bundle.get("recommendations", pd.DataFrame()) if isinstance(social_roi_bundle, dict) else pd.DataFrame()
        social_roi_recommendations_df = social_roi_recommendations_df if isinstance(social_roi_recommendations_df, pd.DataFrame) else pd.DataFrame()

        if social_roi_scores_df.empty:
            render_empty_state(
                "Sin Social ROI calculado",
                "Carga un dataset socioeconómico agregado y pulsa el botón de cálculo para activar esta vista.",
            )
        else:
            top_three = social_roi_scores_df.sort_values("social_roi_score", ascending=False).head(3)
            top_cards = st.columns(min(3, len(top_three)))
            for card_index, (_, card_row) in enumerate(top_three.iterrows()):
                with top_cards[card_index]:
                    render_action_card(
                        str(card_row.get("zone_name", "Zona")),
                        f"{card_row.get('social_roi_label', 'Sin clasificar')} | Score {float(card_row.get('social_roi_score', 0)):.2f}",
                        "media",
                    )

            render_dataframe_clean(
                social_roi_scores_df.sort_values("social_roi_score", ascending=False),
                title="Ranking de retorno social de conectividad",
                height=320,
            )
            st.plotly_chart(
                px.bar(
                    social_roi_scores_df.sort_values("social_roi_score", ascending=False).head(15),
                    x="zone_name",
                    y="social_roi_score",
                    color="social_roi_label",
                    title="Top zonas por retorno social esperado",
                ),
                use_container_width=True,
                key="social_roi_bar_chart",
            )
            st.plotly_chart(
                px.scatter(
                    social_roi_scores_df,
                    x="socioeconomic_vulnerability_score",
                    y="network_risk_score",
                    size="citizen_potential_score",
                    color="social_roi_label",
                    hover_name="zone_name",
                    title="Vulnerabilidad agregada vs riesgo de red",
                ),
                use_container_width=True,
                key="social_roi_scatter",
            )
            render_dataframe_clean(
                social_roi_recommendations_df,
                title="Recomendaciones de infraestructura y acompañamiento social",
                height=240,
            )
            if social_roi_bundle.get("limitations"):
                st.warning(" | ".join(str(item) for item in social_roi_bundle.get("limitations", [])[:4]))

            if st.button("Explicar con Gemini", use_container_width=True, key="social_roi_explain_button"):
                social_roi_context = build_social_roi_context(
                    social_roi_scores_df,
                    socioeconomic_validation,
                    social_roi_bundle.get("limitations", []),
                )
                social_roi_explanation_markdown = (
                    generate_social_roi_explanation_with_gemini(social_roi_context)
                    if is_gemini_configured()
                    else fallback_social_roi_explanation(social_roi_context)
                )
                st.session_state["social_roi_explanation_payload"] = {
                    "dataset_signature": file_signature,
                    "markdown": social_roi_explanation_markdown,
                }
                active_results["social_roi_explanation_markdown"] = social_roi_explanation_markdown
                sync_latest_operational_snapshot(source_hint="mixed")
                st.success(
                    "Explicación de retorno social generada con Gemini."
                    if is_gemini_configured()
                    else "Gemini no está configurado. Se generó fallback determinístico."
                )

            if social_roi_explanation_markdown:
                st.markdown(social_roi_explanation_markdown)


with tabs[17]:
    render_section_header(
        "Agente Ciudadano",
        "Explica en lenguaje claro que zonas lucen mas estables para conectarse y que deberia revisar la Alcaldia con base en datos agregados.",
    )
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin analisis ciudadano", "Carga un dataset y genera resultados operativos primero.")
    else:
        citizen_context = build_citizen_insights_context(
            citizen_bundle.get("citizen_scores", pd.DataFrame()),
            citizen_bundle.get("recommendations", pd.DataFrame()),
            citizen_bundle.get("feedback_summary", {}),
            citizen_bundle.get("digital_equity", pd.DataFrame()),
            citizen_bundle.get("calendar_summary", {}),
        )

        if st.button("Generar analisis ciudadano con Gemini", use_container_width=True, key="citizen_insights_button"):
            citizen_insights_markdown = (
                generate_citizen_insights_with_gemini(citizen_context)
                if is_gemini_configured()
                else fallback_citizen_insights(citizen_context)
            )
            st.session_state[citizen_insights_key] = {
                "dataset_signature": file_signature,
                "markdown": citizen_insights_markdown,
            }
            st.success(
                "Analisis ciudadano actualizado con Gemini."
                if is_gemini_configured()
                else "Gemini no esta configurado. Se genero fallback deterministico."
            )

        if not citizen_insights_markdown:
            citizen_insights_markdown = fallback_citizen_insights(citizen_context)
        st.markdown(citizen_insights_markdown)


with tabs[18]:
    render_section_header(
        "Vista Publica de Calidad",
        "Vista simplificada para ciudadanía y equipos de territorio con el estado agregado de calidad, alertas y recomendaciones de uso.",
    )
    if dataframe is None or dataframe.empty or not isinstance(citizen_bundle, dict):
        render_empty_state("Sin vista pública disponible", "Carga un dataset y genera resultados operativos primero.")
    else:
        citizen_scores_df = citizen_bundle.get("citizen_scores", pd.DataFrame())
        citizen_scores_df = citizen_scores_df if isinstance(citizen_scores_df, pd.DataFrame) else pd.DataFrame()
        citizen_alerts_df = citizen_bundle.get("alerts", pd.DataFrame())
        citizen_alerts_df = citizen_alerts_df if isinstance(citizen_alerts_df, pd.DataFrame) else pd.DataFrame()
        digital_equity_df = citizen_bundle.get("digital_equity", pd.DataFrame())
        digital_equity_df = digital_equity_df if isinstance(digital_equity_df, pd.DataFrame) else pd.DataFrame()
        social_roi_scores_df = social_roi_bundle.get("social_roi_scores", pd.DataFrame()) if isinstance(social_roi_bundle, dict) else pd.DataFrame()
        social_roi_scores_df = social_roi_scores_df if isinstance(social_roi_scores_df, pd.DataFrame) else pd.DataFrame()

        if citizen_scores_df.empty:
            render_empty_state("Sin calidad pública disponible", "No hay suficiente evidencia agregada para publicar un estado estimado.")
        else:
            public_cols = st.columns(4)
            public_cols[0].metric(
                "Experience promedio",
                round(float(pd.to_numeric(citizen_scores_df.get("citizen_experience_score"), errors="coerce").mean()), 2),
            )
            public_cols[1].metric(
                "Zonas con buena experiencia",
                int(citizen_scores_df["citizen_status"].astype(str).isin(["Excelente", "Buena"]).sum())
                if "citizen_status" in citizen_scores_df.columns
                else 0,
            )
            public_cols[2].metric("Alertas activas", len(citizen_alerts_df))
            public_cols[3].metric("Top Social ROI", int(len(social_roi_scores_df)))

            public_view_cols = st.columns(2)
            with public_view_cols[0]:
                render_dataframe_clean(
                    citizen_scores_df.sort_values("citizen_experience_score", ascending=False).head(10),
                    title="Zonas con mejor experiencia estimada",
                    height=260,
                )
            with public_view_cols[1]:
                render_dataframe_clean(
                    citizen_alerts_df.head(10),
                    title="Alertas y precauciones para usuarios",
                    height=260,
                )

            if not social_roi_scores_df.empty:
                st.plotly_chart(
                    px.bar(
                        social_roi_scores_df.sort_values("social_roi_score", ascending=False).head(10),
                        x="zone_name",
                        y="social_roi_score",
                        color="social_roi_label",
                        title="Zonas con mayor retorno social de conectividad",
                    ),
                    use_container_width=True,
                    key="public_social_roi_bar",
                )
            if not digital_equity_df.empty:
                render_dataframe_clean(digital_equity_df.head(10), title="Señales de equidad digital", height=220)
            st.info(
                "Esta vista usa datos agregados por zona/AP/hora. No representa personas individuales ni sustituye verificación en campo."
            )


with tabs[19]:
    render_section_header(
        "Validación Humana",
        "El agente recomienda y prioriza, pero un operador responsable debe revisar, aprobar o escalar las órdenes antes de actuar en campo.",
    )

    if dataframe is None or dataframe.empty or active_results is None:
        render_empty_state(
            "Sin resultados operativos",
            "Carga un dataset y genera resultados operativos primero.",
        )
    else:
        source_orders = active_results.get("work_orders", pd.DataFrame())
        source_orders = source_orders if isinstance(source_orders, pd.DataFrame) else pd.DataFrame()

        if st.button("Crear o refrescar cola de revisión", use_container_width=True):
            review_queue = create_review_queue(source_orders)
            st.session_state[review_queue_key] = review_queue
            st.session_state[manual_audit_key] = append_audit_event(
                manual_audit_log,
                create_audit_event(
                    module="Validación Humana",
                    action="Crear cola de revisión",
                    status="ok",
                    message=f"Se preparó una cola con {len(review_queue)} órdenes.",
                ),
            )
            review_queue = st.session_state[review_queue_key]
            manual_audit_log = st.session_state[manual_audit_key]
            active_results["human_review_log"] = review_queue
            sync_latest_operational_snapshot(source_hint="mixed")

        review_queue = st.session_state.get(review_queue_key)
        if review_queue is None or review_queue.empty:
            render_empty_state(
                "No hay órdenes de trabajo disponibles.",
                "Ejecuta primero Mission Control o Simulación Operativa.",
            )
        else:
            summary = summarize_human_review(review_queue)
            render_metric_row(
                {
                    "Total órdenes": summary["total_ordenes"],
                    "Pendientes": summary["pendientes"],
                    "Aprobadas": summary["aprobadas"],
                    "Rechazadas": summary["rechazadas"],
                    "Requieren visita": summary["requiere_visita"],
                    "Cerradas": summary["cerradas"],
                    "% revisado": f"{summary['porcentaje_revisado']}%",
                }
            )

            review_display_df = review_queue.rename(
                columns={
                    "order_id": "ID orden",
                    "zona": "Zona",
                    "tipo_alerta": "Tipo de alerta",
                    "prioridad": "Prioridad",
                    "evidencia": "Evidencia",
                    "accion_recomendada": "Acción recomendada",
                    "nivel_confianza": "Nivel de confianza",
                    "estado_revision": "Estado revisión",
                    "comentario_operador": "Comentario operador",
                    "reviewed_at": "Revisado en",
                }
            )
            render_dataframe_clean(review_display_df, title="Cola de revisión operativa", height=320)

            st.markdown("**Acciones masivas**")
            bulk_col1, bulk_col2 = st.columns([1, 2])
            bulk_status = bulk_col1.selectbox(
                "Estado masivo",
                ["pendiente", "aprobada", "rechazada", "requiere_visita", "cerrada"],
                key="bulk_review_status",
            )
            bulk_comment = bulk_col2.text_input("Comentario masivo", key="bulk_review_comment")
            apply_all_orders = st.checkbox("Aplicar a todas las órdenes", value=True, key="bulk_apply_all_orders")
            only_pending_orders = st.checkbox(
                "Aplicar solo a órdenes pendientes cuando sea masivo",
                value=False,
                key="bulk_only_pending_orders",
            )
            selected_order_ids_for_bulk: list[str] = []
            if not apply_all_orders:
                selected_order_ids_for_bulk = st.multiselect(
                    "Selecciona órdenes específicas",
                    review_queue["order_id"].astype(str).tolist(),
                    key="bulk_selected_order_ids",
                )

            if st.button("Aplicar cambio masivo", use_container_width=True, key="bulk_update_review_orders"):
                try:
                    review_queue = bulk_update_work_orders(
                        review_queue,
                        bulk_status,
                        bulk_comment,
                        only_current_filter=only_pending_orders,
                        selected_ids=None if apply_all_orders else selected_order_ids_for_bulk,
                    )
                    st.session_state[review_queue_key] = review_queue
                    st.session_state[manual_audit_key] = append_audit_event(
                        manual_audit_log,
                        create_audit_event(
                            module="Validación Humana",
                            action="Aplicar cambio masivo",
                            status="ok",
                            message=f"Se aplicó estado masivo {bulk_status} sobre la cola de revisión.",
                            metadata={
                                "apply_all": apply_all_orders,
                                "selected_ids_count": len(selected_order_ids_for_bulk),
                                "only_pending": only_pending_orders,
                            },
                        ),
                    )
                    manual_audit_log = st.session_state[manual_audit_key]
                    active_results["human_review_log"] = review_queue
                    sync_latest_operational_snapshot(source_hint="mixed")
                    st.success("Cambio masivo aplicado.")
                except ValueError as error:
                    st.error(str(error))

            order_ids = review_queue["order_id"].astype(str).tolist()
            selected_order_id = st.selectbox("Seleccionar orden para revisar", order_ids)
            selected_order_row = review_queue[review_queue["order_id"].astype(str) == selected_order_id].iloc[0]
            detail_col1, detail_col2 = st.columns(2)
            detail_col1.write(f"**Zona:** {selected_order_row['zona']}")
            detail_col1.write(f"**Tipo de alerta:** {selected_order_row['tipo_alerta']}")
            detail_col1.write(f"**Prioridad sugerida:** {selected_order_row['prioridad']}")
            detail_col2.write(f"**Nivel de confianza:** {selected_order_row['nivel_confianza']}")
            detail_col2.write(f"**Estado actual:** {selected_order_row['estado_revision']}")
            detail_col2.write(f"**Acción recomendada:** {selected_order_row['accion_recomendada']}")
            st.write(f"**Evidencia:** {selected_order_row['evidencia']}")

            status_col, comment_col = st.columns([1, 2])
            new_status = status_col.selectbox(
                "Nuevo estado",
                ["pendiente", "aprobada", "rechazada", "requiere_visita", "cerrada"],
            )
            comment = comment_col.text_input("Comentario del operador")

            if st.button("Actualizar estado de la orden", use_container_width=True):
                try:
                    review_queue = update_work_order_status(review_queue, selected_order_id, new_status, comment)
                    st.session_state[review_queue_key] = review_queue
                    st.session_state[manual_audit_key] = append_audit_event(
                        manual_audit_log,
                        create_audit_event(
                            module="Validación Humana",
                            action="Actualizar estado de orden",
                            status="ok",
                            message=f"La orden {selected_order_id} pasó a estado {new_status}.",
                            metadata={"order_id": selected_order_id, "new_status": new_status},
                        ),
                    )
                    review_queue = st.session_state[review_queue_key]
                    manual_audit_log = st.session_state[manual_audit_key]
                    active_results["human_review_log"] = review_queue
                    sync_latest_operational_snapshot(source_hint="mixed")
                    st.success("Estado actualizado.")
                except ValueError as error:
                    st.error(str(error))

            st.download_button(
                "Descargar human_review_log.csv",
                data=dataframe_to_csv_bytes(export_review_log(review_queue)),
                file_name="human_review_log.csv",
                mime="text/csv",
                key="download_human_review_log_tab_csv",
            )


with tabs[20]:
    st.subheader("Blindaje Técnico")
    if dataframe is None or dataframe.empty:
        st.info("Carga un dataset primero.")
    else:
        if st.button("Ejecutar validación técnica", use_container_width=True):
            quality_gate_report = build_quality_gate_report(dataframe, schema_mapping, results=active_results)
            st.session_state[quality_gate_key] = quality_gate_report
            st.session_state[manual_audit_key] = append_audit_event(
                manual_audit_log,
                create_audit_event(
                    module="Blindaje Técnico",
                    action="Ejecutar validación técnica",
                    status="ok",
                    message=f"Quality gate calculado: {quality_gate_report.get('quality_gate')}.",
                    metadata={"demo_readiness": quality_gate_report.get("demo_readiness")},
                ),
            )
            quality_gate_report = st.session_state[quality_gate_key]
            manual_audit_log = st.session_state[manual_audit_key]

        quality_gate_report = st.session_state.get(quality_gate_key) or quality_gate_report
        if quality_gate_report:
            bt1, bt2 = st.columns(2)
            bt1.metric("Quality gate", quality_gate_report.get("quality_gate", "Sin evaluar"))
            bt2.metric("Demo readiness", quality_gate_report.get("demo_readiness", "Sin evaluar"))

            display_list(
                "Problemas críticos",
                quality_gate_report.get("critical_issues", []),
                "No se registraron problemas críticos.",
            )
            display_list(
                "Advertencias",
                quality_gate_report.get("warnings", []),
                "No se registraron advertencias.",
            )
            display_list(
                "Recomendaciones",
                quality_gate_report.get("recommendations", []),
                "No se registraron recomendaciones adicionales.",
            )

            with st.expander("Validación del schema mapping", expanded=False):
                st.json(quality_gate_report.get("schema_validation", {}))
            with st.expander("Readiness operativo", expanded=False):
                st.json(quality_gate_report.get("operational_readiness", {}))
            with st.expander("Validación de outputs", expanded=False):
                st.json(quality_gate_report.get("output_validation", {}))
        else:
            st.info("Ejecuta la validación técnica para construir el quality gate report.")


with tabs[21]:
    render_section_header(
        "Auditoría Operativa",
        "Cada decisión relevante queda registrada para trazabilidad, revisión y mejora continua.",
    )

    if dataframe is None or dataframe.empty:
        render_empty_state(
            "Sin actividad operativa",
            "Carga un dataset y ejecuta al menos un módulo operativo.",
        )
    else:
        replay_log = replay_state.get("audit_log", []) if replay_state else []
        cycle_log = cycle_results.get("audit_log", []) if cycle_results else []
        manual_log = st.session_state.get(manual_audit_key, [])
        combined_audit_log = combine_audit_logs(cycle_log, replay_log, manual_log)
        audit_summary = build_operational_audit_summary(combined_audit_log)

        render_metric_row(
            {
                "Eventos totales": audit_summary["eventos_totales"],
                "OK": audit_summary["eventos_ok"],
                "Warnings": audit_summary["advertencias"],
                "Errores": audit_summary["errores"],
                "Módulos ejecutados": len(audit_summary["modulos_ejecutados"]),
            }
        )

        if audit_summary["modulos_ejecutados"]:
            st.caption(f"Módulos ejecutados: {', '.join(audit_summary['modulos_ejecutados'])}")

        audit_df = format_audit_log_for_display(combined_audit_log)
        if not audit_df.empty:
            status_options = ["Todos", "ok", "warning", "error"]
            selected_status = st.selectbox("Filtrar por estado", status_options, key="audit_status_filter")
            filtered_audit_df = audit_df.copy()
            if selected_status != "Todos":
                filtered_audit_df = filtered_audit_df[
                    filtered_audit_df["Estado"].astype(str).str.lower() == selected_status
                ]

            render_dataframe_clean(filtered_audit_df, title="Bitácora operativa", height=340)
            render_json_advanced("Auditoría operativa", combined_audit_log)
            st.download_button(
                "Descargar operational_audit_log.csv",
                data=dataframe_to_csv_bytes(audit_log_to_dataframe(combined_audit_log)),
                file_name="operational_audit_log.csv",
                mime="text/csv",
                key="download_operational_audit_log_tab_csv",
            )
        else:
            render_empty_state(
                "Sin eventos registrados",
                "Aún no hay eventos de auditoría.",
            )


with tabs[11]:
    st.subheader("Agente Operativo")
    if dataframe is None or dataframe.empty or active_results is None:
        st.info("Carga un dataset y genera resultados operativos primero.")
    else:
        work_orders_df = active_results.get("work_orders", pd.DataFrame())
        work_orders_df = work_orders_df if isinstance(work_orders_df, pd.DataFrame) else pd.DataFrame()
        anomalies_df = build_anomalies_table(work_orders_df)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Órdenes generadas", len(work_orders_df))
        metric_col2.metric(
            "Prioridad alta/media",
            int(work_orders_df["prioridad"].isin(["Alta", "Media"]).sum()) if not work_orders_df.empty else 0,
        )

        st.markdown("**Anomalías preliminares detectadas**")
        st.dataframe(anomalies_df, use_container_width=True)

        st.markdown("**Órdenes de trabajo enriquecidas**")
        st.dataframe(work_orders_df, use_container_width=True)

        if not work_orders_df.empty:
            st.download_button(
                "Descargar órdenes de trabajo (CSV)",
                data=dataframe_to_csv_bytes(work_orders_df),
                file_name="work_orders.csv",
                mime="text/csv",
                key="download_work_orders_operational_tab_csv",
            )


with tabs[12]:
    render_section_header(
        "Impacto Ciudadano",
        "Los pesos son transparentes y se calculan por reglas auditable, no con Gemini.",
    )
    if dataframe is None or dataframe.empty or active_results is None:
        render_empty_state(
            "Sin scores disponibles",
            "Carga un dataset y genera resultados operativos primero.",
        )
    else:
        impact_scores_df = active_results.get("impact_scores", pd.DataFrame())
        impact_scores_df = impact_scores_df if isinstance(impact_scores_df, pd.DataFrame) else pd.DataFrame()
        if impact_scores_df.empty:
            render_empty_state(
                "No se generaron impact scores",
                "El dataset actual no produjo scores de impacto suficientes.",
            )
        else:
            if active_results.get("is_meraki_mode"):
                st.info(
                    "Modo Meraki activo: el score final se deriva del `Operational Risk Score` del mart operativo. "
                    "Los componentes visibles resumen riesgo técnico, demanda y evidencia del paquete curado."
                )
            else:
                st.write(
                    "Fórmula base: 0.35 severidad técnica + 0.25 demanda + 0.25 criticidad social + "
                    "0.10 confianza de datos + 0.05 clima contextual."
                )

            render_dataframe_clean(impact_scores_df, title="Tabla completa de impacto", height=320)

            impact_chart_col1, impact_chart_col2 = st.columns(2)
            with impact_chart_col1:
                st.plotly_chart(
                    create_priority_bar_chart(impact_scores_df),
                    use_container_width=True,
                    key="impact_priority_bar",
                )
            with impact_chart_col2:
                st.plotly_chart(
                    create_classification_donut(impact_scores_df),
                    use_container_width=True,
                    key="impact_classification_donut",
                )

            impact_chart_col3, impact_chart_col4 = st.columns(2)
            with impact_chart_col3:
                st.plotly_chart(
                    create_impact_scatter(impact_scores_df),
                    use_container_width=True,
                    key="impact_scatter_chart",
                )
            with impact_chart_col4:
                selected_zone = st.selectbox(
                    "Zona para radar de score",
                    impact_scores_df["zona"].astype(str).tolist(),
                    key="impact_zone_selector",
                )
                st.plotly_chart(
                    create_score_component_radar(impact_scores_df, selected_zone=selected_zone),
                    use_container_width=True,
                    key="impact_score_radar",
                )

            st.download_button(
                "Descargar impact_scores.csv",
                data=dataframe_to_csv_bytes(impact_scores_df),
                file_name="impact_scores.csv",
                mime="text/csv",
                key="download_impact_scores_impact_tab_csv",
            )


with tabs[13]:
    st.subheader("Simulador de Cuadrillas")
    if dataframe is None or dataframe.empty or active_results is None:
        st.info("Carga un dataset y genera impact scores primero.")
    else:
        impact_scores_df = active_results.get("impact_scores", pd.DataFrame())
        impact_scores_df = impact_scores_df if isinstance(impact_scores_df, pd.DataFrame) else pd.DataFrame()
        if impact_scores_df.empty:
            st.info("No hay scores de impacto disponibles para optimizar cuadrillas.")
        else:
            crews_for_simulation = st.number_input("Número de cuadrillas disponibles", min_value=1, value=3, step=1)
            crew_sim_key = build_state_key("crew_plan_override", file_signature, mapping_signature)
            if st.button("Optimizar atención", use_container_width=True):
                st.session_state[crew_sim_key] = optimize_crews(
                    impact_scores_df,
                    available_crews=int(crews_for_simulation),
                )
                st.session_state[manual_audit_key] = append_audit_event(
                    manual_audit_log,
                    create_audit_event(
                        module="Simulador de Cuadrillas",
                        action="Recalcular plan",
                        status="ok",
                        message=f"Se recalculó el plan para {int(crews_for_simulation)} cuadrillas.",
                    ),
                )
                manual_audit_log = st.session_state[manual_audit_key]

            crew_plan = st.session_state.get(crew_sim_key) or active_results.get("crew_plan", {})
            recommended_df = crew_plan.get("recommended_zones", pd.DataFrame())
            waiting_df = crew_plan.get("waiting_zones", pd.DataFrame())

            st.write(crew_plan.get("explanation", ""))
            st.write(f"- Cobertura territorial: {crew_plan.get('coverage_territorial', 'Sin datos')}")
            st.write(f"- Riesgo de no atención: {crew_plan.get('riesgo_no_atencion', 'Sin datos')}")

            st.markdown("**Zonas recomendadas**")
            st.dataframe(recommended_df, use_container_width=True)
            st.markdown("**Zonas en espera**")
            st.dataframe(waiting_df, use_container_width=True)


with tabs[14]:
    render_section_header(
        "Pasaporte de Decisión",
        "Ficha operativa auditable para entender por qué una zona fue priorizada y qué decisión se recomienda tomar.",
    )
    if dataframe is None or dataframe.empty or active_results is None:
        render_empty_state(
            "Sin pasaportes disponibles",
            "Carga un dataset y genera resultados primero.",
        )
    else:
        passports = active_results.get("decision_passports", [])
        work_orders_df = active_results.get("work_orders", pd.DataFrame())
        impact_scores_df = active_results.get("impact_scores", pd.DataFrame())
        recommendations_df = active_results.get("recommendations", pd.DataFrame())

        if (not passports) and isinstance(impact_scores_df, pd.DataFrame) and not impact_scores_df.empty:
            passports = generate_passports_for_top_zones(
                impact_scores_df,
                work_orders=work_orders_df if isinstance(work_orders_df, pd.DataFrame) else pd.DataFrame(),
                recommendations=recommendations_df if isinstance(recommendations_df, pd.DataFrame) else pd.DataFrame(),
                top_n=10,
            )

        if not passports:
            render_empty_state(
                "Sin pasaportes de decisión",
                "No hay pasaportes de decisión disponibles todavía.",
            )
        else:
            passport_df = flatten_passports(passports)
            passport_summary_df = format_passports_for_display(passports)
            render_dataframe_clean(passport_summary_df, title="Resumen de pasaportes priorizados", height=280)

            selector_options = [
                f"{passport.get('decision_id', 'Sin ID')} | {passport.get('zona', 'Sin zona')}"
                for passport in passports
            ]
            selected_option = st.selectbox("Seleccionar pasaporte", selector_options)
            selected_decision_id = selected_option.split(" | ")[0]
            selected_passport = next(
                passport
                for passport in passports
                if str(passport.get("decision_id", "Sin ID")) == str(selected_decision_id)
            )

            render_metric_row(
                {
                    "ID de decisión": selected_passport.get("decision_id", "N/A"),
                    "Zona": selected_passport.get("zona", "N/A"),
                    "Clasificación": selected_passport.get("clasificacion", "N/A"),
                    "Score final": selected_passport.get("score_final", 0),
                    "Acción recomendada": selected_passport.get("accion_recomendada", "N/A"),
                    "Nivel de confianza": selected_passport.get("nivel_confianza", "N/A"),
                }
            )

            st.markdown("**¿Por qué importa?**")
            st.write(selected_passport.get("por_que_importa", "Sin explicación disponible."))
            display_list(
                "Evidencia técnica",
                selected_passport.get("evidencia_tecnica", []),
                "Sin evidencia técnica.",
            )
            display_list(
                "Evidencia contextual",
                selected_passport.get("evidencia_contextual", []),
                "Sin evidencia contextual.",
            )
            display_list(
                "Datos usados",
                selected_passport.get("datos_usados", []),
                "Sin detalle de datos usados.",
            )
            display_list(
                "Datos faltantes",
                selected_passport.get("datos_faltantes", []),
                "No se registran datos faltantes.",
            )
            display_list(
                "Limitaciones",
                selected_passport.get("limitaciones", []),
                "Sin limitaciones registradas.",
            )
            st.markdown("**Mensaje operativo**")
            st.write(selected_passport.get("mensaje_para_jurado", "Sin mensaje operativo disponible."))
            render_json_advanced("Pasaporte completo", selected_passport)

            st.download_button(
                "Descargar pasaportes top 10 (JSON)",
                data=dict_to_json_bytes(passports),
                file_name="decision_passports.json",
                mime="application/json",
                key="download_decision_passports_passport_tab_json",
            )
            st.download_button(
                "Descargar pasaportes top 10 (CSV)",
                data=dataframe_to_csv_bytes(passport_df),
                file_name="decision_passports.csv",
                mime="text/csv",
                key="download_decision_passports_passport_tab_csv",
            )


with tabs[15]:
    render_section_header(
        "Agente Estratégico",
        "Recomendaciones de mantenimiento e inversión con apoyo territorial y visualizaciones ejecutivas.",
    )
    if dataframe is None or dataframe.empty or active_results is None:
        render_empty_state(
            "Sin contexto estratégico",
            "Carga un dataset y genera resultados primero.",
        )
    else:
        recommendations_df = active_results.get("recommendations", pd.DataFrame())
        work_orders_df = active_results.get("work_orders", pd.DataFrame())
        impact_scores_df = active_results.get("impact_scores", pd.DataFrame())
        weather_context_df = active_results.get("weather_context", pd.DataFrame())
        osm_context_df = active_results.get("osm_context", pd.DataFrame())
        operational_mart_df = active_results.get("operational_mart", pd.DataFrame())
        meraki_anomalies_df = active_results.get("meraki_anomalies", pd.DataFrame())
        social_roi_scores_df = active_results.get("social_roi_scores", pd.DataFrame())
        recommendations_df = recommendations_df if isinstance(recommendations_df, pd.DataFrame) else pd.DataFrame()
        work_orders_df = work_orders_df if isinstance(work_orders_df, pd.DataFrame) else pd.DataFrame()
        impact_scores_df = impact_scores_df if isinstance(impact_scores_df, pd.DataFrame) else pd.DataFrame()
        weather_context_df = weather_context_df if isinstance(weather_context_df, pd.DataFrame) else pd.DataFrame()
        osm_context_df = osm_context_df if isinstance(osm_context_df, pd.DataFrame) else pd.DataFrame()
        operational_mart_df = operational_mart_df if isinstance(operational_mart_df, pd.DataFrame) else pd.DataFrame()
        meraki_anomalies_df = meraki_anomalies_df if isinstance(meraki_anomalies_df, pd.DataFrame) else pd.DataFrame()
        social_roi_scores_df = social_roi_scores_df if isinstance(social_roi_scores_df, pd.DataFrame) else pd.DataFrame()

        if active_results.get("is_meraki_mode"):
            st.info(
                "Modo Meraki activo: esta vista trabaja por Access Point (AP), zona extraída, eventos curados, clientes y métricas horarias. "
                "No hay coordenadas exactas del AP dentro del paquete oficial."
            )
            meraki_info_col1, meraki_info_col2 = st.columns(2)
            with meraki_info_col1:
                render_dataframe_clean(
                    operational_mart_df.head(12),
                    title="Mart operativo por AP",
                    height=260,
                )
            with meraki_info_col2:
                render_dataframe_clean(
                    meraki_anomalies_df.head(12),
                    title="Anomalías horarias Meraki",
                    height=260,
                )

        render_section_header(
            "Inteligencia Contextual Territorial",
            "Activa enriquecimiento contextual controlado con Open-Meteo y OpenStreetMap/Overpass cuando existan coordenadas válidas.",
        )
        context_col1, context_col2 = st.columns([1.2, 1])
        with context_col1:
            external_points_limit = st.slider(
                "Máximo de puntos externos para enriquecimiento",
                min_value=1,
                max_value=50,
                value=20,
                key="strategic_external_points_slider",
            )
        with context_col2:
            osm_radius_m = st.slider(
                "Radio OSM (metros)",
                min_value=200,
                max_value=1200,
                value=600,
                step=100,
                key="strategic_osm_radius_slider",
            )

        if external_points_limit > 20:
            st.warning("Superar 20 puntos externos puede aumentar el tiempo de consulta. Usa este rango solo cuando el dataset lo amerite.")

        if st.button(
            "Enriquecer con Open-Meteo y OpenStreetMap",
            use_container_width=True,
            key="strategic_context_enrichment_button",
            disabled=not (schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col")),
        ):
            with st.spinner("Consultando Open-Meteo y OpenStreetMap/Overpass..."):
                weather_context_df = enrich_weather_context(
                    dataframe,
                    schema_mapping,
                    max_points=int(external_points_limit),
                )
                osm_context_df = enrich_osm_context(
                    dataframe,
                    schema_mapping,
                    max_points=int(external_points_limit),
                    radius_m=int(osm_radius_m),
                )
                st.session_state[manual_weather_context_key] = {
                    "dataset_signature": file_signature,
                    "data": weather_context_df,
                }
                st.session_state[manual_osm_context_key] = {
                    "dataset_signature": file_signature,
                    "data": osm_context_df,
                }
                st.session_state[manual_audit_key] = append_audit_event(
                    manual_audit_log,
                    create_audit_event(
                        module="Agente Estratégico",
                        action="Enriquecer contexto con Open-Meteo y OSM",
                        status="ok" if (not weather_context_df.empty or not osm_context_df.empty) else "warning",
                        message="Se ejecutó enriquecimiento contextual manual desde la pestaña estratégica.",
                        metadata={
                            "weather_rows": int(len(weather_context_df)),
                            "osm_rows": int(len(osm_context_df)),
                            "max_external_points": int(external_points_limit),
                            "osm_radius_m": int(osm_radius_m),
                        },
                    ),
                )
                manual_audit_log = st.session_state[manual_audit_key]
                active_results = get_active_results(base_results, cycle_results, replay_state, synthetic_flag)
                sync_latest_operational_snapshot(source_hint="mixed")
                st.success("Contexto territorial y climático actualizado.")

        if not weather_context_df.empty or not osm_context_df.empty:
            context_preview_col1, context_preview_col2 = st.columns(2)
            with context_preview_col1:
                render_dataframe_clean(
                    weather_context_df.head(12),
                    title="Clima contextual (Open-Meteo)",
                    height=260,
                )
            with context_preview_col2:
                render_dataframe_clean(
                    osm_context_df.head(12),
                    title="POIs y criticidad territorial aproximada (OpenStreetMap)",
                    height=260,
                )
        elif not (schema_mapping.get("latitude_col") and schema_mapping.get("longitude_col")):
            render_empty_state(
                "Sin coordenadas para enriquecimiento",
                "Mapea latitud y longitud reales para activar Open-Meteo y OpenStreetMap/Overpass.",
            )

        if not social_roi_scores_df.empty:
            render_section_header(
                "Retorno social incorporado",
                "Esta vista puede priorizar no solo por falla técnica sino también por retorno social esperado.",
            )
            roi_strategy_cols = st.columns(2)
            with roi_strategy_cols[0]:
                render_dataframe_clean(
                    social_roi_scores_df.sort_values("social_roi_score", ascending=False).head(10),
                    title="Top zonas por Social ROI",
                    height=260,
                )
            with roi_strategy_cols[1]:
                top_roi_row = social_roi_scores_df.sort_values("social_roi_score", ascending=False).iloc[0]
                render_action_card(
                    "Prioridad social sugerida",
                    f"{top_roi_row.get('zone_name', 'Zona')} con Social ROI {float(top_roi_row.get('social_roi_score', 0)):.2f}. Conviene combinar la prioridad técnica con retorno social esperado.",
                    "alta",
                )

        recommendation_button_label = "Generar recomendaciones estratégicas con Gemini"
        if st.button(
            recommendation_button_label,
            use_container_width=True,
            key="strategic_generate_recommendations_button",
        ):
            recommendation_payload = get_or_generate_strategic_recommendations(
                active_results,
                df=dataframe,
                schema_mapping=schema_mapping,
                force_refresh=True,
            )
            recommendations_df = recommendation_payload["recommendations_df"]
            st.session_state[gemini_recommendations_key] = {
                "dataset_signature": file_signature,
                "data": recommendations_df,
                "summary": recommendation_payload.get("summary", ""),
                "limitations": recommendation_payload.get("limitations", []),
                "source": recommendation_payload.get("source", "fallback"),
            }
            active_results["recommendations"] = recommendations_df
            sync_latest_operational_snapshot(source_hint="mixed")
            st.success(
                "Recomendaciones estratégicas actualizadas con Gemini."
                if is_gemini_configured()
                else "Gemini no está configurado. Se generó fallback determinístico."
            )

        strategic_recommendation_payload = st.session_state.get(gemini_recommendations_key, {})
        if isinstance(strategic_recommendation_payload, dict) and strategic_recommendation_payload.get("dataset_signature") == file_signature:
            recommendations_df = strategic_recommendation_payload.get("data", recommendations_df)
            if strategic_recommendation_payload.get("summary"):
                render_insight_card(
                    "Resumen de recomendaciones estratégicas",
                    str(strategic_recommendation_payload.get("summary")),
                    status="info",
                )

        formatted_recommendations_df = format_recommendations_for_display(recommendations_df)
        render_dataframe_clean(formatted_recommendations_df, title="Recomendaciones de inversión y mantenimiento", height=300)

        top_recommendations = formatted_recommendations_df.head(3)
        if not top_recommendations.empty:
            rec_cols = st.columns(min(3, len(top_recommendations)))
            for index, (_, row) in enumerate(top_recommendations.iterrows()):
                with rec_cols[index]:
                    render_action_card(
                        str(row.get("Tipo recomendación", "Recomendación")),
                        f"{row.get('Zona o territorio', 'Sin zona')}. {row.get('Justificación', '')}",
                        "media",
                    )

        strategic_chart_col1, strategic_chart_col2 = st.columns(2)
        recommendations_treemap = create_recommendations_treemap(recommendations_df)
        with strategic_chart_col1:
            if recommendations_treemap is not None:
                st.plotly_chart(
                    recommendations_treemap,
                    use_container_width=True,
                    key="strategic_recommendations_treemap",
                )
            else:
                render_empty_state(
                    "Treemap no disponible",
                    "No hay suficiente estructura de recomendaciones para esta vista.",
                )
        with strategic_chart_col2:
            territory_heatmap = create_territory_heatmap(impact_scores_df, territory_col=schema_mapping.get("territory_col"))
            if territory_heatmap is not None:
                st.plotly_chart(
                    territory_heatmap,
                    use_container_width=True,
                    key="strategic_territory_heatmap",
                )
            else:
                territory_ranking = build_territory_ranking(impact_scores_df, work_orders_df)
                if not territory_ranking.empty:
                    render_dataframe_clean(territory_ranking, title="Ranking territorial", height=280)
                else:
                    render_empty_state(
                        "Sin ranking territorial",
                        "El dataset no contiene información territorial suficiente.",
                    )

        geo_fig = create_cali_priority_map_pro(
            dataframe,
            schema_mapping,
            impact_scores_df=impact_scores_df,
            work_orders_df=work_orders_df,
            recommendations_df=recommendations_df,
            height=760,
        )
        if geo_fig is not None:
            st.plotly_chart(
                geo_fig,
                use_container_width=True,
                key="strategic_geo_priority_map",
            )
            st.caption(
                "Mapa ejecutivo construido con coordenadas reales del dataset cargado, etiquetas de criticidad y límite de Cali de referencia."
            )
        else:
            render_empty_state(
                "Mapa estratégico no disponible",
                "El dataset no contiene información geoespacial suficiente. Se recomienda mapear latitud y longitud.",
            )


with tabs[16]:
    st.subheader("Agente Conversacional Técnico")
    st.caption("El chat usa contexto resumido. No envía toda la base a Gemini.")
    if dataframe is None or dataframe.empty:
        st.info("Carga un dataset primero.")
    elif not is_gemini_configured():
        st.warning("Gemini no está configurado. El resto de la app sigue funcionando sin él.")
    elif cycle_results is None and (replay_state is None or replay_state.get("current_results") is None):
        st.info("Ejecuta primero el ciclo autónomo o la simulación operativa para que el agente tenga contexto completo.")
    else:
        chat_results = get_active_results(base_results, cycle_results, replay_state, synthetic_flag)
        chat_context = build_orchestrated_context(chat_results)
        question = st.text_input(
            "Pregunta técnica",
            placeholder="Ejemplo: ¿Qué zona atiendo primero y por qué?",
        )
        if st.button("Preguntar a Gemini", use_container_width=True):
            with st.spinner("Consultando Gemini..."):
                answer = answer_technical_question(question, chat_context)
                st.session_state["technical_chat_answer"] = answer
        if st.session_state.get("technical_chat_answer"):
            st.markdown(st.session_state["technical_chat_answer"])


with tabs[22]:
    render_section_header(
        "Paquete de Evidencia Operativa",
        "Resumen legible de los resultados generados por el sistema para operación, validación y toma de decisiones.",
    )
    if dataframe is None or dataframe.empty or active_results is None:
        render_empty_state(
            "Sin evidencia disponible",
            "Carga un dataset y genera resultados operativos antes de exportar evidencia.",
        )
    else:
        replay_log = replay_state.get("audit_log", []) if replay_state else []
        cycle_log = cycle_results.get("audit_log", []) if cycle_results else []
        manual_log = st.session_state.get(manual_audit_key, [])
        combined_audit_log = combine_audit_logs(cycle_log, replay_log, manual_log)
        combined_quality_gate = st.session_state.get(quality_gate_key) or active_results.get("quality_gate_report", {})
        export_payload = build_export_payload(
            active_results=active_results,
            replay_state=replay_state,
            human_review_log=review_queue,
            quality_gate_report=combined_quality_gate,
            operational_audit_log=combined_audit_log,
            synthetic_flag=synthetic_flag,
            citizen_bundle=citizen_bundle,
            citizen_insights_markdown=citizen_insights_markdown,
            social_roi_bundle=social_roi_bundle,
            social_roi_explanation_markdown=social_roi_explanation_markdown,
        )

        readable_report = build_readable_evidence_report(export_payload)
        evidence_summary_df = build_evidence_summary_table(export_payload)
        work_orders_full_df = format_work_orders_for_display(export_payload.get("work_orders"))
        work_orders_display_df = work_orders_full_df.head(10)
        impact_scores_display_df = format_impact_scores_for_display(export_payload.get("impact_scores"))
        recommendations_display_df = format_recommendations_for_display(export_payload.get("recommendations"))
        passports_summary_df = format_passports_for_display(export_payload.get("decision_passports"))
        operational_mart_display_df = safe_to_dataframe(export_payload.get("operational_mart"))
        meraki_anomalies_display_df = safe_to_dataframe(export_payload.get("meraki_anomalies"))
        citizen_experience_display_df = safe_to_dataframe(export_payload.get("citizen_experience_scores"))
        citizen_recommendations_display_df = safe_to_dataframe(export_payload.get("citizen_recommendations"))
        citizen_feedback_display_df = safe_to_dataframe(export_payload.get("citizen_feedback"))
        digital_equity_display_df = safe_to_dataframe(export_payload.get("digital_equity_proxy"))
        social_roi_display_df = safe_to_dataframe(export_payload.get("social_roi_scores"))
        social_roi_recommendations_display_df = safe_to_dataframe(export_payload.get("social_roi_recommendations"))
        socioeconomic_validation_payload = export_payload.get("socioeconomic_validation", {})
        citizen_feedback_summary_payload = export_payload.get("citizen_feedback_summary", {})
        recommended_df, waiting_df, crew_summary_text = format_crew_plan_for_display(export_payload.get("crew_plan", {}))
        quality_summary, critical_issues_df, warnings_df, quality_recommendations_df = format_quality_gate_for_display(
            export_payload.get("quality_gate_report", {})
        )
        audit_display_df = format_audit_log_for_display(export_payload.get("operational_audit_log", []))

        critical_zones_count = 0
        if not impact_scores_display_df.empty and "Clasificación" in impact_scores_display_df.columns:
            critical_zones_count = int(
                impact_scores_display_df["Clasificación"].astype(str).eq("Critico").sum()
            )

        render_section_header("Resumen Ejecutivo")
        render_metric_row(
            {
                "Trace ID": export_payload.get("trace_id") or "Sin trace",
                "Fecha de generación": get_timestamp(),
                "Nivel de confianza": export_payload.get("confidence_level", "Baja"),
                "Readiness score": export_payload.get("readiness", {}).get("score", 0),
                "Órdenes": len(work_orders_full_df),
                "Zonas críticas": critical_zones_count,
                "Recomendaciones": len(recommendations_display_df),
                "Pasaportes": len(passports_summary_df),
                "Quality gate": quality_summary.get("quality_gate", "Sin evaluar"),
            }
        )
        render_dataframe_clean(evidence_summary_df, title="Resumen ejecutivo por categoría", height=260)

        render_section_header("Top Órdenes de Trabajo")
        render_dataframe_clean(work_orders_display_df, height=320)
        render_json_advanced("Órdenes de trabajo", safe_to_dataframe(export_payload.get("work_orders")).to_dict("records"))

        render_section_header("Top Zonas por Impacto")
        top_impact_df = impact_scores_display_df.head(10).copy()
        render_dataframe_clean(top_impact_df, height=320)
        if not top_impact_df.empty and {"Zona", "Score final"}.issubset(top_impact_df.columns):
            st.plotly_chart(
                px.bar(
                    top_impact_df,
                    x="Zona",
                    y="Score final",
                    color="Clasificación" if "Clasificación" in top_impact_df.columns else None,
                    title="Top zonas priorizadas por impacto",
                ),
                use_container_width=True,
                key="evidence_top_impact_bar",
            )
        render_json_advanced("Scores de impacto", safe_to_dataframe(export_payload.get("impact_scores")).to_dict("records"))

        render_section_header("Plan de Cuadrillas")
        st.write(crew_summary_text or "No hay plan de cuadrillas disponible.")
        render_dataframe_clean(recommended_df, title="Zonas recomendadas", height=260)
        render_dataframe_clean(waiting_df, title="Zonas en espera", height=260)
        render_json_advanced("Plan de cuadrillas", export_payload.get("crew_plan", {}))

        if not operational_mart_display_df.empty or not meraki_anomalies_display_df.empty:
            render_section_header("Evidencia Meraki por AP")
            meraki_ev_col1, meraki_ev_col2 = st.columns(2)
            with meraki_ev_col1:
                render_dataframe_clean(
                    operational_mart_display_df.head(15),
                    title="Mart operativo Meraki",
                    height=300,
                )
            with meraki_ev_col2:
                render_dataframe_clean(
                    meraki_anomalies_display_df.head(15),
                    title="Anomalías Meraki",
                    height=300,
                )
            render_json_advanced(
                "Evidencia Meraki avanzada",
                {
                    "operational_mart_preview": operational_mart_display_df.head(10).to_dict("records"),
                    "meraki_anomalies_preview": meraki_anomalies_display_df.head(10).to_dict("records"),
                },
            )

        render_section_header("Pasaportes de Decisión")
        render_dataframe_clean(passports_summary_df, title="Resumen de pasaportes", height=280)
        passports = export_payload.get("decision_passports", [])
        if isinstance(passports, list) and passports:
            passport_options = [
                f"{passport.get('decision_id', 'Sin ID')} | {passport.get('zona', 'Sin zona')}"
                for passport in passports
            ]
            selected_passport_option = st.selectbox(
                "Seleccionar pasaporte para ver detalle",
                passport_options,
                key="evidence_passport_selector",
            )
            selected_passport_id = selected_passport_option.split(" | ")[0]
            selected_passport = next(
                passport
                for passport in passports
                if str(passport.get("decision_id", "Sin ID")) == str(selected_passport_id)
            )

            render_metric_row(
                {
                    "Zona": selected_passport.get("zona", "N/A"),
                    "Clasificación": selected_passport.get("clasificacion", "N/A"),
                    "Score": selected_passport.get("score_final", 0),
                    "Confianza": selected_passport.get("nivel_confianza", "N/A"),
                }
            )
            st.write(f"**¿Por qué importa?** {selected_passport.get('por_que_importa', 'Sin detalle disponible.')}")
            display_list(
                "Evidencia técnica",
                selected_passport.get("evidencia_tecnica", []),
                "Sin evidencia técnica.",
            )
            display_list(
                "Evidencia contextual",
                selected_passport.get("evidencia_contextual", []),
                "Sin evidencia contextual.",
            )
            st.write(f"**Acción recomendada:** {selected_passport.get('accion_recomendada', 'N/A')}")
            display_list(
                "Datos usados",
                selected_passport.get("datos_usados", []),
                "Sin detalle de datos usados.",
            )
            display_list(
                "Datos faltantes",
                selected_passport.get("datos_faltantes", []),
                "No se registran datos faltantes.",
            )
            display_list(
                "Limitaciones",
                selected_passport.get("limitaciones", []),
                "Sin limitaciones registradas.",
            )
            render_json_advanced("Pasaporte seleccionado", selected_passport)

        render_section_header("Recomendaciones Estratégicas")
        render_dataframe_clean(recommendations_display_df, height=320)
        render_json_advanced("Recomendaciones", safe_to_dataframe(export_payload.get("recommendations")).to_dict("records"))

        if not citizen_experience_display_df.empty or not citizen_recommendations_display_df.empty or not digital_equity_display_df.empty:
            render_section_header("Experiencia Ciudadana")
            render_dataframe_clean(citizen_experience_display_df.head(15), title="Citizen Experience Score", height=280)
            render_dataframe_clean(citizen_recommendations_display_df.head(10), title="Recomendaciones para usuarios", height=240)
            render_json_advanced("Experiencia ciudadana", citizen_experience_display_df.head(20).to_dict("records"))

            render_section_header("Reportes Ciudadanos Anonimos")
            render_metric_row(
                {
                    "Reportes": citizen_feedback_summary_payload.get("total_reportes", 0),
                    "Rating promedio": citizen_feedback_summary_payload.get("rating_promedio", 0.0),
                    "Sentimiento": citizen_feedback_summary_payload.get("sentimiento_general", "Sin reportes"),
                }
            )
            render_dataframe_clean(citizen_feedback_display_df.tail(15), title="Buzon ciudadano", height=240)
            render_json_advanced("Resumen feedback ciudadano", citizen_feedback_summary_payload)

            render_section_header("Equidad Digital Proxy")
            render_dataframe_clean(digital_equity_display_df.head(15), height=280)
            render_json_advanced("Equidad digital proxy", digital_equity_display_df.head(20).to_dict("records"))

            if not social_roi_display_df.empty or not social_roi_recommendations_display_df.empty:
                render_section_header("Retorno Social de Conectividad")
                render_metric_row(
                    {
                        "Nivel geográfico": socioeconomic_validation_payload.get("level", "Sin validar"),
                        "Indicadores": len(socioeconomic_validation_payload.get("available_indicators", [])),
                        "Zonas priorizadas": len(social_roi_display_df),
                        "Recomendaciones": len(social_roi_recommendations_display_df),
                    }
                )
                render_dataframe_clean(social_roi_display_df.head(15), title="Social ROI Connectivity Score", height=280)
                render_dataframe_clean(
                    social_roi_recommendations_display_df.head(15),
                    title="Recomendaciones de retorno social",
                    height=220,
                )
                render_json_advanced("Validación socioeconómica", socioeconomic_validation_payload)
                if export_payload.get("social_roi_explanation_markdown"):
                    render_section_header("Explicación de retorno social")
                    st.markdown(str(export_payload.get("social_roi_explanation_markdown")))

            if export_payload.get("citizen_insights_markdown"):
                render_section_header("Agente Ciudadano")
                st.markdown(str(export_payload.get("citizen_insights_markdown")))

        render_section_header("Validación Técnica")
        render_metric_row(
            {
                "Quality gate": quality_summary.get("quality_gate", "Sin evaluar"),
                "Readiness operativo": quality_summary.get("operational_status", "Sin evaluar"),
                "Demo readiness": quality_summary.get("demo_readiness", "Sin evaluar"),
            }
        )
        render_dataframe_clean(critical_issues_df, title="Problemas críticos", height=180)
        render_dataframe_clean(warnings_df, title="Advertencias", height=180)
        render_dataframe_clean(quality_recommendations_df, title="Recomendaciones", height=180)
        render_json_advanced("Validación técnica", export_payload.get("quality_gate_report", {}))

        render_section_header("Auditoría Operativa")
        audit_summary = export_payload.get("operational_audit_summary", {})
        render_metric_row(
            {
                "Eventos totales": audit_summary.get("eventos_totales", 0),
                "OK": audit_summary.get("eventos_ok", 0),
                "Warnings": audit_summary.get("advertencias", 0),
                "Errores": audit_summary.get("errores", 0),
            }
        )
        render_dataframe_clean(audit_display_df.tail(15), title="Últimos eventos", height=300)
        render_json_advanced("Auditoría operativa", export_payload.get("operational_audit_log", []))

        render_section_header("Descargas")
        if st.button("Generar paquete de evidencia", use_container_width=True):
            file_paths = create_evidence_files(export_payload)
            st.session_state["evidence_file_paths"] = file_paths
            st.session_state[manual_audit_key] = append_audit_event(
                manual_audit_log,
                create_audit_event(
                    module="Paquete de Evidencia",
                    action="Generar artefactos",
                    status="ok",
                    message="Se generaron artefactos descargables en data/outputs/.",
                    metadata={"files": file_paths},
                ),
            )
            st.success("Paquete de evidencia generado en `data/outputs/`.")
            manual_audit_log = st.session_state[manual_audit_key]

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Descargar executive_report.md",
                data=readable_report.encode("utf-8"),
                file_name="executive_report.md",
                mime="text/markdown",
                key="download_executive_report_evidence_tab_md",
            )
            st.download_button(
                "Descargar readable_evidence_report.md",
                data=readable_report.encode("utf-8"),
                file_name="readable_evidence_report.md",
                mime="text/markdown",
                key="download_readable_evidence_report_tab_md",
            )
            st.download_button(
                "Descargar evidence_summary.csv",
                data=dataframe_to_csv_bytes(evidence_summary_df),
                file_name="evidence_summary.csv",
                mime="text/csv",
                key="download_evidence_summary_tab_csv",
            )
            if isinstance(export_payload.get("work_orders"), pd.DataFrame) and not export_payload["work_orders"].empty:
                st.download_button(
                    "Descargar work_orders.csv",
                    data=dataframe_to_csv_bytes(export_payload["work_orders"]),
                    file_name="work_orders.csv",
                    mime="text/csv",
                    key="download_work_orders_evidence_tab_csv",
                )
            if isinstance(export_payload.get("impact_scores"), pd.DataFrame) and not export_payload["impact_scores"].empty:
                st.download_button(
                    "Descargar impact_scores.csv",
                    data=dataframe_to_csv_bytes(export_payload["impact_scores"]),
                    file_name="impact_scores.csv",
                    mime="text/csv",
                    key="download_impact_scores_evidence_tab_csv",
                )
            if isinstance(export_payload.get("recommendations"), pd.DataFrame) and not export_payload["recommendations"].empty:
                st.download_button(
                    "Descargar strategic_recommendations.csv",
                    data=dataframe_to_csv_bytes(export_payload["recommendations"]),
                    file_name="strategic_recommendations.csv",
                    mime="text/csv",
                    key="download_strategic_recommendations_evidence_tab_csv",
                )
            if isinstance(export_payload.get("citizen_experience_scores"), pd.DataFrame) and not export_payload["citizen_experience_scores"].empty:
                st.download_button(
                    "Descargar citizen_experience_scores.csv",
                    data=dataframe_to_csv_bytes(export_payload["citizen_experience_scores"]),
                    file_name="citizen_experience_scores.csv",
                    mime="text/csv",
                    key="download_citizen_experience_scores_evidence_tab_csv",
                )
            if isinstance(export_payload.get("citizen_recommendations"), pd.DataFrame) and not export_payload["citizen_recommendations"].empty:
                st.download_button(
                    "Descargar citizen_recommendations.csv",
                    data=dataframe_to_csv_bytes(export_payload["citizen_recommendations"]),
                    file_name="citizen_recommendations.csv",
                    mime="text/csv",
                    key="download_citizen_recommendations_evidence_tab_csv",
                )
            if isinstance(export_payload.get("citizen_feedback"), pd.DataFrame) and not export_payload["citizen_feedback"].empty:
                st.download_button(
                    "Descargar citizen_feedback.csv",
                    data=dataframe_to_csv_bytes(export_payload["citizen_feedback"]),
                    file_name="citizen_feedback.csv",
                    mime="text/csv",
                    key="download_citizen_feedback_evidence_tab_csv",
                )
            if isinstance(export_payload.get("digital_equity_proxy"), pd.DataFrame) and not export_payload["digital_equity_proxy"].empty:
                st.download_button(
                    "Descargar digital_equity_proxy.csv",
                    data=dataframe_to_csv_bytes(export_payload["digital_equity_proxy"]),
                    file_name="digital_equity_proxy.csv",
                    mime="text/csv",
                    key="download_digital_equity_proxy_evidence_tab_csv",
                )
            if isinstance(export_payload.get("replay_timeline"), pd.DataFrame) and not export_payload["replay_timeline"].empty:
                st.download_button(
                    "Descargar replay_timeline.csv",
                    data=dataframe_to_csv_bytes(export_payload["replay_timeline"]),
                    file_name="replay_timeline.csv",
                    mime="text/csv",
                    key="download_replay_timeline_evidence_tab_csv",
                )
            if isinstance(export_payload.get("human_review_log"), pd.DataFrame) and not export_payload["human_review_log"].empty:
                st.download_button(
                    "Descargar human_review_log.csv",
                    data=dataframe_to_csv_bytes(export_payload["human_review_log"]),
                    file_name="human_review_log.csv",
                    mime="text/csv",
                    key="download_human_review_log_evidence_tab_csv",
                )
            if isinstance(export_payload.get("operational_mart"), pd.DataFrame) and not export_payload["operational_mart"].empty:
                st.download_button(
                    "Descargar operational_mart.csv",
                    data=dataframe_to_csv_bytes(export_payload["operational_mart"]),
                    file_name="operational_mart.csv",
                    mime="text/csv",
                    key="download_operational_mart_evidence_tab_csv",
                )
            if isinstance(export_payload.get("meraki_anomalies"), pd.DataFrame) and not export_payload["meraki_anomalies"].empty:
                st.download_button(
                    "Descargar meraki_anomalies.csv",
                    data=dataframe_to_csv_bytes(export_payload["meraki_anomalies"]),
                    file_name="meraki_anomalies.csv",
                    mime="text/csv",
                    key="download_meraki_anomalies_evidence_tab_csv",
                )
        with download_col2:
            st.download_button(
                "Descargar crew_plan.json",
                data=dict_to_json_bytes(export_payload.get("crew_plan", {})),
                file_name="crew_plan.json",
                mime="application/json",
                key="download_crew_plan_evidence_tab_json",
            )
            st.download_button(
                "Descargar decision_passports.json",
                data=dict_to_json_bytes(export_payload.get("decision_passports", [])),
                file_name="decision_passports.json",
                mime="application/json",
                key="download_decision_passports_evidence_tab_json",
            )
            st.download_button(
                "Descargar quality_gate_report.json",
                data=dict_to_json_bytes(export_payload.get("quality_gate_report", {})),
                file_name="quality_gate_report.json",
                mime="application/json",
                key="download_quality_gate_report_evidence_tab_json",
            )
            if export_payload.get("operational_audit_log"):
                st.download_button(
                    "Descargar operational_audit_log.csv",
                    data=dataframe_to_csv_bytes(audit_log_to_dataframe(export_payload.get("operational_audit_log", []))),
                    file_name="operational_audit_log.csv",
                    mime="text/csv",
                    key="download_operational_audit_log_evidence_tab_csv",
                )
            st.download_button(
                "Descargar operational_audit_summary.json",
                data=dict_to_json_bytes(export_payload.get("operational_audit_summary", {})),
                file_name="operational_audit_summary.json",
                mime="application/json",
                key="download_operational_audit_summary_evidence_tab_json",
            )
            st.download_button(
                "Descargar evidence_pack.json",
                data=dict_to_json_bytes(export_payload),
                file_name="evidence_pack.json",
                mime="application/json",
                key="download_evidence_pack_evidence_tab_json",
            )
            if export_payload.get("citizen_insights_markdown"):
                st.download_button(
                    "Descargar citizen_insights.md",
                    data=str(export_payload.get("citizen_insights_markdown")).encode("utf-8"),
                    file_name="citizen_insights.md",
                    mime="text/markdown",
                    key="download_citizen_insights_evidence_tab_md",
                )
            decision_passports_payload = export_payload.get("decision_passports", [])
            if isinstance(decision_passports_payload, list) and any(
                isinstance(passport, dict) and passport.get("ap_name")
                for passport in decision_passports_payload
            ):
                st.download_button(
                    "Descargar meraki_decision_passports.json",
                    data=dict_to_json_bytes(decision_passports_payload),
                    file_name="meraki_decision_passports.json",
                    mime="application/json",
                    key="download_meraki_decision_passports_evidence_tab_json",
                )

        if st.session_state.get("evidence_file_paths"):
            st.caption("Archivos generados en disco")
            render_download_buttons_for_evidence(st.session_state["evidence_file_paths"])


floating_messages = st.session_state.get("floating_agent_messages", [])
floating_agent_send, floating_agent_question = render_floating_chat_widget(
    floating_messages,
    description=(
        "Este agente responde preguntas sobre la plataforma, la base cargada y el análisis generado. "
        "No responde preguntas generales fuera del contexto operativo."
    ),
    gemini_configured=is_gemini_configured(),
)
if floating_agent_send and str(floating_agent_question).strip():
    platform_results = active_results if isinstance(active_results, dict) else {}
    platform_context = build_platform_agent_context(
        platform_results,
        df=dataframe,
        schema_mapping=schema_mapping if isinstance(schema_mapping, dict) else {},
    )
    platform_answer = answer_platform_question(floating_agent_question, platform_context)
    updated_messages = list(floating_messages)
    updated_messages.append({"role": "user", "content": str(floating_agent_question).strip()})
    updated_messages.append({"role": "assistant", "content": platform_answer})
    st.session_state["floating_agent_messages"] = updated_messages

    if dataframe is not None and not dataframe.empty and "manual_audit_key" in locals():
        st.session_state[manual_audit_key] = append_audit_event(
            st.session_state.get(manual_audit_key, []),
            create_audit_event(
                module="Agente Conversacional de Plataforma",
                action="Responder pregunta contextual",
                status="ok",
                message="Se respondió una consulta sobre la plataforma y el análisis actual.",
                metadata={"question": str(floating_agent_question).strip()[:180]},
            ),
        )
    st.rerun()
