from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_OUTPUTS_DIR
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
from src.operational_audit import audit_log_to_dataframe, build_operational_audit_summary
from src.profile_storage import convert_to_serializable
from src.utils import get_timestamp


def ensure_output_dir() -> Path:
    """Asegura la existencia de la carpeta de salidas."""
    DATA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_OUTPUTS_DIR


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a bytes CSV."""
    if dataframe is None or dataframe.empty:
        return b""
    return dataframe.to_csv(index=False).encode("utf-8")


def dict_to_json_bytes(data: dict[str, Any] | list[Any]) -> bytes:
    """Convierte una estructura JSON serializable a bytes."""
    return json.dumps(
        convert_to_serializable(data),
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _top_records(dataframe: pd.DataFrame | None, limit: int = 5) -> list[dict[str, object]]:
    """Resume DataFrames para el reporte."""
    if dataframe is None or dataframe.empty:
        return []
    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def _bullet_block(items: list[str], empty_message: str) -> list[str]:
    """Normaliza listas de texto en bullets markdown."""
    if not items:
        return [f"- {empty_message}"]
    return [f"- {item}" for item in items]


def _dataframe_preview_block(dataframe: pd.DataFrame | None, empty_message: str, limit: int = 5) -> str:
    """Convierte un DataFrame pequeño en bullets markdown sin depender de tabulate."""
    if dataframe is None or dataframe.empty:
        return f"- {empty_message}"

    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), "")
    lines: list[str] = []
    for _, row in preview.iterrows():
        row_parts = [f"{column}: {row[column]}" for column in preview.columns]
        lines.append(f"- {' | '.join(row_parts)}")
    return "\n".join(lines)


def _column_bullets(dataframe: pd.DataFrame | None, column_name: str, empty_message: str) -> list[str]:
    """Extrae bullets desde una columna si existe, con fallback limpio."""
    if dataframe is None or dataframe.empty or column_name not in dataframe.columns:
        return [f"- {empty_message}"]
    return dataframe[column_name].map(lambda value: f"- {value}").tolist() or [f"- {empty_message}"]


def build_evidence_summary_table(results: dict[str, Any]) -> pd.DataFrame:
    """Resume indicadores del sistema en una tabla ejecutiva."""
    readiness = results.get("readiness", {})
    work_orders = results.get("work_orders")
    impact_scores = results.get("impact_scores")
    crew_plan = results.get("crew_plan", {})
    passports = results.get("decision_passports", [])
    recommendations = results.get("recommendations")
    limitations = results.get("limitations", [])
    quality_gate_report = results.get("quality_gate_report", {})
    audit_log = results.get("operational_audit_log") or results.get("audit_log", [])
    audit_summary = results.get("operational_audit_summary") or build_operational_audit_summary(audit_log)
    citizen_scores = safe_to_dataframe(results.get("citizen_experience_scores"))
    citizen_recommendations = safe_to_dataframe(results.get("citizen_recommendations"))
    citizen_feedback = safe_to_dataframe(results.get("citizen_feedback"))
    digital_equity = safe_to_dataframe(results.get("digital_equity_proxy"))
    social_roi_df = safe_to_dataframe(results.get("social_roi_scores"))
    social_roi_recommendations = safe_to_dataframe(results.get("social_roi_recommendations"))
    socioeconomic_validation = results.get("socioeconomic_validation", {})
    citizen_feedback_summary = results.get("citizen_feedback_summary", {})
    recommended_df, waiting_df, crew_summary = format_crew_plan_for_display(crew_plan)

    work_orders_count = len(work_orders) if isinstance(work_orders, pd.DataFrame) else 0
    impact_scores_count = len(impact_scores) if isinstance(impact_scores, pd.DataFrame) else 0
    critical_zones_count = 0
    if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty and "classification" in impact_scores.columns:
        critical_zones_count = int(impact_scores["classification"].astype(str).isin(["Critico", "Alto"]).sum())

    summary_rows = [
        {
            "categoria": "Operacion",
            "indicador": "Ordenes generadas",
            "valor": work_orders_count,
            "interpretacion": "Requieren revision tecnica o seguimiento operativo." if work_orders_count else "No hay ordenes generadas todavia.",
        },
        {
            "categoria": "Impacto",
            "indicador": "Zonas priorizadas",
            "valor": impact_scores_count,
            "interpretacion": "Corresponde a zonas con score calculado." if impact_scores_count else "No hay scores calculados.",
        },
        {
            "categoria": "Impacto",
            "indicador": "Zonas criticas o altas",
            "valor": critical_zones_count,
            "interpretacion": "Prioridad alta para operacion y seguimiento." if critical_zones_count else "No se detectan zonas altas con la evidencia actual.",
        },
        {
            "categoria": "Recursos",
            "indicador": "Zonas recomendadas para cuadrillas",
            "valor": len(recommended_df),
            "interpretacion": crew_summary or "Sin plan de cuadrillas disponible.",
        },
        {
            "categoria": "Validacion",
            "indicador": "Quality Gate",
            "valor": quality_gate_report.get("quality_gate", "Sin evaluar"),
            "interpretacion": "Resume si el dataset y el resultado son operables.",
        },
        {
            "categoria": "Validacion",
            "indicador": "Readiness Score",
            "valor": readiness.get("score", 0),
            "interpretacion": readiness.get("classification", "Sin clasificar"),
        },
        {
            "categoria": "Decision",
            "indicador": "Pasaportes generados",
            "valor": len(passports) if isinstance(passports, list) else 0,
            "interpretacion": "Fichas auditable por zona priorizada.",
        },
        {
            "categoria": "Auditoria",
            "indicador": "Eventos auditados",
            "valor": audit_summary.get("eventos_totales", 0),
            "interpretacion": "Trazabilidad total del flujo ejecutado.",
        },
        {
            "categoria": "Estrategia",
            "indicador": "Recomendaciones",
            "valor": len(recommendations) if isinstance(recommendations, pd.DataFrame) else 0,
            "interpretacion": "Acciones sugeridas para operacion o inversion.",
        },
        {
            "categoria": "Ciudadania",
            "indicador": "Citizen Experience registros",
            "valor": int(len(citizen_scores)),
            "interpretacion": "APs o zonas con experiencia agregada calculada." if not citizen_scores.empty else "No hay score ciudadano calculado.",
        },
        {
            "categoria": "Ciudadania",
            "indicador": "Recomendaciones para usuarios",
            "valor": int(len(citizen_recommendations)),
            "interpretacion": "Sugerencias agregadas de donde y cuando conectarse." if not citizen_recommendations.empty else "No hay recomendaciones ciudadanas todavia.",
        },
        {
            "categoria": "Ciudadania",
            "indicador": "Reportes anonimos",
            "valor": int(len(citizen_feedback)),
            "interpretacion": f"Sentimiento general: {citizen_feedback_summary.get('sentimiento_general', 'Sin reportes')}.",
        },
        {
            "categoria": "Ciudadania",
            "indicador": "Equidad digital proxy",
            "valor": int(len(digital_equity)),
            "interpretacion": "Senales relativas de necesidad de mejora por zona." if not digital_equity.empty else "No hay proxy de equidad calculado.",
        },
        {
            "categoria": "Impacto social",
            "indicador": "Social ROI calculado",
            "valor": int(len(social_roi_df)),
            "interpretacion": "Cruce entre red, experiencia y contexto socioeconomico agregado." if not social_roi_df.empty else "No hay Social ROI calculado.",
        },
        {
            "categoria": "Impacto social",
            "indicador": "Recomendaciones de retorno social",
            "valor": int(len(social_roi_recommendations)),
            "interpretacion": "Acciones sugeridas para inversión social y acompañamiento." if not social_roi_recommendations.empty else "No hay recomendaciones de retorno social disponibles.",
        },
        {
            "categoria": "Impacto social",
            "indicador": "Validación socioeconómica",
            "valor": socioeconomic_validation.get("level", "Sin validar"),
            "interpretacion": "Nivel geográfico detectado para el dataset socioeconómico.",
        },
        {
            "categoria": "Riesgo",
            "indicador": "Limitaciones registradas",
            "valor": len(limitations),
            "interpretacion": "Aspectos a revisar antes de escalar el analisis." if limitations else "No se registraron limitaciones adicionales.",
        },
        {
            "categoria": "Recursos",
            "indicador": "Zonas en espera",
            "valor": len(waiting_df),
            "interpretacion": "Pendientes de atencion por capacidad limitada." if len(waiting_df) else "No hay zonas en espera.",
        },
    ]

    return pd.DataFrame(summary_rows)


def build_readable_evidence_report(results: dict[str, Any]) -> str:
    """Genera un reporte tecnico-ejecutivo legible para operacion publica."""
    readiness = results.get("readiness", {})
    executive_summary = results.get("executive_summary", "")
    limitations = results.get("limitations", [])
    confidence_level = results.get("confidence_level", "Baja")
    trace_id = results.get("trace_id") or "Sin trace"
    quality_gate_report = results.get("quality_gate_report", {})
    work_orders_df = format_work_orders_for_display(results.get("work_orders")).head(5)
    impact_scores_df = format_impact_scores_for_display(results.get("impact_scores")).head(5)
    recommended_df, waiting_df, crew_summary = format_crew_plan_for_display(results.get("crew_plan", {}))
    passports_df = format_passports_for_display(results.get("decision_passports")).head(5)
    recommendations_df = format_recommendations_for_display(results.get("recommendations")).head(5)
    operational_mart_df = safe_to_dataframe(results.get("operational_mart")).head(5)
    meraki_anomalies_df = safe_to_dataframe(results.get("meraki_anomalies")).head(5)
    citizen_scores_df = safe_to_dataframe(results.get("citizen_experience_scores")).head(5)
    citizen_recommendations_df = safe_to_dataframe(results.get("citizen_recommendations")).head(5)
    citizen_feedback_df = safe_to_dataframe(results.get("citizen_feedback")).head(5)
    digital_equity_df = safe_to_dataframe(results.get("digital_equity_proxy")).head(5)
    social_roi_scores_df = safe_to_dataframe(results.get("social_roi_scores")).head(5)
    social_roi_recommendations_df = safe_to_dataframe(results.get("social_roi_recommendations")).head(5)
    citizen_feedback_summary = results.get("citizen_feedback_summary", {})
    citizen_insights_markdown = str(results.get("citizen_insights_markdown", "") or "")
    social_roi_explanation_markdown = str(results.get("social_roi_explanation_markdown", "") or "")
    socioeconomic_validation = results.get("socioeconomic_validation", {})
    quality_summary, critical_df, warnings_df, recommendations_qg_df = format_quality_gate_for_display(
        quality_gate_report
    )
    audit_df = format_audit_log_for_display(results.get("operational_audit_log") or results.get("audit_log", [])).tail(5)

    report_lines = [
        "# Cali WiFi Sentinel 360 - Reporte de Evidencia Operativa",
        "",
        "## 1. Resumen ejecutivo",
        f"- Fecha de generacion: {get_timestamp()}",
        f"- Trace ID: {trace_id}",
        f"- Nivel de confianza: {confidence_level}",
        executive_summary or "- No se genero resumen ejecutivo base.",
        "",
        "## 2. Estado del dataset y preparacion operativa",
        f"- Data Readiness Score: {readiness.get('score', 0)} / 100",
        f"- Clasificacion de readiness: {readiness.get('classification', 'Sin clasificar')}",
        f"- Quality Gate: {quality_summary.get('quality_gate', 'Sin evaluar')}",
        f"- Estado operativo: {quality_summary.get('operational_status', 'Sin evaluar')}",
        "",
        "Fortalezas detectadas:",
        *_bullet_block(readiness.get("strengths", []), "No se registraron fortalezas."),
        "",
        "Brechas detectadas:",
        *_bullet_block(readiness.get("gaps", []), "No se registraron brechas."),
        "",
        "## 3. Ordenes de trabajo principales",
        _dataframe_preview_block(work_orders_df, "No hay ordenes de trabajo disponibles."),
        "",
        "## 4. Zonas priorizadas por impacto",
        _dataframe_preview_block(impact_scores_df, "No hay zonas priorizadas por impacto."),
        "",
        "## 5. Plan de cuadrillas recomendado",
        f"- Resumen: {crew_summary or 'No hay plan de cuadrillas disponible.'}",
        f"- Zonas recomendadas: {len(recommended_df)}",
        f"- Zonas en espera: {len(waiting_df)}",
        "",
        "## 6. Pasaportes de decision principales",
        _dataframe_preview_block(passports_df, "No hay pasaportes de decision generados."),
        "",
        "## 7. Recomendaciones estrategicas",
        _dataframe_preview_block(recommendations_df, "No hay recomendaciones estrategicas disponibles."),
        "",
        "### Recomendaciones Estratégicas Generadas por Agente",
        _dataframe_preview_block(
            _safe_agent_recommendations_preview(results.get("recommendations")),
            "No hay recomendaciones estructuradas del agente disponibles.",
        ),
        "",
        "## 8. Validacion tecnica",
        f"- Quality Gate: {quality_summary.get('quality_gate', 'Sin evaluar')}",
        f"- Demo / readiness operativo: {quality_summary.get('demo_readiness', 'Sin evaluar')}",
        "Problemas criticos:",
        *_column_bullets(critical_df, "problema_critico", "No se registraron problemas criticos."),
        "Advertencias:",
        *_column_bullets(warnings_df, "advertencia", "No se registraron advertencias."),
        "Recomendaciones:",
        *_column_bullets(recommendations_qg_df, "recomendacion", "No se registraron recomendaciones adicionales."),
        "",
        "## 9. Auditoria operativa",
        _dataframe_preview_block(audit_df, "No hay eventos de auditoria registrados."),
        "",
        "## 10. Limitaciones responsables",
        *_bullet_block(limitations, "No se registraron limitaciones adicionales."),
        "- El sistema no afirma causalidad tecnica sin evidencia directa.",
        "- El clima y los POIs se usan como contexto y no como prueba definitiva de falla.",
        "- Los datos sinteticos no representan informacion oficial de la Alcaldia ni del operador.",
        "",
        "## 11. Proximos pasos sugeridos",
        "- Validar manualmente las zonas de mayor score antes de ejecutar acciones en campo.",
        "- Completar coordenadas, fecha y estado cuando falten para mejorar la confianza operativa.",
        "- Usar la auditoria y la validacion humana para cerrar el ciclo de mejora continua.",
    ]

    meraki_insert = [
        "## 7.1. Mart operativo Meraki",
        _dataframe_preview_block(operational_mart_df, "No hay mart operativo Meraki disponible."),
        "",
        "## 7.2. Anomalias Meraki",
        _dataframe_preview_block(meraki_anomalies_df, "No hay anomalias Meraki registradas."),
        "",
    ]
    citizen_insert = [
        "## 7.3. Experiencia ciudadana",
        _dataframe_preview_block(citizen_scores_df, "No hay score ciudadano disponible."),
        "",
        "## 7.4. Recomendaciones para usuarios",
        _dataframe_preview_block(citizen_recommendations_df, "No hay recomendaciones ciudadanas disponibles."),
        "",
        "## 7.5. Reportes ciudadanos anonimos",
        f"- Total reportes: {citizen_feedback_summary.get('total_reportes', 0)}",
        f"- Rating promedio: {citizen_feedback_summary.get('rating_promedio', 0.0)}",
        f"- Sentimiento general: {citizen_feedback_summary.get('sentimiento_general', 'Sin reportes')}",
        _dataframe_preview_block(citizen_feedback_df, "No hay comentarios anonimos registrados."),
        "",
        "## 7.6. Equidad digital proxy",
        _dataframe_preview_block(digital_equity_df, "No hay proxy de equidad digital disponible."),
        "",
        "## 7.7. Recomendaciones Estratégicas Generadas por Agente",
        _dataframe_preview_block(
            _safe_agent_recommendations_preview(results.get("recommendations")),
            "No hay recomendaciones estructuradas del agente disponibles.",
        ),
        "",
        "## 7.8. Analisis ciudadano con agente",
        citizen_insights_markdown or "- No hay analisis ciudadano generado todavia.",
        "",
        "## 7.9. Retorno Social de Conectividad",
        f"- Nivel geografico socioeconomico: {socioeconomic_validation.get('level', 'Sin validar')}",
        f"- Indicadores disponibles: {', '.join(socioeconomic_validation.get('available_indicators', [])) or 'Sin indicadores detectados'}",
        _dataframe_preview_block(social_roi_scores_df, "No hay Social ROI calculado."),
        "",
        "## 7.10. Recomendaciones de retorno social",
        _dataframe_preview_block(social_roi_recommendations_df, "No hay recomendaciones de retorno social disponibles."),
        "",
        "## 7.11. Explicacion de retorno social",
        social_roi_explanation_markdown or "- No hay explicacion de retorno social generada todavia.",
        "",
    ]
    insert_index = next(
        (index for index, value in enumerate(report_lines) if value == "## 8. Validacion tecnica"),
        len(report_lines),
    )
    report_lines[insert_index:insert_index] = meraki_insert
    insert_index = next(
        (index for index, value in enumerate(report_lines) if value == "## 8. Validacion tecnica"),
        len(report_lines),
    )
    report_lines[insert_index:insert_index] = citizen_insert

    return "\n".join(report_lines)


def build_evidence_markdown(
    results: dict[str, Any],
    project_name: str = "Cali WiFi Sentinel 360",
) -> str:
    """Mantiene compatibilidad hacia atras devolviendo un reporte legible."""
    report = build_readable_evidence_report(results)
    if project_name == "Cali WiFi Sentinel 360":
        return report
    return report.replace("Cali WiFi Sentinel 360", project_name, 1)


def _safe_agent_recommendations_preview(recommendations: object) -> pd.DataFrame:
    """Resume recomendaciones con columnas ampliadas si existen."""
    dataframe = recommendations if isinstance(recommendations, pd.DataFrame) else pd.DataFrame()
    if dataframe.empty:
        return pd.DataFrame()

    preview_columns = [
        column_name
        for column_name in [
            "zona_o_territorio",
            "tipo_recomendacion",
            "justificacion",
            "impacto_estimado",
            "esfuerzo_estimado",
            "nivel_confianza",
            "evidencia_usada",
            "limitaciones",
        ]
        if column_name in dataframe.columns
    ]
    return dataframe[preview_columns].head(5) if preview_columns else dataframe.head(5)


def create_evidence_files(results: dict[str, Any]) -> dict[str, str]:
    """Genera archivos descargables del paquete de evidencia."""
    output_dir = ensure_output_dir()

    markdown_content = build_evidence_markdown(results)
    readable_markdown_content = build_readable_evidence_report(results)
    evidence_summary_df = build_evidence_summary_table(results)

    executive_report_path = output_dir / "executive_report.md"
    executive_report_path.write_text(markdown_content, encoding="utf-8")

    file_paths: dict[str, str] = {
        "executive_report": str(executive_report_path),
    }

    work_orders = results.get("work_orders")
    if isinstance(work_orders, pd.DataFrame) and not work_orders.empty:
        work_orders_path = output_dir / "work_orders.csv"
        work_orders_path.write_bytes(dataframe_to_csv_bytes(work_orders))
        file_paths["work_orders"] = str(work_orders_path)
        if "source" in work_orders.columns and work_orders["source"].astype(str).eq("meraki_package").any():
            meraki_work_orders_path = output_dir / "meraki_work_orders.csv"
            meraki_work_orders_path.write_bytes(dataframe_to_csv_bytes(work_orders))
            file_paths["meraki_work_orders"] = str(meraki_work_orders_path)

    impact_scores = results.get("impact_scores")
    if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty:
        impact_scores_path = output_dir / "impact_scores.csv"
        impact_scores_path.write_bytes(dataframe_to_csv_bytes(impact_scores))
        file_paths["impact_scores"] = str(impact_scores_path)

    recommendations = results.get("recommendations")
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        recommendations_path = output_dir / "strategic_recommendations.csv"
        recommendations_path.write_bytes(dataframe_to_csv_bytes(recommendations))
        file_paths["strategic_recommendations"] = str(recommendations_path)

    citizen_scores = results.get("citizen_experience_scores")
    if isinstance(citizen_scores, pd.DataFrame) and not citizen_scores.empty:
        citizen_scores_path = output_dir / "citizen_experience_scores.csv"
        citizen_scores_path.write_bytes(dataframe_to_csv_bytes(citizen_scores))
        file_paths["citizen_experience_scores"] = str(citizen_scores_path)

    citizen_recommendations = results.get("citizen_recommendations")
    if isinstance(citizen_recommendations, pd.DataFrame) and not citizen_recommendations.empty:
        citizen_recommendations_path = output_dir / "citizen_recommendations.csv"
        citizen_recommendations_path.write_bytes(dataframe_to_csv_bytes(citizen_recommendations))
        file_paths["citizen_recommendations"] = str(citizen_recommendations_path)

    citizen_feedback = results.get("citizen_feedback")
    if isinstance(citizen_feedback, pd.DataFrame) and not citizen_feedback.empty:
        citizen_feedback_path = output_dir / "citizen_feedback.csv"
        citizen_feedback_path.write_bytes(dataframe_to_csv_bytes(citizen_feedback))
        file_paths["citizen_feedback"] = str(citizen_feedback_path)

    digital_equity = results.get("digital_equity_proxy")
    if isinstance(digital_equity, pd.DataFrame) and not digital_equity.empty:
        digital_equity_path = output_dir / "digital_equity_proxy.csv"
        digital_equity_path.write_bytes(dataframe_to_csv_bytes(digital_equity))
        file_paths["digital_equity_proxy"] = str(digital_equity_path)

    social_roi_scores = results.get("social_roi_scores")
    if isinstance(social_roi_scores, pd.DataFrame) and not social_roi_scores.empty:
        social_roi_scores_path = output_dir / "social_roi_scores.csv"
        social_roi_scores_path.write_bytes(dataframe_to_csv_bytes(social_roi_scores))
        file_paths["social_roi_scores"] = str(social_roi_scores_path)

    social_roi_recommendations = results.get("social_roi_recommendations")
    if isinstance(social_roi_recommendations, pd.DataFrame) and not social_roi_recommendations.empty:
        social_roi_recommendations_path = output_dir / "social_roi_recommendations.csv"
        social_roi_recommendations_path.write_bytes(dataframe_to_csv_bytes(social_roi_recommendations))
        file_paths["social_roi_recommendations"] = str(social_roi_recommendations_path)

    operational_mart = results.get("operational_mart")
    if isinstance(operational_mart, pd.DataFrame) and not operational_mart.empty:
        operational_mart_path = output_dir / "operational_mart.csv"
        operational_mart_path.write_bytes(dataframe_to_csv_bytes(operational_mart))
        file_paths["operational_mart"] = str(operational_mart_path)

    meraki_anomalies = results.get("meraki_anomalies")
    if isinstance(meraki_anomalies, pd.DataFrame) and not meraki_anomalies.empty:
        meraki_anomalies_path = output_dir / "meraki_anomalies.csv"
        meraki_anomalies_path.write_bytes(dataframe_to_csv_bytes(meraki_anomalies))
        file_paths["meraki_anomalies"] = str(meraki_anomalies_path)

    crew_plan_path = output_dir / "crew_plan.json"
    crew_plan_payload = results.get("crew_plan", {})
    crew_plan_path.write_bytes(dict_to_json_bytes(crew_plan_payload))
    file_paths["crew_plan"] = str(crew_plan_path)

    decision_passports_path = output_dir / "decision_passports.json"
    decision_passports_path.write_bytes(
        dict_to_json_bytes(results.get("decision_passports", []))
    )
    file_paths["decision_passports"] = str(decision_passports_path)
    decision_passports = results.get("decision_passports", [])
    if isinstance(decision_passports, list) and any(
        isinstance(passport, dict) and passport.get("ap_name")
        for passport in decision_passports
    ):
        meraki_passports_path = output_dir / "meraki_decision_passports.json"
        meraki_passports_path.write_bytes(dict_to_json_bytes(decision_passports))
        file_paths["meraki_decision_passports"] = str(meraki_passports_path)

    replay_timeline = results.get("replay_timeline")
    if isinstance(replay_timeline, pd.DataFrame) and not replay_timeline.empty:
        replay_timeline_path = output_dir / "replay_timeline.csv"
        replay_timeline_path.write_bytes(dataframe_to_csv_bytes(replay_timeline))
        file_paths["replay_timeline"] = str(replay_timeline_path)

    human_review_log = results.get("human_review_log")
    if isinstance(human_review_log, pd.DataFrame) and not human_review_log.empty:
        human_review_log_path = output_dir / "human_review_log.csv"
        human_review_log_path.write_bytes(dataframe_to_csv_bytes(human_review_log))
        file_paths["human_review_log"] = str(human_review_log_path)

    quality_gate_report = results.get("quality_gate_report")
    if isinstance(quality_gate_report, dict) and quality_gate_report:
        quality_gate_path = output_dir / "quality_gate_report.json"
        quality_gate_path.write_bytes(dict_to_json_bytes(quality_gate_report))
        file_paths["quality_gate_report"] = str(quality_gate_path)

    socioeconomic_validation = results.get("socioeconomic_validation")
    if isinstance(socioeconomic_validation, dict) and socioeconomic_validation:
        socioeconomic_validation_path = output_dir / "socioeconomic_validation.json"
        socioeconomic_validation_path.write_bytes(dict_to_json_bytes(socioeconomic_validation))
        file_paths["socioeconomic_validation"] = str(socioeconomic_validation_path)

    audit_log = results.get("operational_audit_log") or results.get("audit_log", [])
    if audit_log:
        audit_log_df = audit_log_to_dataframe(audit_log)
        audit_log_path = output_dir / "operational_audit_log.csv"
        audit_log_path.write_bytes(dataframe_to_csv_bytes(audit_log_df))
        file_paths["operational_audit_log"] = str(audit_log_path)

        audit_summary_path = output_dir / "operational_audit_summary.json"
        audit_summary_path.write_bytes(
            dict_to_json_bytes(
                results.get("operational_audit_summary") or build_operational_audit_summary(audit_log)
            )
        )
        file_paths["operational_audit_summary"] = str(audit_summary_path)

    evidence_pack_path = output_dir / "evidence_pack.json"
    evidence_pack_path.write_bytes(dict_to_json_bytes(results))
    file_paths["evidence_pack"] = str(evidence_pack_path)

    readable_report_path = output_dir / "readable_evidence_report.md"
    readable_report_path.write_text(readable_markdown_content, encoding="utf-8")
    file_paths["readable_evidence_report"] = str(readable_report_path)

    citizen_insights_markdown = results.get("citizen_insights_markdown")
    if citizen_insights_markdown:
        citizen_insights_path = output_dir / "citizen_insights.md"
        citizen_insights_path.write_text(str(citizen_insights_markdown), encoding="utf-8")
        file_paths["citizen_insights"] = str(citizen_insights_path)

    social_roi_explanation_markdown = results.get("social_roi_explanation_markdown")
    if social_roi_explanation_markdown:
        social_roi_explanation_path = output_dir / "social_roi_explanation.md"
        social_roi_explanation_path.write_text(str(social_roi_explanation_markdown), encoding="utf-8")
        file_paths["social_roi_explanation"] = str(social_roi_explanation_path)

    evidence_summary_path = output_dir / "evidence_summary.csv"
    evidence_summary_path.write_bytes(dataframe_to_csv_bytes(evidence_summary_df))
    file_paths["evidence_summary"] = str(evidence_summary_path)

    return file_paths
