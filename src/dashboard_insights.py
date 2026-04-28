from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame sin romper."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def build_executive_dashboard_summary(results: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    """Construye un resumen ejecutivo corto sin depender de Gemini."""
    impact_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    readiness = results.get("readiness", {})
    quality_gate = results.get("quality_gate_report", {})
    limitations = results.get("limitations", [])

    top_zone = "Sin identificar"
    if not impact_df.empty and {"zona", "final_impact_score"}.issubset(impact_df.columns):
        top_row = impact_df.sort_values("final_impact_score", ascending=False).iloc[0]
        top_zone = str(top_row.get("zona", "Sin identificar"))

    principal_risk = (
        f"La principal presión operativa se concentra en {top_zone}."
        if top_zone != "Sin identificar"
        else "Todavía no hay una zona crítica dominante con la evidencia actual."
    )

    recommendation = (
        "Ejecutar revisión humana de las órdenes prioritarias y validar pasaportes de decisión antes de escalar a campo."
        if not work_orders_df.empty
        else "Completar el mapeo operativo y ejecutar Mission Control para generar resultados accionables."
    )

    limitation_text = (
        limitations[0]
        if isinstance(limitations, list) and limitations
        else "El análisis depende de la calidad del dataset cargado y del mapeo de columnas."
    )

    if results.get("is_meraki_mode"):
        return (
            f"Estado general: paquete Meraki activo, readiness {readiness.get('classification', 'Sin clasificar')} "
            f"y quality gate {quality_gate.get('quality_gate', 'Sin evaluar')}. "
            f"{principal_risk} Recomendación inmediata: revisar el AP o zona priorizada y validar su orden de trabajo. "
            f"Limitación principal: {limitation_text}"
        )

    return (
        f"Estado general: readiness {readiness.get('classification', 'Sin clasificar')} "
        f"y quality gate {quality_gate.get('quality_gate', 'Sin evaluar')}. "
        f"{principal_risk} Recomendación inmediata: {recommendation} "
        f"Limitación principal: {limitation_text}"
    )


def build_top_findings(results: dict[str, Any]) -> list[str]:
    """Resume hallazgos ejecutivos principales."""
    impact_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    review_df = _safe_dataframe(results.get("human_review_log"))
    quality_gate = results.get("quality_gate_report", {})

    findings: list[str] = [f"Se identificaron {len(work_orders_df)} órdenes de trabajo."]
    if results.get("is_meraki_mode"):
        operational_mart = _safe_dataframe(results.get("operational_mart"))
        if not operational_mart.empty and "ap_name" in operational_mart.columns:
            findings.append(f"El mart operativo Meraki consolidó {operational_mart['ap_name'].astype(str).nunique()} APs.")

    if not impact_df.empty and {"zona", "final_impact_score"}.issubset(impact_df.columns):
        top_row = impact_df.sort_values("final_impact_score", ascending=False).iloc[0]
        findings.append(
            f"La zona con mayor score es {top_row.get('zona')} con {float(top_row.get('final_impact_score', 0)):.2f}."
        )

    findings.append(f"El quality gate está en estado {quality_gate.get('quality_gate', 'Sin evaluar')}.")

    if not impact_df.empty and "classification" in impact_df.columns:
        critical_or_high = int(impact_df["classification"].astype(str).isin(["Critico", "Alto"]).sum())
        findings.append(f"Hay {critical_or_high} zonas en clasificación crítica o alta.")

    if not review_df.empty and "estado_revision" in review_df.columns:
        pending = int(review_df["estado_revision"].astype(str).eq("pendiente").sum())
        findings.append(f"La validación humana tiene {pending} órdenes pendientes.")

    return findings


def build_next_best_actions(results: dict[str, Any]) -> list[str]:
    """Sugiere próximas acciones operativas."""
    impact_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    readiness = results.get("readiness", {})
    review_df = _safe_dataframe(results.get("human_review_log"))

    actions: list[str] = []
    if not work_orders_df.empty:
        actions.append("Revisar la orden prioritaria más urgente en Agente Operativo.")
    else:
        actions.append("Ejecutar Mission Control para generar órdenes de trabajo preliminares.")

    if readiness.get("score", 0) < 70:
        actions.append("Validar el mapeo de columnas y completar métricas operativas faltantes.")

    if impact_df.empty or "territorio" not in impact_df.columns:
        actions.append("Completar coordenadas o territorio para mejorar priorización geoespacial.")

    if results.get("replay_timeline") is None or _safe_dataframe(results.get("replay_timeline")).empty:
        actions.append("Ejecutar Simulación Operativa para observar evolución por lotes.")

    if not review_df.empty and "estado_revision" in review_df.columns:
        pending = int(review_df["estado_revision"].astype(str).eq("pendiente").sum())
        if pending > 0:
            actions.append("Cerrar la validación humana de las órdenes pendientes.")

    actions.append("Exportar el Paquete de Evidencia para compartir hallazgos con operación y tomadores de decisión.")
    return actions


def build_risk_alerts(results: dict[str, Any]) -> list[str]:
    """Identifica riesgos o alertas operativas clave."""
    alerts: list[str] = []
    readiness = results.get("readiness", {})
    quality_gate = results.get("quality_gate_report", {})
    impact_df = _safe_dataframe(results.get("impact_scores"))
    review_df = _safe_dataframe(results.get("human_review_log"))

    gap_text = " ".join(str(item) for item in readiness.get("gaps", []))
    if "fecha" in gap_text.lower():
        alerts.append("Falta fecha o histórico suficiente para análisis temporal robusto.")
    if any(keyword in gap_text.lower() for keyword in ["latitud", "longitud", "territorio", "geograf"]):
        alerts.append("Falta geografía suficiente para mapa o priorización territorial más confiable.")
    if results.get("is_synthetic_data"):
        alerts.append("El dataset activo es sintético y no representa operación oficial.")
    if results.get("is_meraki_mode"):
        alerts.append("El paquete Meraki no incluye coordenadas exactas de los APs; la priorización geográfica fina puede ser limitada.")
    if results.get("gemini_configured") is False:
        alerts.append("Gemini no está configurado; el análisis explicativo asistido no está disponible.")
    if quality_gate.get("quality_gate") == "Bloqueado":
        alerts.append("El quality gate está bloqueado; se recomienda corregir datos antes de operar.")
    if readiness.get("score", 0) < 50:
        alerts.append("El readiness score es bajo; el análisis actual tiene limitaciones relevantes.")

    if not review_df.empty and "estado_revision" in review_df.columns:
        pending = int(review_df["estado_revision"].astype(str).eq("pendiente").sum())
        if pending >= 5:
            alerts.append("Hay varias órdenes pendientes de validación humana.")

    if not impact_df.empty and "classification" in impact_df.columns:
        critical = int(impact_df["classification"].astype(str).eq("Critico").sum())
        if critical > 0:
            alerts.append(f"Existen {critical} zonas en estado crítico que requieren seguimiento prioritario.")

    if not alerts:
        alerts.append("No se detectan alertas estructurales adicionales con la evidencia actual.")
    return alerts
