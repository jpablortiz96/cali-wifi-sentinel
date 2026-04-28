from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.gemini_client import generate_gemini_text, is_gemini_configured


def _top_records(dataframe: pd.DataFrame | None, limit: int = 5) -> list[dict[str, object]]:
    """Convierte un DataFrame a pocos registros serializables."""
    if dataframe is None or dataframe.empty:
        return []
    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_citizen_insights_context(
    citizen_scores: pd.DataFrame,
    recommendations: pd.DataFrame,
    feedback_summary: dict[str, Any],
    equity_df: pd.DataFrame,
    calendar_summary: dict[str, Any],
) -> dict[str, object]:
    """Construye contexto agregado y seguro para análisis ciudadano."""
    scores_df = citizen_scores.copy() if isinstance(citizen_scores, pd.DataFrame) else pd.DataFrame()
    recommendations_df = recommendations.copy() if isinstance(recommendations, pd.DataFrame) else pd.DataFrame()
    equity_frame = equity_df.copy() if isinstance(equity_df, pd.DataFrame) else pd.DataFrame()

    best_zones = pd.DataFrame()
    worst_zones = pd.DataFrame()
    if not scores_df.empty and "citizen_experience_score" in scores_df.columns:
        best_zones = scores_df.sort_values("citizen_experience_score", ascending=False).head(5)
        worst_zones = scores_df.sort_values("citizen_experience_score", ascending=True).head(5)

    return {
        "citizen_scores_count": int(len(scores_df)),
        "top_best_zones": _top_records(best_zones, limit=5),
        "top_worst_zones": _top_records(worst_zones, limit=5),
        "citizen_recommendations": _top_records(recommendations_df, limit=5),
        "feedback_summary": feedback_summary or {},
        "equity_proxy_top": _top_records(equity_frame, limit=5),
        "calendar_summary": calendar_summary or {},
        "limitations": [
            "Los resultados usan datos agregados por AP/zona/hora y no identifican personas.",
            "El proxy de equidad digital no representa población real ni confirma brechas por sí solo.",
            "Las mejores zonas y horarios son estimaciones basadas en evidencia observada en el dataset cargado.",
        ],
    }


def generate_citizen_insights_with_gemini(context: dict[str, object]) -> str:
    """Genera análisis ciudadano claro con Gemini, solo sobre agregados reales."""
    prompt = (
        "Eres un agente de análisis ciudadano para una red WiFi pública. "
        "Debes explicar en lenguaje claro qué zonas ofrecen mejor experiencia, dónde hay alertas, "
        "qué recomienda el sistema a ciudadanos y qué debe revisar la Alcaldía. "
        "No inventes datos. No uses identificadores individuales. "
        "No afirmes brechas reales si solo hay proxy. "
        "No afirmes causalidad climática.\n\n"
        "Usa solo este contexto agregado:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Entrega tu respuesta en Markdown con estas secciones exactas:\n"
        "## Resumen ciudadano\n"
        "## Mejores zonas para conectarse\n"
        "## Zonas con alerta\n"
        "## Recomendaciones para usuarios\n"
        "## Recomendaciones para la Alcaldía\n"
        "## Limitaciones"
    )
    return generate_gemini_text(prompt)


def fallback_citizen_insights(context: dict[str, object]) -> str:
    """Genera un resumen determinístico cuando Gemini no está disponible."""
    best_zones = context.get("top_best_zones", [])
    worst_zones = context.get("top_worst_zones", [])
    recommendations = context.get("citizen_recommendations", [])
    feedback_summary = context.get("feedback_summary", {})
    equity_proxy_top = context.get("equity_proxy_top", [])
    limitations = context.get("limitations", [])

    best_lines = [
        f"- {item.get('zone_name') or item.get('ap_name')}: score {item.get('citizen_experience_score', 'N/A')}"
        for item in best_zones[:5]
    ] or ["- Aún no hay zonas destacadas con la evidencia actual."]
    worst_lines = [
        f"- {item.get('zone_name') or item.get('ap_name')}: score {item.get('citizen_experience_score', 'N/A')}"
        for item in worst_zones[:5]
    ] or ["- No se detectan zonas con alerta fuerte todavía."]
    recommendation_lines = [
        f"- {item.get('zone_name') or item.get('ap_name')}: {item.get('motivo', 'Sin explicación adicional.')}"
        for item in recommendations[:5]
    ] or ["- Usa las zonas con mejor score y evita las franjas inestables."]
    mayor_action_lines = [
        f"- {item.get('zone_name')}: {item.get('equity_label', 'Revisión recomendada')}."
        for item in equity_proxy_top[:5]
    ] or ["- Completar más evidencia agregada antes de priorizar intervenciones ciudadanas."]
    limitation_lines = [f"- {item}" for item in limitations] or ["- Sin limitaciones adicionales registradas."]

    feedback_line = (
        f"- Reportes ciudadanos: {feedback_summary.get('total_reportes', 0)} | "
        f"sentimiento general: {feedback_summary.get('sentimiento_general', 'Sin reportes')}."
    )

    return "\n".join(
        [
            "## Resumen ciudadano",
            "Gemini no está configurado. Se muestra análisis determinístico básico sobre datos agregados.",
            feedback_line,
            "",
            "## Mejores zonas para conectarse",
            *best_lines,
            "",
            "## Zonas con alerta",
            *worst_lines,
            "",
            "## Recomendaciones para usuarios",
            *recommendation_lines,
            "",
            "## Recomendaciones para la Alcaldía",
            *mayor_action_lines,
            "",
            "## Limitaciones",
            *limitation_lines,
        ]
    )
