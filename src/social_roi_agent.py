from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.gemini_client import generate_gemini_text


def _top_records(dataframe: pd.DataFrame | None, limit: int = 8) -> list[dict[str, object]]:
    """Convierte una vista corta del DataFrame en registros serializables."""
    if dataframe is None or dataframe.empty:
        return []
    preview = dataframe.head(limit).copy().astype(object)
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def build_social_roi_context(
    social_roi_df: pd.DataFrame,
    socioeconomic_validation: dict[str, Any],
    limitations: list[str],
) -> dict[str, object]:
    """Construye un contexto agregado y responsable para explicar retorno social."""
    roi_df = social_roi_df.copy() if isinstance(social_roi_df, pd.DataFrame) else pd.DataFrame()
    top_roi = pd.DataFrame()
    if not roi_df.empty and "social_roi_score" in roi_df.columns:
        top_roi = roi_df.sort_values("social_roi_score", ascending=False).head(10)

    return {
        "top_social_roi_zones": _top_records(top_roi, limit=10),
        "available_indicators": socioeconomic_validation.get("available_indicators", []),
        "geo_level": socioeconomic_validation.get("level", "desconocido"),
        "warnings": socioeconomic_validation.get("warnings", []),
        "privacy_warnings": socioeconomic_validation.get("privacy_warnings", []),
        "methodology": {
            "formula": (
                "0.30 vulnerabilidad socioeconómica + 0.25 necesidad digital + "
                "0.20 riesgo de red + 0.15 potencial ciudadano + 0.10 confianza de datos"
            ),
            "note": "El score es un apoyo a la decisión con datos agregados. No usa población individual ni inferencias personales.",
        },
        "limitations": limitations or [],
    }


def generate_social_roi_explanation_with_gemini(context: dict[str, object]) -> str:
    """Solicita a Gemini una explicación ejecutiva del retorno social usando solo agregados."""
    prompt = (
        "Eres un agente estratégico de inversión social en conectividad pública. "
        "Debes explicar dónde una mejora de infraestructura WiFi tendría mayor retorno social, "
        "usando únicamente los indicadores agregados entregados. "
        "No inventes pobreza, desempleo ni población. No estigmatices zonas. "
        "Diferencia hechos, hipótesis, recomendaciones y limitaciones.\n\n"
        "Contexto agregado:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Responde en Markdown con estas secciones exactas:\n"
        "## Resumen de retorno social\n"
        "## Zonas con mayor prioridad social\n"
        "## Recomendaciones de infraestructura\n"
        "## Recomendaciones de acompañamiento social\n"
        "## Datos faltantes\n"
        "## Limitaciones responsables"
    )
    return generate_gemini_text(prompt)


def fallback_social_roi_explanation(context: dict[str, object]) -> str:
    """Entrega una explicación básica cuando Gemini no está disponible."""
    top_zones = context.get("top_social_roi_zones", [])
    available_indicators = context.get("available_indicators", [])
    warnings = context.get("warnings", [])
    privacy_warnings = context.get("privacy_warnings", [])
    limitations = context.get("limitations", [])

    zone_lines = [
        f"- {row.get('zone_name', 'Zona sin nombre')}: score {row.get('social_roi_score', 'N/A')} | {row.get('social_roi_label', 'Sin etiqueta')}"
        for row in top_zones[:5]
    ] or ["- No hay zonas priorizadas todavía."]
    indicator_lines = [f"- {value}" for value in available_indicators] or ["- No hay indicadores socioeconómicos agregados disponibles."]
    warning_lines = [f"- {value}" for value in warnings] or ["- No se registran advertencias adicionales."]
    privacy_lines = [f"- {value}" for value in privacy_warnings] or ["- No se detectaron riesgos adicionales de privacidad en el archivo cargado."]
    limitation_lines = [f"- {value}" for value in limitations] or ["- Sin limitaciones adicionales registradas."]

    return "\n".join(
        [
            "## Resumen de retorno social",
            "Gemini no está configurado. Se muestra una explicación determinística basada en el score ya calculado.",
            "",
            "## Zonas con mayor prioridad social",
            *zone_lines,
            "",
            "## Recomendaciones de infraestructura",
            "- Priorizar refuerzo de conectividad en las zonas con score más alto y riesgo operativo observable.",
            "- Validar en campo la cobertura real cuando la evidencia territorial sea intermedia.",
            "",
            "## Recomendaciones de acompañamiento social",
            "- Revisar señalización, apropiación digital o articulación con equipamientos comunitarios cuando aplique.",
            "",
            "## Datos faltantes",
            *indicator_lines,
            *warning_lines,
            "",
            "## Limitaciones responsables",
            *privacy_lines,
            *limitation_lines,
        ]
    )
