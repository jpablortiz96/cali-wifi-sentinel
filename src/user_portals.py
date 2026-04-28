from __future__ import annotations

import streamlit as st


TECHNICAL_PROFILE = "Técnico / Operativo"
CITIZEN_PROFILE = "Ciudadano / Impacto Social"


def get_technical_sections() -> list[str]:
    """Devuelve las secciones visibles para el backoffice operativo."""
    return [
        "Carga e Inspeccion",
        "Mapeo de Columnas",
        "Mission Control",
        "Simulacion Operativa",
        "Vista Ejecutiva 360",
        "Agente Operativo",
        "Impacto Ciudadano",
        "Cuadrillas",
        "Pasaporte de Decision",
        "Agente Estrategico",
        "Validacion Humana",
        "Blindaje Tecnico",
        "Auditoria Operativa",
        "Paquete de Evidencia",
    ]


def get_citizen_sections() -> list[str]:
    """Devuelve las secciones visibles para el portal ciudadano e impacto social."""
    return [
        "Portal Ciudadano",
        "Experiencia Ciudadana",
        "Recomendador de Zonas WiFi",
        "Buzon Ciudadano",
        "Equidad Digital",
        "Retorno Social de Conectividad",
        "Agente Ciudadano",
        "Vista Publica de Calidad",
    ]


def get_default_section_for_profile(profile: str) -> str:
    """Devuelve la primera sección recomendada para el perfil activo."""
    if profile == CITIZEN_PROFILE:
        return get_citizen_sections()[0]
    return get_technical_sections()[0]


def render_profile_selector() -> str:
    """Renderiza el selector principal de perfil de usuario."""
    options = [TECHNICAL_PROFILE, CITIZEN_PROFILE]
    return st.sidebar.radio(
        "Selecciona el tipo de usuario",
        options=options,
        index=0,
        key="user_profile_selector",
        help="Cambia la navegación para ver solo módulos relevantes para operación o impacto social.",
    )
