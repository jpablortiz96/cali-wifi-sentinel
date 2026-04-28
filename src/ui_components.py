from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st


def inject_premium_css() -> None:
    """Inyecta una capa visual premium oscura y responsive para la app."""
    st.markdown(
        """
        <style>
        :root {
            --cw-bg: #08111f;
            --cw-panel: rgba(15, 23, 42, 0.78);
            --cw-panel-strong: rgba(15, 23, 42, 0.92);
            --cw-border: rgba(148, 163, 184, 0.18);
            --cw-text: #e2e8f0;
            --cw-muted: #94a3b8;
            --cw-blue: #3b82f6;
            --cw-cyan: #22d3ee;
            --cw-green: #10b981;
            --cw-yellow: #facc15;
            --cw-orange: #f97316;
            --cw-red: #ef4444;
            --cw-purple: #8b5cf6;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            overflow-x: hidden !important;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.12), transparent 28%),
                linear-gradient(180deg, #06101d 0%, #0b1220 100%);
            color: var(--cw-text);
        }

        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 3.5rem;
            max-width: 1480px;
        }

        section[data-testid="stSidebar"] > div {
            background: rgba(7, 12, 22, 0.78);
            border-right: 1px solid var(--cw-border);
        }

        div[data-testid="stMetric"] {
            background: var(--cw-panel);
            border: 1px solid var(--cw-border);
            backdrop-filter: blur(12px);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 16px 40px rgba(2, 8, 23, 0.25);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
            padding-bottom: 0.2rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid transparent;
            color: var(--cw-muted);
            white-space: nowrap;
            padding: 0.55rem 0.9rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--cw-text);
            background: rgba(59, 130, 246, 0.16);
            border-color: rgba(59, 130, 246, 0.28);
        }

        .cw-kpi-card,
        .cw-insight-card,
        .cw-action-card,
        .cw-empty-card {
            background: var(--cw-panel);
            border: 1px solid var(--cw-border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 40px rgba(2, 8, 23, 0.24);
            backdrop-filter: blur(14px);
            margin-bottom: 0.8rem;
        }

        .cw-kpi-title,
        .cw-insight-title,
        .cw-action-title,
        .cw-empty-title {
            color: var(--cw-muted);
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .cw-kpi-value {
            color: var(--cw-text);
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .cw-kpi-subtitle,
        .cw-insight-body,
        .cw-action-body,
        .cw-empty-body {
            color: var(--cw-text);
            font-size: 0.95rem;
            line-height: 1.45;
            margin-top: 0.35rem;
        }

        .cw-chip {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .cw-header-block {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(15, 23, 42, 0.82));
            border: 1px solid var(--cw-border);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 18px 48px rgba(2, 8, 23, 0.3);
        }

        .cw-header-title {
            color: var(--cw-text);
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }

        .cw-header-subtitle {
            color: var(--cw-muted);
            font-size: 1rem;
            margin-top: 0.3rem;
        }

        .cw-flow-banner {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid var(--cw-border);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.95rem;
            color: var(--cw-text);
        }

        .cw-back-to-top {
            position: fixed;
            right: 22px;
            bottom: 22px;
            z-index: 9999;
            min-width: 58px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.92), rgba(59, 130, 246, 0.94));
            color: white;
            text-decoration: none;
            font-size: 0.96rem;
            font-weight: 700;
            box-shadow: 0 18px 42px rgba(2, 8, 23, 0.35);
            border: 1px solid rgba(255,255,255,0.18);
            padding: 0 0.95rem;
        }

        .cw-back-to-top:hover {
            filter: brightness(1.08);
        }

        div[data-testid="stPopover"] {
            position: fixed !important;
            right: 22px;
            bottom: 95px;
            z-index: 9998;
            width: auto !important;
            max-width: none !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stPopover"] > div {
            gap: 0 !important;
        }

        div[data-testid="stPopover"] > button {
            width: 60px;
            min-width: 60px;
            height: 60px;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.96), rgba(59, 130, 246, 0.96));
            color: white;
            font-size: 1.2rem;
            font-weight: 700;
            box-shadow: 0 18px 42px rgba(2, 8, 23, 0.35);
            border: 1px solid rgba(255,255,255,0.18);
            padding: 0;
            line-height: 1;
        }

        div[data-testid="stPopover"] > button:hover {
            filter: brightness(1.08);
            border-color: rgba(255,255,255,0.3);
        }

        .cw-chat-panel-header {
            display: flex;
            align-items: flex-start;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            margin-bottom: 0.75rem;
        }

        .cw-chat-title {
            color: var(--cw-text);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .cw-chat-subtitle {
            color: var(--cw-muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .cw-chat-history {
            max-height: 290px;
            overflow-y: auto;
            padding-right: 0.2rem;
            margin-bottom: 0.35rem;
        }

        .cw-chat-message {
            display: flex;
            margin-bottom: 0.55rem;
        }

        .cw-chat-message.user {
            justify-content: flex-end;
        }

        .cw-chat-message.assistant {
            justify-content: flex-start;
        }

        .cw-chat-bubble {
            max-width: 88%;
            padding: 0.62rem 0.78rem;
            border-radius: 16px;
            font-size: 0.88rem;
            line-height: 1.4;
            box-shadow: 0 10px 24px rgba(2, 8, 23, 0.18);
            white-space: pre-wrap;
            word-break: break-word;
        }

        .cw-chat-bubble.user {
            background: rgba(59, 130, 246, 0.22);
            color: #eff6ff;
            border: 1px solid rgba(59, 130, 246, 0.38);
        }

        .cw-chat-bubble.assistant {
            background: rgba(15, 23, 42, 0.88);
            color: var(--cw-text);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .cw-chat-empty {
            color: var(--cw-muted);
            font-size: 0.85rem;
            padding: 0.45rem 0;
        }

        div[data-baseweb="popover"] .stTextInput > div > div > input {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: var(--cw-text);
        }

        div[data-baseweb="popover"] div.stButton > button,
        div[data-baseweb="popover"] div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
        }

        div[data-baseweb="popover"] p {
            margin-bottom: 0;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--cw-border);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.55);
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
                padding-top: 0.8rem;
            }

            .cw-header-title {
                font-size: 1.55rem;
            }

            .cw-header-subtitle {
                font-size: 0.92rem;
            }

            .cw-kpi-card,
            .cw-insight-card,
            .cw-action-card,
            .cw-empty-card {
                padding: 0.85rem 0.9rem;
                border-radius: 16px;
            }

            .cw-kpi-value {
                font-size: 1.45rem;
            }

            button[data-baseweb="tab"] {
                padding: 0.45rem 0.7rem;
                font-size: 0.84rem;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                width: 100%;
            }

            .cw-back-to-top {
                width: 46px;
                height: 46px;
                right: 16px;
                bottom: 16px;
                min-width: 46px;
                padding: 0 0.6rem;
                font-size: 0.9rem;
            }

            div[data-testid="stPopover"] {
                right: 16px;
                bottom: 84px;
            }

            div[data-testid="stPopover"] > button {
                width: 54px;
                min-width: 54px;
                height: 54px;
                font-size: 1.08rem;
            }

            div[data-baseweb="popover"] {
                width: 90vw !important;
                max-width: 90vw !important;
            }

            .cw-chat-history {
                max-height: 36vh;
            }

            .js-plotly-plot,
            .plotly-graph-div {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_back_to_top_button() -> None:
    """Inyecta un botón flotante para volver arriba."""
    st.markdown(
        """
        <a class="cw-back-to-top" href="#top-anchor" title="Volver al menú principal">⬆️</a>
        """,
        unsafe_allow_html=True,
    )


def render_floating_chat_widget(
    messages: list[dict[str, str]] | None = None,
    description: str | None = None,
    gemini_configured: bool = True,
) -> tuple[bool, str]:
    """Renderiza un widget flotante de chat global para el agente de plataforma."""
    recent_messages = list(messages or [])[-12:]

    with st.popover("💬", key="floating_chat_popover", help="Abrir agente conversacional", width="content"):
        st.markdown(
            """
            <div class="cw-chat-panel-header">
                <div class="cw-chat-title">🤖 Agente de Plataforma</div>
                <div class="cw-chat-subtitle">Preguntas sobre la plataforma y el análisis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if description:
            st.caption(description)
        if not gemini_configured:
            st.caption("Gemini no está configurado. Responderé con ayuda contextual básica de la plataforma.")

        if recent_messages:
            history_html = ["<div class='cw-chat-history'>"]
            for message in recent_messages:
                role = "user" if message.get("role") == "user" else "assistant"
                content = html.escape(str(message.get("content", "")))
                history_html.append(
                    f"<div class='cw-chat-message {role}'><div class='cw-chat-bubble {role}'>{content}</div></div>"
                )
            history_html.append("</div>")
            st.markdown("".join(history_html), unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='cw-chat-empty'>Aún no hay conversación. Pregunta sobre módulos, indicadores, órdenes o análisis.</div>",
                unsafe_allow_html=True,
            )

        with st.form("floating_chat_form", clear_on_submit=True):
            question = st.text_input(
                "Pregunta al agente",
                key="floating_agent_question",
                placeholder="Pregunta sobre módulos, indicadores, órdenes o análisis...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Enviar", use_container_width=True)
    return send, question


def render_floating_agent_button(
    messages: list[dict[str, str]] | None = None,
    description: str | None = None,
    gemini_configured: bool = True,
) -> tuple[bool, str]:
    """Compatibilidad retroactiva con el nombre anterior del widget flotante."""
    return render_floating_chat_widget(
        messages=messages,
        description=description,
        gemini_configured=gemini_configured,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Muestra un encabezado consistente para secciones principales."""
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def render_status_badge(label: str, status: str) -> None:
    """Muestra un estado con color simple y legible."""
    normalized = str(status).strip().lower()
    color_map = {
        "ok": "#0f766e",
        "warning": "#b45309",
        "error": "#b91c1c",
        "listo": "#0f766e",
        "limitado": "#b45309",
        "bloqueado": "#b91c1c",
        "critico": "#b91c1c",
        "alto": "#b45309",
        "medio": "#1d4ed8",
        "bajo": "#4b5563",
    }
    color = color_map.get(normalized, "#334155")
    st.markdown(
        (
            f"<div style='display:inline-block;padding:0.35rem 0.65rem;border-radius:999px;"
            f"background:{color}15;border:1px solid {color}40;color:{color};font-size:0.9rem;'>"
            f"<strong>{label}:</strong> {status}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str, action_hint: str | None = None) -> None:
    """Muestra un estado vacío compacto y elegante."""
    hint_html = f"<div class='cw-empty-body' style='opacity:0.8;'>{action_hint}</div>" if action_hint else ""
    st.markdown(
        (
            "<div class='cw-empty-card'>"
            f"<div class='cw-empty-title'>{title}</div>"
            f"<div class='cw-empty-body'>{message}</div>"
            f"{hint_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: dict[str, object]) -> None:
    """Renderiza una fila de métricas simples."""
    if not metrics:
        return

    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        if isinstance(value, dict):
            column.metric(label, value.get("value"), delta=value.get("delta"))
        else:
            column.metric(label, value)


def render_dataframe_clean(df: pd.DataFrame | None, title: str | None = None, height: int = 350) -> None:
    """Muestra dataframes limpios y maneja vacíos sin ruido."""
    if title:
        st.markdown(f"**{title}**")

    if df is None or df.empty:
        st.caption("No hay datos disponibles para esta vista.")
        return

    st.dataframe(df, use_container_width=True, height=height)


def render_download_buttons_for_evidence(files_dict: dict[str, str]) -> None:
    """Renderiza descargas desde archivos ya generados en disco."""
    if not files_dict:
        st.caption("No hay archivos generados en disco todavía.")
        return

    mime_map = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }

    for label, path_str in sorted(files_dict.items()):
        file_path = Path(path_str)
        if not file_path.exists() or not file_path.is_file():
            continue

        mime = mime_map.get(file_path.suffix.lower(), "application/octet-stream")
        st.download_button(
            f"Descargar {file_path.name}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
            key=f"generated_file_{label}",
        )


def render_premium_kpi_card(
    title: str,
    value: object,
    subtitle: str | None = None,
    status: str | None = None,
    icon: str | None = None,
) -> None:
    """Renderiza una tarjeta KPI oscura y compacta."""
    status_colors = {
        "ok": "#10b981",
        "warning": "#facc15",
        "error": "#ef4444",
        "critical": "#ef4444",
        "high": "#f97316",
        "medium": "#22d3ee",
        "low": "#94a3b8",
        "neutral": "#94a3b8",
        "info": "#22d3ee",
    }
    chip = ""
    if status:
        color = status_colors.get(status.lower(), "#94a3b8")
        chip = (
            f"<span class='cw-chip' style='color:{color};background:{color}22;border-color:{color}55;'>"
            f"{status}</span>"
        )
    prefix = f"{icon} " if icon else ""
    subtitle_html = f"<div class='cw-kpi-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        (
            "<div class='cw-kpi-card'>"
            f"<div class='cw-kpi-title'>{prefix}{title}</div>"
            f"<div class='cw-kpi-value'>{value}</div>"
            f"{subtitle_html}"
            f"{chip}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, body: str, status: str = "info") -> None:
    """Tarjeta para hallazgos ejecutivos."""
    color_map = {
        "info": "#22d3ee",
        "success": "#10b981",
        "warning": "#facc15",
        "error": "#ef4444",
    }
    color = color_map.get(status, "#22d3ee")
    st.markdown(
        (
            "<div class='cw-insight-card'>"
            f"<div class='cw-insight-title' style='color:{color};'>{title}</div>"
            f"<div class='cw-insight-body'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_action_card(title: str, body: str, priority: str = "media") -> None:
    """Tarjeta para próximas acciones."""
    color_map = {
        "alta": "#ef4444",
        "media": "#f97316",
        "baja": "#22d3ee",
    }
    color = color_map.get(priority, "#f97316")
    st.markdown(
        (
            "<div class='cw-action-card'>"
            f"<div class='cw-action-title' style='color:{color};'>{title}</div>"
            f"<div class='cw-action-body'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_compact_badge(text: str, status: str = "neutral") -> None:
    """Badge compacto para estados o etiquetas."""
    color_map = {
        "neutral": "#94a3b8",
        "ok": "#10b981",
        "warning": "#facc15",
        "error": "#ef4444",
        "info": "#22d3ee",
    }
    color = color_map.get(status, "#94a3b8")
    st.markdown(
        (
            f"<span class='cw-chip' style='color:{color};background:{color}22;border-color:{color}55;'>"
            f"{text}</span>"
        ),
        unsafe_allow_html=True,
    )


def render_dashboard_empty_state() -> None:
    """Estado vacío elegante para la vista ejecutiva."""
    render_empty_state(
        "Vista Ejecutiva 360 sin resultados",
        "Ejecuta Mission Control o Simulación Operativa para alimentar la Vista Ejecutiva 360.",
        action_hint="Ruta sugerida: 1) Carga dataset 2) Mapea columnas 3) Ejecuta Mission Control 4) Vuelve a esta vista.",
    )


def render_json_advanced(label: str, data: object) -> None:
    """Deja JSON solo como vista técnica avanzada."""
    with st.expander(f"Ver estructura técnica avanzada: {label}", expanded=False):
        st.json(data)
