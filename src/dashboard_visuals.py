from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.geo_visuals import create_cali_priority_map_pro


COLOR_PALETTE = {
    "electric_blue": "#3b82f6",
    "cyan": "#22d3ee",
    "success": "#10b981",
    "warning": "#facc15",
    "high": "#f97316",
    "critical": "#ef4444",
    "soft_gray": "#94a3b8",
    "tech_purple": "#8b5cf6",
    "bg": "rgba(7, 12, 22, 0)",
    "panel": "#0f172a",
    "grid": "rgba(148, 163, 184, 0.15)",
    "font": "#e2e8f0",
}


CLASSIFICATION_COLORS = {
    "Critico": COLOR_PALETTE["critical"],
    "Alto": COLOR_PALETTE["high"],
    "Medio": COLOR_PALETTE["warning"],
    "Bajo": COLOR_PALETTE["success"],
    "Observacion": COLOR_PALETTE["cyan"],
}


def get_plotly_template() -> dict[str, Any]:
    """Devuelve un layout premium oscuro y compacto para Plotly."""
    return {
        "paper_bgcolor": COLOR_PALETTE["bg"],
        "plot_bgcolor": COLOR_PALETTE["bg"],
        "font": {"family": "Segoe UI, Inter, Arial, sans-serif", "color": COLOR_PALETTE["font"], "size": 13},
        "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(15, 23, 42, 0.0)",
        },
        "xaxis": {
            "showgrid": True,
            "gridcolor": COLOR_PALETTE["grid"],
            "zeroline": False,
            "linecolor": COLOR_PALETTE["grid"],
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": COLOR_PALETTE["grid"],
            "zeroline": False,
            "linecolor": COLOR_PALETTE["grid"],
        },
        "hoverlabel": {
            "bgcolor": "#111827",
            "bordercolor": "#334155",
            "font": {"color": COLOR_PALETTE["font"]},
        },
    }


def _apply_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Aplica el template premium a una figura."""
    fig.update_layout(**get_plotly_template())
    if title:
        fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
    return fig


def _empty_figure(title: str, subtitle: str) -> go.Figure:
    """Crea una figura vacía compacta."""
    fig = go.Figure()
    _apply_layout(fig, title)
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    fig.add_annotation(
        text=subtitle,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": COLOR_PALETTE["soft_gray"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Normaliza estructuras conocidas a DataFrame."""
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


def _series_or_default(dataframe: pd.DataFrame, column_name: str, default_value: object = "") -> pd.Series:
    """Devuelve una serie existente o una serie por defecto del mismo largo."""
    if column_name in dataframe.columns:
        return dataframe[column_name]
    return pd.Series([default_value] * len(dataframe), index=dataframe.index)


def create_priority_bar_chart(impact_scores_df: pd.DataFrame) -> go.Figure:
    """Top zonas por score final."""
    df = _safe_dataframe(impact_scores_df)
    if df.empty or "zona" not in df.columns or "final_impact_score" not in df.columns:
        return _empty_figure("Top zonas por impacto", "No hay scores de impacto disponibles.")

    top_df = df.sort_values("final_impact_score", ascending=False).head(15).copy()
    top_df["hover_explicacion"] = _series_or_default(top_df, "explanation_short", "").astype(str)
    top_df["hover_limitaciones"] = _series_or_default(top_df, "limitations", "").astype(str)
    fig = px.bar(
        top_df.sort_values("final_impact_score", ascending=True),
        x="final_impact_score",
        y="zona",
        color="classification" if "classification" in top_df.columns else None,
        orientation="h",
        color_discrete_map=CLASSIFICATION_COLORS,
        hover_data={
            "final_impact_score": ":.2f",
            "hover_explicacion": True,
            "hover_limitaciones": True,
            "classification": True if "classification" in top_df.columns else False,
        },
    )
    fig.update_traces(marker_line_width=0, opacity=0.95)
    fig.update_xaxes(title="Score final")
    fig.update_yaxes(title="")
    return _apply_layout(fig, "Top zonas por impacto")


def create_classification_donut(impact_scores_df: pd.DataFrame) -> go.Figure:
    """Distribución de zonas por clasificación."""
    df = _safe_dataframe(impact_scores_df)
    if df.empty or "classification" not in df.columns:
        return _empty_figure("Distribución por clasificación", "No hay clasificaciones disponibles.")

    summary = df["classification"].fillna("Observacion").value_counts().reset_index()
    summary.columns = ["classification", "count"]
    fig = px.pie(
        summary,
        values="count",
        names="classification",
        hole=0.58,
        color="classification",
        color_discrete_map=CLASSIFICATION_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_layout(fig, "Distribución por clasificación")


def create_score_component_radar(impact_scores_df: pd.DataFrame, selected_zone: str | None = None) -> go.Figure:
    """Radar de componentes del score."""
    df = _safe_dataframe(impact_scores_df)
    component_columns = {
        "Severidad técnica": "technical_severity_score",
        "Demanda": "demand_score",
        "Criticidad social": "social_criticality_score",
        "Confianza datos": "data_confidence_score",
        "Clima/contexto": "weather_context_score",
    }
    if df.empty or "zona" not in df.columns:
        return _empty_figure("Componentes del score", "No hay datos disponibles para construir el radar.")

    if selected_zone and selected_zone in df["zona"].astype(str).tolist():
        base_row = df[df["zona"].astype(str) == str(selected_zone)].iloc[0]
        title = f"Componentes del score: {selected_zone}"
    else:
        base_candidates = df.sort_values("final_impact_score", ascending=False).head(5)
        base_row = base_candidates[list(component_columns.values())].mean(numeric_only=True)
        title = "Componentes promedio de zonas prioritarias"

    values = [float(base_row.get(column_name, 0) or 0) for column_name in component_columns.values()]
    categories = list(component_columns.keys())
    values.append(values[0] if values else 0.0)
    categories.append(categories[0])

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                line={"color": COLOR_PALETTE["tech_purple"], "width": 3},
                fillcolor="rgba(139, 92, 246, 0.25)",
                name="Score",
            )
        ]
    )
    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": COLOR_PALETTE["grid"],
                "linecolor": COLOR_PALETTE["grid"],
            },
            "angularaxis": {"gridcolor": COLOR_PALETTE["grid"], "linecolor": COLOR_PALETTE["grid"]},
            "bgcolor": COLOR_PALETTE["bg"],
        }
    )
    return _apply_layout(fig, title)


def create_territory_heatmap(impact_scores_df: pd.DataFrame, territory_col: str | None = None) -> go.Figure | None:
    """Heatmap territorial por clasificación y score promedio."""
    df = _safe_dataframe(impact_scores_df)
    territory_field = territory_col or ("territorio" if "territorio" in df.columns else None)
    if df.empty or not territory_field or territory_field not in df.columns or "classification" not in df.columns:
        return None

    clean_df = df.dropna(subset=[territory_field]).copy()
    if clean_df.empty:
        return None

    matrix_df = (
        clean_df.groupby([territory_field, "classification"], dropna=False)["final_impact_score"]
        .mean()
        .reset_index()
    )
    if matrix_df.empty:
        return None

    pivot_df = matrix_df.pivot(index=territory_field, columns="classification", values="final_impact_score").fillna(0)
    fig = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#0f172a"],
            [0.2, COLOR_PALETTE["electric_blue"]],
            [0.5, COLOR_PALETTE["warning"]],
            [0.75, COLOR_PALETTE["high"]],
            [1.0, COLOR_PALETTE["critical"]],
        ],
        labels={"x": "Clasificación", "y": "Territorio", "color": "Score promedio"},
    )
    return _apply_layout(fig, "Heatmap territorial")


def create_impact_scatter(impact_scores_df: pd.DataFrame) -> go.Figure:
    """Dispersión de severidad técnica vs demanda."""
    df = _safe_dataframe(impact_scores_df)
    required = {"demand_score", "technical_severity_score", "final_impact_score", "zona"}
    if df.empty or not required.issubset(df.columns):
        return _empty_figure("Dispersión de impacto", "No hay componentes suficientes para este gráfico.")

    fig = px.scatter(
        df,
        x="demand_score",
        y="technical_severity_score",
        size="final_impact_score",
        color="classification" if "classification" in df.columns else None,
        color_discrete_map=CLASSIFICATION_COLORS,
        hover_name="zona",
        hover_data={
            "final_impact_score": ":.2f",
            "explanation_short": True if "explanation_short" in df.columns else False,
        },
        size_max=36,
    )
    fig.update_xaxes(title="Demanda")
    fig.update_yaxes(title="Severidad técnica")
    return _apply_layout(fig, "Demanda vs severidad técnica")


def create_replay_timeline_chart(replay_timeline_df: pd.DataFrame) -> go.Figure | None:
    """Timeline consolidado de la simulación."""
    df = _safe_dataframe(replay_timeline_df)
    if df.empty or "step" not in df.columns:
        return None

    fig = go.Figure()
    uses_meraki_series = False
    if "work_orders_count" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["work_orders_count"],
                mode="lines+markers",
                name="Órdenes",
                line={"color": COLOR_PALETTE["electric_blue"], "width": 3},
            )
        )
    if "critical_zones_count" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["critical_zones_count"],
                mode="lines+markers",
                name="Zonas críticas",
                line={"color": COLOR_PALETTE["critical"], "width": 3},
            )
        )
    if "critical_aps_count" in df.columns and pd.to_numeric(df["critical_aps_count"], errors="coerce").fillna(0).sum() > 0:
        uses_meraki_series = True
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["critical_aps_count"],
                mode="lines+markers",
                name="APs críticos/altos",
                line={"color": COLOR_PALETTE["high"], "width": 3},
            )
        )
    if "total_connections" in df.columns and pd.to_numeric(df["total_connections"], errors="coerce").fillna(0).sum() > 0:
        uses_meraki_series = True
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["total_connections"],
                mode="lines+markers",
                name="Conexiones",
                line={"color": COLOR_PALETTE["cyan"], "width": 3},
                yaxis="y2",
            )
        )
    if "total_disconnections" in df.columns and pd.to_numeric(df["total_disconnections"], errors="coerce").fillna(0).sum() > 0:
        uses_meraki_series = True
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["total_disconnections"],
                mode="lines+markers",
                name="Desconexiones",
                line={"color": COLOR_PALETTE["tech_purple"], "width": 3, "dash": "dot"},
                yaxis="y2",
            )
        )
    if "top_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["top_score"],
                mode="lines+markers",
                name="Score máximo",
                line={"color": COLOR_PALETTE["warning"], "width": 3, "dash": "dot"},
                yaxis="y2",
            )
        )

    fig.update_layout(
        yaxis={"title": "Órdenes / zonas" if not uses_meraki_series else "Órdenes / criticidad"},
        yaxis2={
            "title": "Score / conexiones / desconexiones" if uses_meraki_series else "Score máximo",
            "overlaying": "y",
            "side": "right",
        },
        xaxis={"title": "Paso"},
        height=360,
    )
    title = "Evolución horaria de APs y órdenes" if uses_meraki_series else "Evolución de la simulación"
    return _apply_layout(fig, title)


def create_work_order_status_chart(
    work_orders_df: pd.DataFrame | None = None,
    review_queue_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Distribución por prioridad o estado de revisión."""
    review_df = _safe_dataframe(review_queue_df)
    if not review_df.empty and "estado_revision" in review_df.columns:
        summary = review_df["estado_revision"].fillna("pendiente").value_counts().reset_index()
        summary.columns = ["categoria", "count"]
        title = "Estado de revisión de órdenes"
    else:
        orders_df = _safe_dataframe(work_orders_df)
        if orders_df.empty or "prioridad" not in orders_df.columns:
            return _empty_figure("Órdenes y validación", "No hay órdenes o validaciones disponibles.")
        summary = orders_df["prioridad"].fillna("Observacion").value_counts().reset_index()
        summary.columns = ["categoria", "count"]
        title = "Distribución por prioridad"

    fig = px.bar(
        summary,
        x="categoria",
        y="count",
        color="categoria",
        color_discrete_sequence=[
            COLOR_PALETTE["critical"],
            COLOR_PALETTE["high"],
            COLOR_PALETTE["warning"],
            COLOR_PALETTE["success"],
            COLOR_PALETTE["soft_gray"],
        ],
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Cantidad")
    fig.update_layout(showlegend=False, height=360)
    return _apply_layout(fig, title)


def create_recommendations_treemap(recommendations_df: pd.DataFrame) -> go.Figure | None:
    """Treemap de recomendaciones estratégicas."""
    df = _safe_dataframe(recommendations_df)
    if df.empty:
        return None

    label_col = None
    for candidate in ["zona_o_territorio", "zona", "territorio", "zona_territorio"]:
        if candidate in df.columns:
            label_col = candidate
            break
    type_col = "tipo_recomendacion" if "tipo_recomendacion" in df.columns else None
    if not label_col or not type_col:
        return None

    temp_df = df.copy()
    temp_df["size_value"] = 1
    for candidate in ["impact_score", "final_impact_score", "prioridad"]:
        if candidate in temp_df.columns:
            parsed = pd.to_numeric(temp_df[candidate], errors="coerce")
            if parsed.notna().any():
                temp_df["size_value"] = parsed.fillna(1).clip(lower=1)
                break
    if temp_df["size_value"].eq(1).all() and "impacto_estimado" in temp_df.columns:
        temp_df["size_value"] = temp_df["impacto_estimado"].map({"Bajo": 1, "Medio": 2, "Alto": 3}).fillna(1)

    if "impacto_estimado" not in temp_df.columns:
        temp_df["impacto_estimado"] = ""
    if "nivel_confianza" not in temp_df.columns:
        temp_df["nivel_confianza"] = ""
    if "justificacion" not in temp_df.columns:
        temp_df["justificacion"] = ""

    fig = px.treemap(
        temp_df,
        path=[px.Constant("Recomendaciones"), type_col, label_col, "impacto_estimado"],
        values="size_value",
        color=type_col,
        hover_data={
            "justificacion": True,
            "nivel_confianza": True,
            "size_value": ":.2f",
        },
        color_discrete_sequence=[
            COLOR_PALETTE["electric_blue"],
            COLOR_PALETTE["tech_purple"],
            COLOR_PALETTE["success"],
            COLOR_PALETTE["high"],
        ],
    )
    fig.update_layout(height=420)
    return _apply_layout(fig, "Árbol de recomendaciones")


def _estimate_map_zoom(lat_range: float, lon_range: float) -> float:
    """Estima un zoom razonable según la dispersión observada."""
    max_range = max(lat_range, lon_range)
    if max_range <= 0.03:
        return 13
    if max_range <= 0.08:
        return 12
    if max_range <= 0.18:
        return 11
    return 10


def create_cali_priority_map(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    impact_scores_df: pd.DataFrame | None = None,
    height: int = 700,
) -> go.Figure | None:
    """Mantiene compatibilidad y delega al mapa profesional de Cali."""
    return create_cali_priority_map_pro(
        dataframe,
        schema_mapping,
        impact_scores_df=impact_scores_df,
        work_orders_df=None,
        recommendations_df=None,
        height=max(int(height), 700),
    )


def create_geo_priority_map(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    impact_scores_df: pd.DataFrame | None = None,
) -> go.Figure | None:
    """Compatibilidad hacia atrás con el mapa ejecutivo."""
    return create_cali_priority_map_pro(
        dataframe,
        schema_mapping,
        impact_scores_df=impact_scores_df,
        work_orders_df=None,
        recommendations_df=None,
        height=700,
    )


def create_calendar_heatmap(
    dataframe: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    impact_scores_df: pd.DataFrame | None = None,
) -> go.Figure | None:
    """Heatmap temporal por día de semana y mes."""
    date_col = schema_mapping.get("date_col")
    if not date_col or date_col not in dataframe.columns:
        return None

    parsed_dates = pd.to_datetime(dataframe[date_col], errors="coerce")
    temp_df = pd.DataFrame({"fecha": parsed_dates}).dropna()
    if temp_df.empty:
        return None

    temp_df["dia_semana"] = pd.Categorical(
        temp_df["fecha"].dt.day_name(),
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    temp_df["mes"] = temp_df["fecha"].dt.strftime("%Y-%m")
    summary = temp_df.groupby(["dia_semana", "mes"], observed=False).size().reset_index(name="registros")
    pivot_df = summary.pivot(index="dia_semana", columns="mes", values="registros").fillna(0)
    if pivot_df.empty:
        return None

    fig = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#0b1220"],
            [0.25, COLOR_PALETTE["electric_blue"]],
            [0.5, COLOR_PALETTE["cyan"]],
            [0.75, COLOR_PALETTE["success"]],
            [1.0, COLOR_PALETTE["warning"]],
        ],
        labels={"x": "Mes", "y": "Día de semana", "color": "Registros"},
    )
    fig.update_layout(height=360)
    return _apply_layout(fig, "Mapa temporal del dataset")


def create_kpi_cards_data(results: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Devuelve KPIs listos para renderizar en la vista ejecutiva."""
    impact_df = _safe_dataframe(results.get("impact_scores"))
    work_orders_df = _safe_dataframe(results.get("work_orders"))
    review_df = _safe_dataframe(results.get("human_review_log"))
    readiness = results.get("readiness", {})
    quality_gate = results.get("quality_gate_report", {})

    critical_count = 0
    if not impact_df.empty and "classification" in impact_df.columns:
        critical_count = int(impact_df["classification"].astype(str).eq("Critico").sum())

    high_priority = 0
    if not work_orders_df.empty and "prioridad" in work_orders_df.columns:
        high_priority = int(work_orders_df["prioridad"].astype(str).isin(["Alta", "Media"]).sum())

    reviewed_orders = 0
    approved_orders = 0
    pending_orders = 0
    if not review_df.empty and "estado_revision" in review_df.columns:
        reviewed_orders = int(review_df["estado_revision"].astype(str).ne("pendiente").sum())
        approved_orders = int(review_df["estado_revision"].astype(str).eq("aprobada").sum())
        pending_orders = int(review_df["estado_revision"].astype(str).eq("pendiente").sum())

    return {
        "total_zones": int(impact_df["zona"].nunique()) if not impact_df.empty and "zona" in impact_df.columns else 0,
        "total_records": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "work_orders_count": int(len(work_orders_df)),
        "critical_zones_count": critical_count,
        "high_priority_zones_count": high_priority,
        "readiness_score": readiness.get("score", 0),
        "confidence_level": results.get("confidence_level", "Baja"),
        "quality_gate": quality_gate.get("quality_gate", "Sin evaluar"),
        "reviewed_orders": reviewed_orders,
        "approved_orders": approved_orders,
        "pending_orders": pending_orders,
    }
