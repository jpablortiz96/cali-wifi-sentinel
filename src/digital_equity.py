from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras comunes a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def _safe_numeric_series(dataframe: pd.DataFrame, column_name: str, default_value: float = 0.0) -> pd.Series:
    """Devuelve una serie numérica segura o default."""
    if column_name in dataframe.columns:
        return pd.to_numeric(dataframe[column_name], errors="coerce").fillna(default_value)
    return pd.Series([default_value] * len(dataframe), index=dataframe.index, dtype="float64")


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Escala una serie al rango 0-100."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.empty:
        return pd.Series(dtype="float64")

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value == min_value:
        base = pd.Series([70.0 if max_value > 0 else 0.0] * len(numeric), index=numeric.index)
    else:
        base = ((numeric - min_value) / (max_value - min_value)) * 100.0
    return base.clip(0, 100).round(2)


def calculate_digital_equity_proxy(
    operational_mart: pd.DataFrame,
    citizen_scores: pd.DataFrame,
    osm_context: pd.DataFrame | None = None,
    feedback_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Construye un proxy responsable de equidad digital por zona."""
    mart = _safe_dataframe(operational_mart)
    citizen_df = _safe_dataframe(citizen_scores)
    if mart.empty and citizen_df.empty:
        return pd.DataFrame(
            columns=[
                "zone_name",
                "digital_equity_proxy",
                "usage_mb_total",
                "clients_reported",
                "citizen_experience_score",
                "operational_risk_score",
                "social_criticality_score",
                "feedback_pressure_score",
                "equity_label",
                "interpretation",
                "limitations",
            ]
        )

    if "zone_name" not in mart.columns and "ap_name" in mart.columns:
        mart["zone_name"] = mart["ap_name"].astype(str)
    if "zone_name" not in citizen_df.columns and "ap_name" in citizen_df.columns:
        citizen_df["zone_name"] = citizen_df["ap_name"].astype(str)

    zone_mart = pd.DataFrame()
    if not mart.empty and "zone_name" in mart.columns:
        zone_mart = (
            mart.groupby("zone_name", dropna=False)
            .agg(
                usage_mb_total=("usage_mb_total", "sum"),
                clients_reported=("clients_reported", "sum"),
                operational_risk_score=("operational_risk_score", "mean"),
                evidence_level=("evidence_level", "mean"),
            )
            .reset_index()
        )

    zone_citizen = pd.DataFrame()
    if not citizen_df.empty and "zone_name" in citizen_df.columns:
        zone_citizen = (
            citizen_df.groupby("zone_name", dropna=False)
            .agg(
                citizen_experience_score=("citizen_experience_score", "mean"),
                citizen_data_confidence=("data_confidence_score", "mean"),
            )
            .reset_index()
        )

    if zone_mart.empty:
        zone_df = zone_citizen.copy()
        zone_df["usage_mb_total"] = 0.0
        zone_df["clients_reported"] = 0.0
        zone_df["operational_risk_score"] = 0.0
        zone_df["evidence_level"] = zone_df.get("citizen_data_confidence", 0.0)
    elif zone_citizen.empty:
        zone_df = zone_mart.copy()
        zone_df["citizen_experience_score"] = 0.0
        zone_df["citizen_data_confidence"] = zone_df.get("evidence_level", 0.0)
    else:
        zone_df = zone_mart.merge(zone_citizen, on="zone_name", how="outer")

    zone_df["usage_mb_total"] = _safe_numeric_series(zone_df, "usage_mb_total", 0.0)
    zone_df["clients_reported"] = _safe_numeric_series(zone_df, "clients_reported", 0.0)
    zone_df["citizen_experience_score"] = _safe_numeric_series(zone_df, "citizen_experience_score", 0.0)
    zone_df["operational_risk_score"] = _safe_numeric_series(zone_df, "operational_risk_score", 0.0)
    zone_df["evidence_level"] = _safe_numeric_series(zone_df, "evidence_level", 0.0)
    zone_df["citizen_data_confidence"] = _safe_numeric_series(zone_df, "citizen_data_confidence", 0.0)

    social_scores = {}
    osm_df = _safe_dataframe(osm_context)
    if not osm_df.empty:
        zone_column = "zone_name" if "zone_name" in osm_df.columns else ("zona" if "zona" in osm_df.columns else None)
        score_column = "social_criticality_score" if "social_criticality_score" in osm_df.columns else None
        if zone_column and score_column:
            social_scores = (
                osm_df.groupby(zone_column, dropna=False)[score_column].mean().round(2).to_dict()
            )

    feedback_pressure_map = {}
    if isinstance(feedback_summary, dict):
        feedback_pressure_map = {
            str(key): float(value)
            for key, value in feedback_summary.get("zone_report_counts", {}).items()
        }

    zone_df["social_criticality_score"] = zone_df["zone_name"].map(social_scores).fillna(0.0)
    feedback_base = zone_df["zone_name"].map(feedback_pressure_map).fillna(0.0)
    zone_df["feedback_pressure_score"] = _min_max_scale(feedback_base)

    usage_pressure = (_min_max_scale(zone_df["usage_mb_total"]) + _min_max_scale(zone_df["clients_reported"])) / 2.0
    experience_need = 100.0 - zone_df["citizen_experience_score"]
    evidence_penalty = 100.0 - ((zone_df["evidence_level"] * 0.5) + (zone_df["citizen_data_confidence"] * 0.5))

    zone_df["digital_equity_proxy"] = (
        experience_need * 0.30
        + zone_df["operational_risk_score"] * 0.25
        + usage_pressure * 0.20
        + zone_df["social_criticality_score"] * 0.15
        + zone_df["feedback_pressure_score"] * 0.10
    ).clip(0, 100).round(2)

    labels: list[str] = []
    interpretations: list[str] = []
    limitations: list[str] = []
    for _, row in zone_df.iterrows():
        if row["evidence_level"] < 35 and row["citizen_data_confidence"] < 35:
            label = "Baja evidencia / requiere validación"
            interpretation = "La zona requiere más datos agregados antes de emitir conclusiones fuertes."
        elif row["citizen_experience_score"] >= 75 and row["operational_risk_score"] < 45:
            label = "Buena experiencia"
            interpretation = "La experiencia agregada luce favorable y el riesgo operativo relativo es bajo."
        elif row["usage_mb_total"] > zone_df["usage_mb_total"].median() and row["operational_risk_score"] >= 60:
            label = "Alta demanda con riesgo operativo"
            interpretation = "La zona combina uso relevante con señales operativas que podrían afectar continuidad o calidad."
        elif row["digital_equity_proxy"] >= 65:
            label = "Alta necesidad de mejora"
            interpretation = "La señal agregada sugiere prioridad de revisión o mejora para sostener acceso público de calidad."
        else:
            label = "Posible baja adopción"
            interpretation = "La actividad es relativamente baja o ambigua. Conviene revisar visibilidad, acceso o más evidencia."

        labels.append(label)
        interpretations.append(interpretation)
        limitations.append(
            "Proxy de equidad digital basado en uso agregado, experiencia estimada, riesgo operativo y evidencia disponible. "
            "No representa población real ni confirma brechas estructurales por sí sola."
        )

    zone_df["equity_label"] = labels
    zone_df["interpretation"] = interpretations
    zone_df["limitations"] = limitations

    return zone_df[
        [
            "zone_name",
            "digital_equity_proxy",
            "usage_mb_total",
            "clients_reported",
            "citizen_experience_score",
            "operational_risk_score",
            "social_criticality_score",
            "feedback_pressure_score",
            "equity_label",
            "interpretation",
            "limitations",
        ]
    ].sort_values(by=["digital_equity_proxy", "operational_risk_score"], ascending=[False, False]).reset_index(drop=True)
