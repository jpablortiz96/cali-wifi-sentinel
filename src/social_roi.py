from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


SOCIAL_ROI_COLUMNS = [
    "social_roi_score",
    "social_roi_label",
    "socioeconomic_vulnerability_score",
    "digital_need_score",
    "network_risk_score",
    "citizen_potential_score",
    "data_confidence_score",
    "explanation",
    "limitations",
]


def _normalize_join_key(value: object) -> str:
    """Normaliza nombres de zonas o territorios para cruces flexibles."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    return normalized


def _safe_dataframe(data: object) -> pd.DataFrame:
    """Convierte estructuras conocidas a DataFrame de forma segura."""
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


def _safe_numeric_series(dataframe: pd.DataFrame, column_name: str, default_value: float = 0.0) -> pd.Series:
    """Devuelve una serie numérica robusta para cálculos transparentes."""
    if column_name in dataframe.columns:
        return pd.to_numeric(dataframe[column_name], errors="coerce").fillna(default_value)
    return pd.Series([default_value] * len(dataframe), index=dataframe.index, dtype="float64")


def _scale_to_100(series: pd.Series, invert: bool = False, fallback: float = 0.0) -> pd.Series:
    """Escala una serie a 0-100 con control de casos constantes."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([fallback] * len(series), index=series.index, dtype="float64")

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value == min_value:
        base = pd.Series([70.0 if max_value > 0 else fallback] * len(series), index=series.index, dtype="float64")
    else:
        base = ((numeric.fillna(min_value) - min_value) / (max_value - min_value)) * 100.0
    if invert:
        base = 100.0 - base
    return base.clip(0, 100)


def prepare_social_roi_inputs(
    operational_mart: pd.DataFrame | None = None,
    citizen_scores: pd.DataFrame | None = None,
    digital_equity_df: pd.DataFrame | None = None,
    socioeconomic_df: pd.DataFrame | None = None,
    osm_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Une las entradas más útiles disponibles para calcular retorno social de conectividad."""
    mart_df = _safe_dataframe(operational_mart)
    citizen_df = _safe_dataframe(citizen_scores)
    equity_df = _safe_dataframe(digital_equity_df)
    socio_df = _safe_dataframe(socioeconomic_df)
    osm_df = _safe_dataframe(osm_context)

    base_df = pd.DataFrame()
    if not mart_df.empty:
        base_df = mart_df.copy()
    elif not citizen_df.empty:
        base_df = citizen_df.copy()
        if "zone_name" not in base_df.columns and "zona" in base_df.columns:
            base_df["zone_name"] = base_df["zona"]
    elif not equity_df.empty:
        base_df = equity_df.copy()
        if "zone_name" not in base_df.columns and "zona" in base_df.columns:
            base_df["zone_name"] = base_df["zona"]
    else:
        return pd.DataFrame()

    if "zone_name" not in base_df.columns:
        if "ap_name" in base_df.columns:
            base_df["zone_name"] = base_df["ap_name"].astype(str)
        else:
            base_df["zone_name"] = base_df.index.map(lambda value: f"zona_{value}")

    base_df["join_key"] = base_df["zone_name"].map(_normalize_join_key)
    base_df["match_level"] = "zona"
    base_df["limitations"] = ""

    if not citizen_df.empty:
        citizen_merge = citizen_df.copy()
        if "zone_name" not in citizen_merge.columns and "zona" in citizen_merge.columns:
            citizen_merge["zone_name"] = citizen_merge["zona"]
        if "zone_name" in citizen_merge.columns:
            citizen_merge["join_key"] = citizen_merge["zone_name"].map(_normalize_join_key)
            merge_columns = [
                "join_key",
                "citizen_experience_score",
                "citizen_status",
                "stability_score",
                "availability_score",
                "perceived_capacity_score",
                "citizen_activity_score",
                "data_confidence_score",
            ]
            available_columns = [column_name for column_name in merge_columns if column_name in citizen_merge.columns]
            base_df = base_df.merge(citizen_merge[available_columns].drop_duplicates("join_key"), on="join_key", how="left")

    if not equity_df.empty:
        equity_merge = equity_df.copy()
        if "zone_name" not in equity_merge.columns and "zona" in equity_merge.columns:
            equity_merge["zone_name"] = equity_merge["zona"]
        if "zone_name" in equity_merge.columns:
            equity_merge["join_key"] = equity_merge["zone_name"].map(_normalize_join_key)
            equity_columns = [
                "join_key",
                "digital_equity_proxy",
                "equity_label",
                "social_criticality_score",
                "feedback_pressure_score",
            ]
            available_columns = [column_name for column_name in equity_columns if column_name in equity_merge.columns]
            base_df = base_df.merge(equity_merge[available_columns].drop_duplicates("join_key"), on="join_key", how="left")

    if not socio_df.empty:
        socio_key_column = None
        for candidate in ["zona", "comuna", "barrio", "corregimiento", "codigo_manzana", "municipio"]:
            if candidate in socio_df.columns and socio_df[candidate].notna().any():
                socio_key_column = candidate
                break
        if socio_key_column:
            socio_merge = socio_df.copy()
            socio_merge["join_key"] = socio_merge[socio_key_column].map(_normalize_join_key)
            socio_merge["match_level"] = "manzana" if socio_key_column == "codigo_manzana" else socio_key_column
            socio_columns = ["join_key", "match_level"] + [
                column_name
                for column_name in [
                    "zona",
                    "comuna",
                    "barrio",
                    "corregimiento",
                    "municipio",
                    "ipm",
                    "nbi",
                    "desempleo",
                    "poblacion",
                    "sisben_grupo_a_pct",
                    "sisben_grupo_b_pct",
                    "alfabetizacion_digital_proxy",
                    "fuente",
                    "anio",
                ]
                if column_name in socio_merge.columns
            ]
            base_df = base_df.merge(socio_merge[socio_columns].drop_duplicates("join_key"), on="join_key", how="left", suffixes=("", "_socio"))
            base_df["match_level"] = base_df["match_level_y"].fillna(base_df["match_level_x"]) if "match_level_y" in base_df.columns else base_df["match_level"]
            base_df = base_df.drop(columns=[column_name for column_name in ["match_level_x", "match_level_y"] if column_name in base_df.columns])
        else:
            base_df["limitations"] = base_df["limitations"].astype(str) + " Sin nivel geográfico claro en el dataset socioeconómico."
    else:
        base_df["limitations"] = base_df["limitations"].astype(str) + " No hay dataset socioeconómico cargado."

    if not osm_df.empty and "zona" in osm_df.columns and "social_criticality_score" in osm_df.columns:
        osm_merge = osm_df.copy()
        osm_merge["join_key"] = osm_merge["zona"].map(_normalize_join_key)
        osm_grouped = osm_merge.groupby("join_key", dropna=False)["social_criticality_score"].mean().reset_index()
        base_df = base_df.merge(osm_grouped, on="join_key", how="left", suffixes=("", "_osm"))
        if "social_criticality_score_osm" in base_df.columns:
            base_df["social_criticality_score"] = base_df["social_criticality_score"].fillna(base_df["social_criticality_score_osm"])
            base_df = base_df.drop(columns=["social_criticality_score_osm"])

    unmatched = base_df["join_key"].isna() | base_df["join_key"].astype(str).eq("")
    if unmatched.any():
        base_df.loc[unmatched, "limitations"] = base_df.loc[unmatched, "limitations"].astype(str) + " No fue posible normalizar el nombre de zona para el cruce."

    return base_df


def classify_social_roi(score: float) -> str:
    """Clasifica el retorno social esperado."""
    if score >= 80:
        return "Muy alto retorno social"
    if score >= 60:
        return "Alto retorno social"
    if score >= 40:
        return "Retorno social medio"
    if score >= 20:
        return "Retorno social bajo"
    return "Requiere mas datos"


def calculate_social_roi_score(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Social ROI Connectivity Score con reglas transparentes y agregadas."""
    if merged_df is None or merged_df.empty:
        return pd.DataFrame(
            columns=[
                "zone_name",
                "social_roi_score",
                "social_roi_label",
                "socioeconomic_vulnerability_score",
                "digital_need_score",
                "network_risk_score",
                "citizen_potential_score",
                "data_confidence_score",
                "explanation",
                "limitations",
            ]
        )

    roi_df = merged_df.copy()
    if "zone_name" not in roi_df.columns:
        roi_df["zone_name"] = roi_df.get("zona", roi_df.index.map(lambda value: f"zona_{value}"))

    socioeconomic_components: list[pd.Series] = []
    for column_name in ["ipm", "nbi", "desempleo", "sisben_grupo_a_pct", "sisben_grupo_b_pct"]:
        if column_name in roi_df.columns:
            socioeconomic_components.append(_scale_to_100(_safe_numeric_series(roi_df, column_name), invert=False))
    if "alfabetizacion_digital_proxy" in roi_df.columns:
        socioeconomic_components.append(_scale_to_100(_safe_numeric_series(roi_df, "alfabetizacion_digital_proxy"), invert=True))

    if socioeconomic_components:
        roi_df["socioeconomic_vulnerability_score"] = pd.concat(socioeconomic_components, axis=1).mean(axis=1).round(2)
    else:
        roi_df["socioeconomic_vulnerability_score"] = 0.0

    digital_need_components: list[pd.Series] = []
    if "digital_equity_proxy" in roi_df.columns:
        digital_need_components.append(_scale_to_100(_safe_numeric_series(roi_df, "digital_equity_proxy"), invert=False))
    if "citizen_experience_score" in roi_df.columns:
        digital_need_components.append(_scale_to_100(_safe_numeric_series(roi_df, "citizen_experience_score"), invert=True))
    if "availability_score" in roi_df.columns:
        digital_need_components.append(_scale_to_100(_safe_numeric_series(roi_df, "availability_score"), invert=True))
    roi_df["digital_need_score"] = (
        pd.concat(digital_need_components, axis=1).mean(axis=1).round(2)
        if digital_need_components
        else pd.Series([0.0] * len(roi_df), index=roi_df.index)
    )

    network_components: list[pd.Series] = []
    for column_name in ["operational_risk_score", "anomaly_risk", "disconnection_risk", "status_risk"]:
        if column_name in roi_df.columns:
            network_components.append(_scale_to_100(_safe_numeric_series(roi_df, column_name), invert=False))
    roi_df["network_risk_score"] = (
        pd.concat(network_components, axis=1).mean(axis=1).round(2)
        if network_components
        else pd.Series([0.0] * len(roi_df), index=roi_df.index)
    )

    potential_components: list[pd.Series] = []
    for column_name in ["clients_reported", "usage_mb_total", "demand_impact", "social_criticality_score"]:
        if column_name in roi_df.columns:
            potential_components.append(_scale_to_100(_safe_numeric_series(roi_df, column_name), invert=False))
    roi_df["citizen_potential_score"] = (
        pd.concat(potential_components, axis=1).mean(axis=1).round(2)
        if potential_components
        else pd.Series([0.0] * len(roi_df), index=roi_df.index)
    )

    confidence_parts = []
    for indicator_name in [
        "ipm",
        "nbi",
        "desempleo",
        "poblacion",
        "sisben_grupo_a_pct",
        "sisben_grupo_b_pct",
        "alfabetizacion_digital_proxy",
        "citizen_experience_score",
        "operational_risk_score",
    ]:
        if indicator_name in roi_df.columns:
            confidence_parts.append(roi_df[indicator_name].notna().astype(float))
    if "evidence_level" in roi_df.columns:
        confidence_parts.append(_safe_numeric_series(roi_df, "evidence_level") / 100.0)
    if confidence_parts:
        roi_df["data_confidence_score"] = (pd.concat(confidence_parts, axis=1).mean(axis=1) * 100.0).round(2)
    else:
        roi_df["data_confidence_score"] = 25.0

    roi_df["social_roi_score"] = (
        roi_df["socioeconomic_vulnerability_score"] * 0.30
        + roi_df["digital_need_score"] * 0.25
        + roi_df["network_risk_score"] * 0.20
        + roi_df["citizen_potential_score"] * 0.15
        + roi_df["data_confidence_score"] * 0.10
    ).round(2)
    roi_df["social_roi_label"] = roi_df["social_roi_score"].map(classify_social_roi)
    roi_df["explanation"] = (
        "Vulnerabilidad socioeconómica: "
        + roi_df["socioeconomic_vulnerability_score"].round(1).astype(str)
        + " | Necesidad digital: "
        + roi_df["digital_need_score"].round(1).astype(str)
        + " | Riesgo de red: "
        + roi_df["network_risk_score"].round(1).astype(str)
    )
    roi_df["limitations"] = roi_df.get("limitations", "").astype(str).str.strip()
    roi_df.loc[roi_df["socioeconomic_vulnerability_score"].eq(0), "limitations"] = (
        roi_df.loc[roi_df["socioeconomic_vulnerability_score"].eq(0), "limitations"].astype(str)
        + " Faltan indicadores socioeconómicos suficientes para una señal fuerte de retorno social."
    ).str.strip()

    return roi_df


def generate_social_infrastructure_recommendations(social_roi_df: pd.DataFrame) -> pd.DataFrame:
    """Genera recomendaciones de infraestructura y acompañamiento social a partir del score."""
    if social_roi_df is None or social_roi_df.empty:
        return pd.DataFrame(
            columns=["zone_name", "tipo_recomendacion", "justificacion", "social_roi_label", "social_roi_score"]
        )

    recommendations: list[dict[str, Any]] = []
    for _, row in social_roi_df.iterrows():
        label = str(row.get("social_roi_label", "Requiere mas datos"))
        zone_name = str(row.get("zone_name", "Zona sin nombre"))
        score = float(row.get("social_roi_score", 0) or 0)
        if label in {"Muy alto retorno social", "Alto retorno social"}:
            recommendations.append(
                {
                    "zone_name": zone_name,
                    "tipo_recomendacion": "Priorizar refuerzo de conectividad",
                    "justificacion": (
                        f"La zona alcanza un Social ROI de {score:.2f}. Conviene priorizar mejoras de conectividad "
                        "porque combina señales de vulnerabilidad agregada, necesidad digital y retorno público esperado."
                    ),
                    "social_roi_label": label,
                    "social_roi_score": score,
                }
            )
            recommendations.append(
                {
                    "zone_name": zone_name,
                    "tipo_recomendacion": "Coordinar con instituciones educativas o comunitarias",
                    "justificacion": (
                        "La mejora de conectividad podría complementarse con apropiación social, orientación de uso "
                        "o actividades de alfabetización digital en el territorio."
                    ),
                    "social_roi_label": label,
                    "social_roi_score": score,
                }
            )
        elif label == "Retorno social medio":
            recommendations.append(
                {
                    "zone_name": zone_name,
                    "tipo_recomendacion": "Validar cobertura real en campo",
                    "justificacion": (
                        "La evidencia sugiere una oportunidad intermedia. Se recomienda validar experiencia, cobertura "
                        "y señalización antes de priorizar inversión adicional."
                    ),
                    "social_roi_label": label,
                    "social_roi_score": score,
                }
            )
        else:
            recommendations.append(
                {
                    "zone_name": zone_name,
                    "tipo_recomendacion": "Complementar datos socioeconómicos oficiales",
                    "justificacion": (
                        "La señal de retorno social es baja o insuficiente. Conviene completar indicadores agregados "
                        "antes de escalar decisiones de inversión."
                    ),
                    "social_roi_label": label,
                    "social_roi_score": score,
                }
            )

    return pd.DataFrame(recommendations)
