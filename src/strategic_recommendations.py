from __future__ import annotations

import pandas as pd

from src.schema_mapper import SchemaMapping


RECOMMENDATION_COLUMNS = [
    "zona_o_territorio",
    "tipo_recomendacion",
    "justificacion",
    "impacto_estimado",
    "esfuerzo_estimado",
    "nivel_confianza",
]


def _empty_recommendations() -> pd.DataFrame:
    """Devuelve una tabla vacia de recomendaciones."""
    return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)


def _build_recommendation(
    zona_o_territorio: str,
    tipo_recomendacion: str,
    justificacion: str,
    impacto_estimado: str,
    esfuerzo_estimado: str,
    nivel_confianza: str,
) -> dict[str, object]:
    """Construye una recomendacion en formato uniforme."""
    return {
        "zona_o_territorio": zona_o_territorio,
        "tipo_recomendacion": tipo_recomendacion,
        "justificacion": justificacion,
        "impacto_estimado": impacto_estimado,
        "esfuerzo_estimado": esfuerzo_estimado,
        "nivel_confianza": nivel_confianza,
    }


def generate_strategic_recommendations(
    dataframe: pd.DataFrame,
    schema_mapping: SchemaMapping,
    work_orders: pd.DataFrame | None = None,
    osm_context: pd.DataFrame | None = None,
    weather_context: pd.DataFrame | None = None,
    impact_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Genera recomendaciones de mantenimiento e inversion con evidencia disponible."""
    if dataframe.empty:
        return _empty_recommendations()

    zone_col = schema_mapping.get("zone_col")
    territory_col = schema_mapping.get("territory_col")
    latitude_col = schema_mapping.get("latitude_col")
    longitude_col = schema_mapping.get("longitude_col")
    connections_col = schema_mapping.get("connections_col")
    traffic_col = schema_mapping.get("traffic_col")

    if not zone_col:
        return pd.DataFrame(
            [
                _build_recommendation(
                    zona_o_territorio="Global",
                    tipo_recomendacion="Completar mapeo funcional",
                    justificacion=(
                        "El dataset no tiene una columna de zona mapeada, por lo que la priorizacion "
                        "territorial y operativa todavia es limitada."
                    ),
                    impacto_estimado="Alto",
                    esfuerzo_estimado="Bajo",
                    nivel_confianza="Alto",
                )
            ]
        )

    recommendations: list[dict[str, object]] = []
    work_orders = work_orders if work_orders is not None else pd.DataFrame()
    impact_scores_df = impact_scores_df if impact_scores_df is not None else pd.DataFrame()

    zone_reference = pd.DataFrame({"zona": dataframe[zone_col].astype(str)})
    if territory_col:
        zone_reference["territorio"] = dataframe[territory_col].astype(str)

    if connections_col:
        zone_reference["connections_value"] = pd.to_numeric(dataframe[connections_col], errors="coerce")
    if traffic_col:
        zone_reference["traffic_value"] = pd.to_numeric(dataframe[traffic_col], errors="coerce")

    work_order_counts = (
        work_orders.groupby("zona").size().to_dict() if not work_orders.empty else {}
    )
    high_priority_counts = (
        work_orders[work_orders["prioridad"].isin(["Alta", "Media"])].groupby("zona").size().to_dict()
        if not work_orders.empty
        else {}
    )

    social_scores = (
        osm_context.groupby("zona", dropna=False)["social_criticality_score"].mean().to_dict()
        if osm_context is not None and not osm_context.empty and "social_criticality_score" in osm_context.columns
        else {}
    )
    weather_notes = (
        weather_context.groupby("zona", dropna=False)["weather_note"].first().to_dict()
        if weather_context is not None and not weather_context.empty and "weather_note" in weather_context.columns
        else {}
    )

    if impact_scores_df is not None and not impact_scores_df.empty:
        for _, row in impact_scores_df.iterrows():
            zona = str(row["zona"])
            social_score = float(row.get("social_criticality_score") or 0)
            final_score = float(row.get("final_impact_score") or 0)
            technical_score = float(row.get("technical_severity_score") or 0)
            demand_score = float(row.get("demand_score") or 0)

            contextual_suffix = ""
            if zona in weather_notes:
                contextual_suffix = (
                    " El clima se considera una variable contextual secundaria y no una prueba causal "
                    f"({weather_notes[zona]})."
                )

            if social_score >= 60 and technical_score >= 60:
                recommendations.append(
                    _build_recommendation(
                        zona_o_territorio=zona,
                        tipo_recomendacion="Mantenimiento urgente por criticidad social",
                        justificacion=(
                            f"La zona combina severidad tecnica ({technical_score:.1f}) y criticidad territorial "
                            f"aproximada ({social_score:.1f}).{contextual_suffix}"
                        ),
                        impacto_estimado="Alto",
                        esfuerzo_estimado="Medio",
                        nivel_confianza="Medio",
                    )
                )

            if social_score >= 60 and high_priority_counts.get(zona, 0) >= 2:
                recommendations.append(
                    _build_recommendation(
                        zona_o_territorio=zona,
                        tipo_recomendacion="Inversion y mantenimiento prioritario",
                        justificacion=(
                            "Hay alta criticidad territorial aproximada y varias ordenes operativas preliminares "
                            "que sugieren proteger la continuidad del servicio."
                            + contextual_suffix
                        ),
                        impacto_estimado="Alto",
                        esfuerzo_estimado="Alto",
                        nivel_confianza="Medio",
                    )
                )

            if social_score < 40 and final_score >= 60 and demand_score >= 60:
                recommendations.append(
                    _build_recommendation(
                        zona_o_territorio=zona,
                        tipo_recomendacion="Reforzar capacidad y monitoreo",
                        justificacion=(
                            f"La zona muestra alta demanda ({demand_score:.1f}) y un score final elevado ({final_score:.1f}). "
                            "Conviene evaluar capacidad, ancho de banda y mantenimiento preventivo."
                            + contextual_suffix
                        ),
                        impacto_estimado="Alto",
                        esfuerzo_estimado="Medio",
                        nivel_confianza="Medio",
                    )
                )

            if social_score >= 60 and demand_score < 40 and work_order_counts.get(zona, 0) == 0:
                recommendations.append(
                    _build_recommendation(
                        zona_o_territorio=zona,
                        tipo_recomendacion="Revisar visibilidad, ubicacion o experiencia de uso",
                        justificacion=(
                            "Hay equipamientos urbanos relevantes cerca, pero la actividad observada es baja. "
                            "Puede ser util revisar senalizacion, ubicacion o facilidad de acceso al servicio."
                            + contextual_suffix
                        ),
                        impacto_estimado="Medio",
                        esfuerzo_estimado="Medio",
                        nivel_confianza="Bajo",
                    )
                )

    if traffic_col and "traffic_value" in zone_reference.columns:
        traffic_df = zone_reference.dropna(subset=["traffic_value"])
        if not traffic_df.empty:
            zone_traffic = traffic_df.groupby("zona")["traffic_value"].mean()
            overall_traffic_mean = float(traffic_df["traffic_value"].mean())
            for zona, zone_mean in zone_traffic.items():
                if overall_traffic_mean > 0 and zone_mean >= overall_traffic_mean * 1.50:
                    recommendations.append(
                        _build_recommendation(
                            zona_o_territorio=str(zona),
                            tipo_recomendacion="Evaluar ampliacion de capacidad",
                            justificacion=(
                                f"La zona registra trafico promedio de {zone_mean:.2f}, por encima del 150% del "
                                f"promedio general de {overall_traffic_mean:.2f}."
                            ),
                            impacto_estimado="Alto",
                            esfuerzo_estimado="Medio",
                            nivel_confianza="Medio",
                        )
                    )

    if osm_context is not None and not osm_context.empty and territory_col and not work_orders.empty:
        territory_lookup = (
            dataframe[[zone_col, territory_col]]
            .dropna(subset=[zone_col, territory_col])
            .astype(str)
            .drop_duplicates()
        )
        orders_with_territory = work_orders.merge(
            territory_lookup,
            left_on="zona",
            right_on=zone_col,
            how="left",
        )
        territory_counts = orders_with_territory[territory_col].value_counts()
        for territory, alert_count in territory_counts.items():
            if int(alert_count) >= 2:
                recommendations.append(
                    _build_recommendation(
                        zona_o_territorio=str(territory),
                        tipo_recomendacion="Priorizar intervencion territorial coordinada",
                        justificacion=(
                            f"El territorio concentra {int(alert_count)} alertas preliminares. "
                            "Se recomienda intervenirlo como frente coordinado de mantenimiento."
                        ),
                        impacto_estimado="Alto",
                        esfuerzo_estimado="Medio",
                        nivel_confianza="Medio",
                    )
                )

    if latitude_col and longitude_col and (osm_context is None or osm_context.empty):
        recommendations.append(
            _build_recommendation(
                zona_o_territorio="Global",
                tipo_recomendacion="Activar enriquecimiento territorial con OSM",
                justificacion=(
                    "Hay coordenadas disponibles, pero todavia no se consulto contexto urbano. "
                    "Enriquecer con OpenStreetMap mejoraria la priorizacion social."
                ),
                impacto_estimado="Medio",
                esfuerzo_estimado="Bajo",
                nivel_confianza="Alto",
            )
        )
    elif not latitude_col or not longitude_col:
        if territory_col:
            recommendations.append(
                _build_recommendation(
                    zona_o_territorio="Global",
                    tipo_recomendacion="Fortalecer georreferenciacion detallada",
                    justificacion=(
                        "El dataset tiene informacion territorial, pero no latitud y longitud completas. "
                        "Agregar coordenadas permitiria priorizacion espacial mas precisa."
                    ),
                    impacto_estimado="Medio",
                    esfuerzo_estimado="Bajo",
                    nivel_confianza="Alto",
                )
            )
        else:
            recommendations.append(
                _build_recommendation(
                    zona_o_territorio="Global",
                    tipo_recomendacion="Solicitar georreferenciacion",
                    justificacion=(
                        "El dataset no contiene informacion geoespacial suficiente. "
                        "Se recomienda solicitar latitud/longitud o comuna/barrio para priorizacion territorial."
                    ),
                    impacto_estimado="Alto",
                    esfuerzo_estimado="Bajo",
                    nivel_confianza="Alto",
                )
            )

    if not recommendations:
        recommendations.append(
            _build_recommendation(
                zona_o_territorio="Global",
                tipo_recomendacion="Continuar monitoreo y enriquecer evidencia",
                justificacion=(
                    "Con el mapeo actual no se observan senales suficientes para priorizaciones fuertes. "
                    "Conviene seguir monitoreando y enriquecer datos tecnicos y territoriales."
                ),
                impacto_estimado="Medio",
                esfuerzo_estimado="Bajo",
                nivel_confianza="Bajo",
            )
        )

    recommendations_df = pd.DataFrame(recommendations).drop_duplicates().reset_index(drop=True)
    return recommendations_df[RECOMMENDATION_COLUMNS]
