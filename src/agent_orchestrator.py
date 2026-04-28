from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd

from src.calendar_context import enrich_calendar_features
from src.decision_passport import generate_passports_for_top_zones
from src.impact_scoring import calculate_impact_scores
from src.meraki_anomaly_engine import (
    build_meraki_decision_passports,
    detect_hourly_anomalies,
    generate_meraki_work_orders,
)
from src.meraki_features import build_operational_mart
from src.meraki_schema import build_meraki_schema_mapping
from src.osm_context import enrich_osm_context
from src.operational_audit import (
    append_audit_event,
    build_operational_audit_summary,
    create_audit_event,
)
from src.readiness_score import calculate_data_readiness
from src.resource_optimizer import optimize_crews
from src.strategic_recommendations import generate_strategic_recommendations
from src.utils import get_timestamp
from src.validation_suite import build_quality_gate_report
from src.weather_context import enrich_weather_context
from src.work_orders import generate_work_orders
from src.wifi_package_loader import get_package_summary


def _empty_results() -> dict[str, object]:
    """Construye contenedores vacios para el ciclo autonomo."""
    return {
        "readiness": {},
        "calendar_context": pd.DataFrame(),
        "work_orders": pd.DataFrame(),
        "recommendations": pd.DataFrame(),
        "weather_context": pd.DataFrame(),
        "osm_context": pd.DataFrame(),
        "impact_scores": pd.DataFrame(),
        "trace_id": None,
        "audit_log": [],
        "audit_summary": {},
        "quality_gate_report": {},
        "crew_plan": {
            "recommended_zones": pd.DataFrame(),
            "waiting_zones": pd.DataFrame(),
            "coverage_territorial": "Sin datos",
            "riesgo_no_atencion": "Sin datos",
            "explanation": "Sin ejecutar.",
        },
        "decision_passports": [],
        "executive_summary": "",
        "agent_event_log": [],
        "limitations": [],
        "confidence_level": "Baja",
    }


def _log_event(
    event_log: list[dict[str, object]],
    agent: str,
    action: str,
    status: str,
    message: str,
) -> None:
    """Registra una accion del ciclo autonomo."""
    event_log.append(
        {
            "agent": agent,
            "action": action,
            "status": status,
            "message": message,
            "timestamp": get_timestamp(),
        }
    )


def _results_confidence_level(results: dict[str, object]) -> str:
    """Deriva un nivel global de confianza del ciclo."""
    readiness = results.get("readiness", {})
    readiness_score = float(readiness.get("score", 0) or 0)
    impact_scores = results.get("impact_scores")
    work_orders = results.get("work_orders")
    recommendations = results.get("recommendations")

    if (
        readiness_score >= 75
        and isinstance(impact_scores, pd.DataFrame)
        and not impact_scores.empty
        and isinstance(work_orders, pd.DataFrame)
        and not work_orders.empty
    ):
        return "Alta"

    if readiness_score >= 50 and (
        (isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty)
        or (isinstance(recommendations, pd.DataFrame) and not recommendations.empty)
    ):
        return "Media"

    return "Baja"


def build_executive_summary(results: dict[str, object]) -> str:
    """Genera un resumen ejecutivo en markdown sin depender de Gemini."""
    readiness = results.get("readiness", {})
    work_orders = results.get("work_orders")
    impact_scores = results.get("impact_scores")
    crew_plan = results.get("crew_plan", {})
    recommendations = results.get("recommendations")
    limitations = results.get("limitations", [])
    confidence_level = results.get("confidence_level", "Baja")

    total_orders = len(work_orders) if isinstance(work_orders, pd.DataFrame) else 0
    top_zones = []
    if isinstance(impact_scores, pd.DataFrame) and not impact_scores.empty:
        for _, row in impact_scores.head(5).iterrows():
            top_zones.append(
                f"- {row['zona']}: score {float(row['final_impact_score']):.2f} ({row['classification']})"
            )

    top_actions = []
    recommended_df = crew_plan.get("recommended_zones")
    if isinstance(recommended_df, pd.DataFrame) and not recommended_df.empty:
        for _, row in recommended_df.head(3).iterrows():
            top_actions.append(
                f"- Priorizar {row['zona']} con score {float(row['final_impact_score']):.2f}."
            )

    strategic_actions = []
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        for _, row in recommendations.head(3).iterrows():
            strategic_actions.append(
                f"- {row['zona_o_territorio']}: {row['tipo_recomendacion']}."
            )

    top_limitations = [f"- {item}" for item in limitations[:5]] or ["- No se registraron limitaciones adicionales."]

    return "\n".join(
        [
            "## Resumen",
            f"- Data Readiness Score: {readiness.get('score', 0)} / 100 ({readiness.get('classification', 'Sin clasificar')})",
            f"- Ordenes de trabajo generadas: {total_orders}",
            f"- Nivel de confianza del ciclo: {confidence_level}",
            "",
            "## Hallazgos principales",
            *(top_zones or ["- No hay zonas priorizadas con la evidencia actual."]),
            "",
            "## Zonas prioritarias",
            *(top_zones[:3] or ["- No hay zonas prioritarias definidas."]),
            "",
            "## Acciones recomendadas",
            *(top_actions or ["- No hay plan de cuadrillas calculado."]),
            *(strategic_actions or []),
            "",
            "## Riesgos y limitaciones",
            *top_limitations,
            "",
            "## Nivel de confianza",
            f"- {confidence_level}",
            "",
            "## Proximo paso recomendado",
            "- Validar las zonas priorizadas en campo y completar los datos faltantes del dataset si la confianza aun es media o baja.",
        ]
    )


def _run_meraki_cycle(
    package: dict[str, object],
    available_crews: int,
    trace_id: str,
    results: dict[str, object],
    event_log: list[dict[str, object]],
    audit_log: list[dict[str, object]],
) -> tuple[dict[str, object], list[str], list[dict[str, object]], list[dict[str, object]]]:
    """Ejecuta el flujo especializado para el paquete oficial Meraki."""
    limitations: list[str] = []
    meraki_mapping = build_meraki_schema_mapping()
    package_summary = get_package_summary(package)
    results["wifi_package_summary"] = package_summary
    results["is_meraki_mode"] = True

    operational_mart = build_operational_mart(package)
    operational_mart.attrs["source"] = "meraki_package"
    hourly_metrics = package.get("hourly_metrics", pd.DataFrame())
    if isinstance(hourly_metrics, pd.DataFrame):
        operational_mart.attrs["meraki_hourly_metrics"] = hourly_metrics

    if operational_mart.empty:
        limitations.append("No fue posible construir el mart operativo Meraki.")
        return results, limitations, event_log, audit_log

    results["operational_mart"] = operational_mart
    results["readiness"] = calculate_data_readiness(operational_mart, meraki_mapping)
    _log_event(
        event_log,
        "Agente Meraki",
        "Construir mart operativo",
        "ok",
        f"Se construyó el mart operativo con {len(operational_mart)} APs.",
    )
    audit_log = append_audit_event(
        audit_log,
        create_audit_event(
            module="Agente Meraki",
            action="Construir mart operativo",
            status="ok",
            message=f"Mart operativo construido con {len(operational_mart)} APs.",
            metadata={"trace_id": trace_id, "aps": int(len(operational_mart))},
        ),
    )

    results["meraki_anomalies"] = detect_hourly_anomalies(hourly_metrics) if isinstance(hourly_metrics, pd.DataFrame) else pd.DataFrame()
    _log_event(
        event_log,
        "Agente Operativo",
        "Detectar anomalías Meraki",
        "ok",
        f"Se detectaron {len(results['meraki_anomalies'])} anomalías horarias.",
    )
    audit_log = append_audit_event(
        audit_log,
        create_audit_event(
            module="Agente Operativo",
            action="Detectar anomalías Meraki",
            status="ok",
            message=f"Se detectaron {len(results['meraki_anomalies'])} anomalías horarias.",
            metadata={"trace_id": trace_id, "anomalies": int(len(results["meraki_anomalies"]))},
        ),
    )

    results["work_orders"] = generate_meraki_work_orders(operational_mart, results["meraki_anomalies"])
    results["impact_scores"] = calculate_impact_scores(operational_mart, meraki_mapping, work_orders=results["work_orders"])
    results["crew_plan"] = optimize_crews(results["impact_scores"], available_crews=available_crews)
    results["recommendations"] = generate_strategic_recommendations(
        operational_mart,
        meraki_mapping,
        work_orders=results["work_orders"],
        impact_scores_df=results["impact_scores"],
    )
    results["decision_passports"] = build_meraki_decision_passports(
        operational_mart,
        results["work_orders"],
    )

    results["work_orders"] = generate_work_orders(
        operational_mart,
        meraki_mapping,
        impact_scores_df=results["impact_scores"],
        decision_passports=results["decision_passports"],
    )

    if not package_summary.get("warnings"):
        package_warnings = []
    else:
        package_warnings = package_summary["warnings"]
    limitations.extend(package_warnings)
    if "Sin coordenadas exactas del AP dentro del paquete oficial." not in limitations:
        limitations.append("Sin coordenadas exactas del AP dentro del paquete oficial.")
    limitations.append(
        "connectivity_history está disponible como texto exportado; los códigos requieren validación técnica con Meraki."
    )

    _log_event(
        event_log,
        "Agente Estratégico",
        "Consolidar recomendaciones Meraki",
        "ok",
        f"Se consolidaron {len(results['recommendations'])} recomendaciones y {len(results['decision_passports'])} pasaportes.",
    )
    audit_log = append_audit_event(
        audit_log,
        create_audit_event(
            module="Agente Estratégico",
            action="Consolidar modo Meraki",
            status="ok",
            message="Se generaron scores, cuadrillas, recomendaciones y pasaportes Meraki.",
            metadata={
                "trace_id": trace_id,
                "work_orders": int(len(results["work_orders"])),
                "impact_scores": int(len(results["impact_scores"])),
                "passports": int(len(results["decision_passports"])),
            },
        ),
    )
    return results, limitations, event_log, audit_log


def run_autonomous_cycle(
    df: pd.DataFrame,
    schema_mapping: dict[str, str | None],
    available_crews: int = 3,
    use_weather_context: bool = False,
    use_osm_context: bool = False,
    max_external_points: int = 15,
    wifi_package: dict[str, object] | None = None,
) -> dict[str, object]:
    """Ejecuta un ciclo autonomo deterministico y tolerante a fallos."""
    results = _empty_results()
    limitations: list[str] = []
    event_log: list[dict[str, object]] = []
    audit_log: list[dict[str, object]] = []
    trace_id = f"TRACE-{get_timestamp()}-{uuid4().hex[:6].upper()}"

    results["trace_id"] = trace_id
    audit_log = append_audit_event(
        audit_log,
        create_audit_event(
            module="Mission Control",
            action="Iniciar ciclo autonomo",
            status="ok",
            message="Se inicio un nuevo ciclo autonomo.",
            metadata={
                "trace_id": trace_id,
                "available_crews": int(available_crews),
                "use_weather_context": bool(use_weather_context),
                "use_osm_context": bool(use_osm_context),
                "max_external_points": int(max_external_points),
            },
        ),
    )

    if isinstance(wifi_package, dict) and bool(wifi_package.get("is_official_package")):
        results, meraki_limitations, event_log, audit_log = _run_meraki_cycle(
            wifi_package,
            available_crews=available_crews,
            trace_id=trace_id,
            results=results,
            event_log=event_log,
            audit_log=audit_log,
        )
        limitations.extend(meraki_limitations)
        results["limitations"] = list(dict.fromkeys(limitations))
        results["confidence_level"] = _results_confidence_level(results)
        try:
            meraki_mapping = build_meraki_schema_mapping()
            quality_gate_report = build_quality_gate_report(
                results.get("operational_mart", df if isinstance(df, pd.DataFrame) else pd.DataFrame()),
                meraki_mapping,
                results=results,
            )
            results["quality_gate_report"] = quality_gate_report
        except Exception as error:  # noqa: BLE001
            limitations.append(f"No fue posible construir quality gate report: {error}")
            results["quality_gate_report"] = {}

        try:
            results["executive_summary"] = build_executive_summary(results)
        except Exception as error:  # noqa: BLE001
            limitations.append(f"No fue posible construir resumen ejecutivo: {error}")
        results["agent_event_log"] = event_log
        results["limitations"] = list(dict.fromkeys(limitations))
        results["audit_log"] = append_audit_event(
            audit_log,
            create_audit_event(
                module="Mission Control",
                action="Finalizar ciclo autonomo",
                status="ok",
                message=f"Ciclo Meraki finalizado con confianza {results['confidence_level']}.",
                metadata={"trace_id": trace_id, "mode": "meraki"},
            ),
        )
        results["audit_summary"] = build_operational_audit_summary(results["audit_log"])
        return results

    try:
        results["readiness"] = calculate_data_readiness(df, schema_mapping)
        readiness_score = results["readiness"].get("score", 0)
        _log_event(
            event_log,
            "Agente de Preparacion de Datos",
            "Calcular Data Readiness Score",
            "ok",
            f"Dataset evaluado con score {readiness_score}.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Preparacion de Datos",
                action="Calcular readiness",
                status="ok",
                message=f"Readiness calculado con score {readiness_score}.",
                metadata={"trace_id": trace_id, "score": readiness_score},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible calcular readiness: {error}")
        _log_event(
            event_log,
            "Agente de Preparacion de Datos",
            "Calcular Data Readiness Score",
            "error",
            f"Fallo en readiness: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Preparacion de Datos",
                action="Calcular readiness",
                status="error",
                message=f"Fallo en readiness: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        calendar_context = enrich_calendar_features(df, schema_mapping)
        results["calendar_context"] = calendar_context
        if calendar_context.empty:
            warning = calendar_context.attrs.get("warning", "Sin contexto calendario util.")
            limitations.append(warning)
            _log_event(
                event_log,
                "Agente Temporal",
                "Enriquecer calendario",
                "warning",
                warning,
            )
            audit_log = append_audit_event(
                audit_log,
                create_audit_event(
                    module="Agente Temporal",
                    action="Enriquecer calendario",
                    status="warning",
                    message=warning,
                    metadata={"trace_id": trace_id},
                ),
            )
        else:
            _log_event(
                event_log,
                "Agente Temporal",
                "Enriquecer calendario",
                "ok",
                f"Se generaron {len(calendar_context)} filas de contexto calendario.",
            )
            audit_log = append_audit_event(
                audit_log,
                create_audit_event(
                    module="Agente Temporal",
                    action="Enriquecer calendario",
                    status="ok",
                    message=f"Se generaron {len(calendar_context)} filas de contexto calendario.",
                    metadata={"trace_id": trace_id, "rows": int(len(calendar_context))},
                ),
            )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible enriquecer calendario: {error}")
        _log_event(
            event_log,
            "Agente Temporal",
            "Enriquecer calendario",
            "error",
            f"Fallo en calendario: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Temporal",
                action="Enriquecer calendario",
                status="error",
                message=f"Fallo en calendario: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["work_orders"] = generate_work_orders(df, schema_mapping)
        _log_event(
            event_log,
            "Agente Operativo",
            "Generar ordenes preliminares",
            "ok",
            f"Se generaron {len(results['work_orders'])} ordenes preliminares.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Operativo",
                action="Generar ordenes de trabajo",
                status="ok",
                message=f"Se generaron {len(results['work_orders'])} ordenes preliminares.",
                metadata={"trace_id": trace_id, "orders": int(len(results["work_orders"]))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible generar ordenes de trabajo: {error}")
        _log_event(
            event_log,
            "Agente Operativo",
            "Generar ordenes preliminares",
            "error",
            f"Fallo en ordenes: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Operativo",
                action="Generar ordenes de trabajo",
                status="error",
                message=f"Fallo en ordenes: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["recommendations"] = generate_strategic_recommendations(
            df,
            schema_mapping,
            work_orders=results["work_orders"],
        )
        _log_event(
            event_log,
            "Agente Estrategico",
            "Generar recomendaciones base",
            "ok",
            f"Se generaron {len(results['recommendations'])} recomendaciones base.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Estrategico",
                action="Generar recomendaciones base",
                status="ok",
                message=f"Se generaron {len(results['recommendations'])} recomendaciones base.",
                metadata={"trace_id": trace_id, "recommendations": int(len(results["recommendations"]))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible generar recomendaciones base: {error}")
        _log_event(
            event_log,
            "Agente Estrategico",
            "Generar recomendaciones base",
            "error",
            f"Fallo en recomendaciones base: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Estrategico",
                action="Generar recomendaciones base",
                status="error",
                message=f"Fallo en recomendaciones base: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    if use_weather_context:
        try:
            results["weather_context"] = enrich_weather_context(
                df,
                schema_mapping,
                max_points=max_external_points,
            )
            weather_context = results["weather_context"]
            if weather_context.empty:
                warning = weather_context.attrs.get("warning", "Sin contexto meteorologico util.")
                limitations.append(warning)
                _log_event(
                    event_log,
                    "Agente Contextual Climatico",
                    "Consultar Open-Meteo",
                    "warning",
                    warning,
                )
                audit_log = append_audit_event(
                    audit_log,
                    create_audit_event(
                        module="Agente Contextual Climatico",
                        action="Consultar Open-Meteo",
                        status="warning",
                        message=warning,
                        metadata={"trace_id": trace_id},
                    ),
                )
            else:
                _log_event(
                    event_log,
                    "Agente Contextual Climatico",
                    "Consultar Open-Meteo",
                    "ok",
                    f"Se enriquecieron {len(weather_context)} puntos climaticos.",
                )
                audit_log = append_audit_event(
                    audit_log,
                    create_audit_event(
                        module="Agente Contextual Climatico",
                        action="Consultar Open-Meteo",
                        status="ok",
                        message=f"Se enriquecieron {len(weather_context)} puntos climaticos.",
                        metadata={"trace_id": trace_id, "rows": int(len(weather_context))},
                    ),
                )
        except Exception as error:  # noqa: BLE001
            limitations.append(f"No fue posible obtener contexto climatico: {error}")
            _log_event(
                event_log,
                "Agente Contextual Climatico",
                "Consultar Open-Meteo",
                "error",
                f"Fallo en clima: {error}",
            )
            audit_log = append_audit_event(
                audit_log,
                create_audit_event(
                    module="Agente Contextual Climatico",
                    action="Consultar Open-Meteo",
                    status="error",
                    message=f"Fallo en clima: {error}",
                    metadata={"trace_id": trace_id},
                ),
            )
    else:
        limitations.append("El contexto climatico no fue activado en este ciclo.")
        _log_event(
            event_log,
            "Agente Contextual Climatico",
            "Consultar Open-Meteo",
            "warning",
            "El usuario no activo clima contextual.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Contextual Climatico",
                action="Consultar Open-Meteo",
                status="warning",
                message="El usuario no activo clima contextual.",
                metadata={"trace_id": trace_id},
            ),
        )

    if use_osm_context:
        try:
            results["osm_context"] = enrich_osm_context(
                df,
                schema_mapping,
                max_points=max_external_points,
                radius_m=600,
            )
            osm_context = results["osm_context"]
            if osm_context.empty:
                warning = osm_context.attrs.get("warning", "Sin contexto OSM util.")
                limitations.append(warning)
                _log_event(
                    event_log,
                    "Agente Contextual Territorial",
                    "Consultar OpenStreetMap Overpass",
                    "warning",
                    warning,
                )
                audit_log = append_audit_event(
                    audit_log,
                    create_audit_event(
                        module="Agente Contextual Territorial",
                        action="Consultar OpenStreetMap Overpass",
                        status="warning",
                        message=warning,
                        metadata={"trace_id": trace_id},
                    ),
                )
            else:
                _log_event(
                    event_log,
                    "Agente Contextual Territorial",
                    "Consultar OpenStreetMap Overpass",
                    "ok",
                    f"Se enriquecieron {len(osm_context)} puntos territoriales.",
                )
                audit_log = append_audit_event(
                    audit_log,
                    create_audit_event(
                        module="Agente Contextual Territorial",
                        action="Consultar OpenStreetMap Overpass",
                        status="ok",
                        message=f"Se enriquecieron {len(osm_context)} puntos territoriales.",
                        metadata={"trace_id": trace_id, "rows": int(len(osm_context))},
                    ),
                )
        except Exception as error:  # noqa: BLE001
            limitations.append(f"No fue posible obtener contexto OSM: {error}")
            _log_event(
                event_log,
                "Agente Contextual Territorial",
                "Consultar OpenStreetMap Overpass",
                "error",
                f"Fallo en OSM: {error}",
            )
            audit_log = append_audit_event(
                audit_log,
                create_audit_event(
                    module="Agente Contextual Territorial",
                    action="Consultar OpenStreetMap Overpass",
                    status="error",
                    message=f"Fallo en OSM: {error}",
                    metadata={"trace_id": trace_id},
                ),
            )
    else:
        limitations.append("El contexto urbano OSM no fue activado en este ciclo.")
        _log_event(
            event_log,
            "Agente Contextual Territorial",
            "Consultar OpenStreetMap Overpass",
            "warning",
            "El usuario no activo contexto OSM.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Contextual Territorial",
                action="Consultar OpenStreetMap Overpass",
                status="warning",
                message="El usuario no activo contexto OSM.",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["impact_scores"] = calculate_impact_scores(
            df,
            schema_mapping,
            work_orders=results["work_orders"],
            osm_context=results["osm_context"],
            weather_context=results["weather_context"],
        )
        _log_event(
            event_log,
            "Agente de Impacto",
            "Calcular indice de impacto ciudadano",
            "ok",
            f"Se calcularon {len(results['impact_scores'])} scores de impacto.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Impacto",
                action="Calcular indice de impacto",
                status="ok",
                message=f"Se calcularon {len(results['impact_scores'])} scores de impacto.",
                metadata={"trace_id": trace_id, "scores": int(len(results["impact_scores"]))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible calcular impact scores: {error}")
        _log_event(
            event_log,
            "Agente de Impacto",
            "Calcular indice de impacto ciudadano",
            "error",
            f"Fallo en impact scoring: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Impacto",
                action="Calcular indice de impacto",
                status="error",
                message=f"Fallo en impact scoring: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["recommendations"] = generate_strategic_recommendations(
            df,
            schema_mapping,
            work_orders=results["work_orders"],
            osm_context=results["osm_context"],
            weather_context=results["weather_context"],
            impact_scores_df=results["impact_scores"],
        )
        _log_event(
            event_log,
            "Agente Estrategico",
            "Actualizar recomendaciones con contexto",
            "ok",
            f"Se consolidaron {len(results['recommendations'])} recomendaciones finales.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Estrategico",
                action="Consolidar recomendaciones",
                status="ok",
                message=f"Se consolidaron {len(results['recommendations'])} recomendaciones finales.",
                metadata={"trace_id": trace_id, "recommendations": int(len(results["recommendations"]))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible actualizar recomendaciones finales: {error}")
        _log_event(
            event_log,
            "Agente Estrategico",
            "Actualizar recomendaciones con contexto",
            "error",
            f"Fallo en recomendaciones finales: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Estrategico",
                action="Consolidar recomendaciones",
                status="error",
                message=f"Fallo en recomendaciones finales: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["crew_plan"] = optimize_crews(
            results["impact_scores"],
            available_crews=available_crews,
        )
        recommended_df = results["crew_plan"].get("recommended_zones", pd.DataFrame())
        _log_event(
            event_log,
            "Agente de Recursos",
            "Optimizar cuadrillas",
            "ok",
            f"Se priorizaron {len(recommended_df)} zonas para atencion.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Recursos",
                action="Optimizar cuadrillas",
                status="ok",
                message=f"Se priorizaron {len(recommended_df)} zonas para atencion.",
                metadata={"trace_id": trace_id, "recommended_zones": int(len(recommended_df))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible optimizar cuadrillas: {error}")
        _log_event(
            event_log,
            "Agente de Recursos",
            "Optimizar cuadrillas",
            "error",
            f"Fallo en optimizacion de cuadrillas: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Recursos",
                action="Optimizar cuadrillas",
                status="error",
                message=f"Fallo en optimizacion de cuadrillas: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        results["decision_passports"] = generate_passports_for_top_zones(
            results["impact_scores"],
            work_orders=results["work_orders"],
            recommendations=results["recommendations"],
            top_n=10,
        )
        results["work_orders"] = generate_work_orders(
            df,
            schema_mapping,
            impact_scores_df=results["impact_scores"],
            decision_passports=results["decision_passports"],
        )
        _log_event(
            event_log,
            "Agente de Decision",
            "Generar pasaportes y enriquecer ordenes",
            "ok",
            f"Se generaron {len(results['decision_passports'])} pasaportes de decision.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Decision",
                action="Generar pasaportes de decision",
                status="ok",
                message=f"Se generaron {len(results['decision_passports'])} pasaportes de decision.",
                metadata={"trace_id": trace_id, "passports": int(len(results["decision_passports"]))},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible generar pasaportes de decision: {error}")
        _log_event(
            event_log,
            "Agente de Decision",
            "Generar pasaportes y enriquecer ordenes",
            "error",
            f"Fallo en pasaportes: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente de Decision",
                action="Generar pasaportes de decision",
                status="error",
                message=f"Fallo en pasaportes: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    results["limitations"] = list(dict.fromkeys(limitations))
    results["confidence_level"] = _results_confidence_level(results)

    try:
        results["executive_summary"] = build_executive_summary(results)
        _log_event(
            event_log,
            "Agente Ejecutivo",
            "Construir resumen ejecutivo",
            "ok",
            "Se construyo el resumen ejecutivo base.",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Ejecutivo",
                action="Construir resumen ejecutivo",
                status="ok",
                message="Se construyo el resumen ejecutivo base.",
                metadata={"trace_id": trace_id},
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible construir resumen ejecutivo: {error}")
        results["executive_summary"] = (
            "## Resumen\n- No fue posible construir el resumen ejecutivo base.\n"
            f"- Detalle: {error}"
        )
        _log_event(
            event_log,
            "Agente Ejecutivo",
            "Construir resumen ejecutivo",
            "error",
            f"Fallo en resumen ejecutivo: {error}",
        )
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Agente Ejecutivo",
                action="Construir resumen ejecutivo",
                status="error",
                message=f"Fallo en resumen ejecutivo: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    try:
        quality_gate_report = build_quality_gate_report(df, schema_mapping, results=results)
        results["quality_gate_report"] = quality_gate_report
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Blindaje Tecnico",
                action="Construir quality gate report",
                status="ok",
                message=f"Quality gate: {quality_gate_report.get('quality_gate', 'Sin evaluar')}.",
                metadata={
                    "trace_id": trace_id,
                    "quality_gate": quality_gate_report.get("quality_gate"),
                    "demo_readiness": quality_gate_report.get("demo_readiness"),
                },
            ),
        )
    except Exception as error:  # noqa: BLE001
        limitations.append(f"No fue posible construir quality gate report: {error}")
        results["quality_gate_report"] = {}
        audit_log = append_audit_event(
            audit_log,
            create_audit_event(
                module="Blindaje Tecnico",
                action="Construir quality gate report",
                status="error",
                message=f"Fallo en quality gate: {error}",
                metadata={"trace_id": trace_id},
            ),
        )

    results["agent_event_log"] = event_log
    results["limitations"] = list(dict.fromkeys(limitations))
    results["audit_log"] = audit_log
    results["audit_summary"] = build_operational_audit_summary(audit_log)
    audit_log = append_audit_event(
        audit_log,
        create_audit_event(
            module="Mission Control",
            action="Finalizar ciclo autonomo",
            status="ok",
            message=f"Ciclo finalizado con confianza {results['confidence_level']}.",
            metadata={
                "trace_id": trace_id,
                "confidence_level": results["confidence_level"],
                "limitations": len(results["limitations"]),
            },
        ),
    )
    results["audit_log"] = audit_log
    results["audit_summary"] = build_operational_audit_summary(audit_log)
    return results
