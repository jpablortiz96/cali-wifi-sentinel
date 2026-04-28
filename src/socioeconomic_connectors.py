from __future__ import annotations

from typing import Any

import pandas as pd

from src.external_sources import get_with_cache
from src.socioeconomic_sources import normalize_socioeconomic_columns


def get_dane_ipm_metadata() -> dict[str, object]:
    """Devuelve metadatos útiles para trabajar con IPM de DANE sin forzar descarga automática."""
    return {
        "name": "DANE - Pobreza Multidimensional (IPM)",
        "provider": "DANE",
        "official_url": "https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-multidimensional",
        "microdata_url": "https://microdatos.dane.gov.co/index.php/catalog/903",
        "notes": [
            "Útil para enriquecer priorización social con indicadores agregados.",
            "Se recomienda usar tablas agregadas por territorio, no microdatos individuales.",
        ],
    }


def get_dane_nbi_metadata() -> dict[str, object]:
    """Devuelve metadatos útiles para NBI de DANE."""
    return {
        "name": "DANE - Necesidades Básicas Insatisfechas (NBI)",
        "provider": "DANE",
        "official_url": "https://www.dane.gov.co/index.php/estadisticas-por-tema/salud/calidad-de-vida-ecv/necesidades-basicas-insatisfechas",
        "reference_url": "https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/necesidades-basicas-insatisfechas-nbi?id=134&view=category",
        "notes": [
            "Sirve como señal agregada de privación territorial.",
            "Debe usarse de forma responsable y sin estigmatizar territorios.",
        ],
    }


def get_sisben_open_data_metadata() -> dict[str, object]:
    """Devuelve metadatos orientativos sobre fuentes abiertas o configurables de SISBÉN agregado."""
    return {
        "name": "SISBÉN agregado / Datos abiertos configurables",
        "provider": "SISBÉN / DNP / Datos Abiertos Colombia",
        "official_url": "https://www.sisben.gov.co/Paginas/landing.html",
        "open_data_portal": "https://www.datos.gov.co/",
        "notes": [
            "Solo usar tablas agregadas o muestras anonimizadas.",
            "No cargar ni procesar identificadores personales de hogares o personas.",
            "El conector Socrata es configurable porque la disponibilidad territorial puede cambiar por dataset.",
        ],
    }


def fetch_socrata_dataset(
    domain: str,
    dataset_id: str,
    query: dict[str, Any] | str | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """Consulta un dataset Socrata de manera configurable y con cache."""
    clean_domain = str(domain or "").strip().rstrip("/")
    clean_dataset_id = str(dataset_id or "").strip()
    if not clean_domain or not clean_dataset_id:
        return pd.DataFrame()

    base_url = f"https://{clean_domain}/resource/{clean_dataset_id}.json"
    params: dict[str, Any] = {"$limit": int(limit)}
    if isinstance(query, dict):
        params.update(query)
    elif isinstance(query, str) and query.strip():
        params["$query"] = query.strip()

    payload = get_with_cache(
        source_name="socrata_dataset",
        url=base_url,
        params=params,
        ttl_hours=24,
    )
    if payload is None:
        return pd.DataFrame()

    try:
        dataframe = pd.DataFrame(payload)
    except ValueError:
        return pd.DataFrame()

    return normalize_socioeconomic_columns(dataframe)
