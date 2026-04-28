from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from src.config import DATA_EXTERNAL_CACHE_DIR


def get_default_headers() -> dict[str, str]:
    """Devuelve un User-Agent identificable para llamadas HTTP responsables."""
    return {
        "User-Agent": "CaliWiFiSentinel360-Hackathon/1.0 contact: info@eduky.co",
        "Accept": "application/json",
    }


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> dict[str, Any] | list[Any] | None:
    """Hace una peticion GET segura y devuelve JSON o None si falla."""
    request_headers = get_default_headers()
    if headers:
        request_headers.update(headers)

    try:
        response = requests.get(
            url,
            params=params,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def safe_post_json(
    url: str,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> dict[str, Any] | list[Any] | None:
    """Hace una peticion POST segura y devuelve JSON o None si falla."""
    request_headers = get_default_headers()
    if headers:
        request_headers.update(headers)

    try:
        response = requests.post(
            url,
            data=data,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def cache_key_from_payload(source_name: str, payload: dict[str, Any]) -> str:
    """Construye una llave estable de cache a partir de un payload."""
    serialized_payload = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.md5(serialized_payload.encode("utf-8")).hexdigest()
    return f"{source_name}_{digest}"


def _cache_path(cache_key: str) -> Path:
    """Devuelve la ruta del archivo de cache asociada a la llave."""
    return DATA_EXTERNAL_CACHE_DIR / f"{cache_key}.json"


def load_cache(cache_key: str) -> dict[str, Any] | None:
    """Busca un JSON de cache y devuelve su contenido bruto."""
    cache_path = _cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cache(cache_key: str, data: dict[str, Any] | list[Any]) -> Path:
    """Guarda un payload JSON con metadata minima de cache."""
    cache_path = _cache_path(cache_key)
    wrapper = {
        "cached_at": datetime.utcnow().isoformat(),
        "data": data,
    }
    cache_path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def _is_cache_fresh(cached_payload: dict[str, Any], ttl_hours: int) -> bool:
    """Evalua si un cache sigue vigente segun TTL."""
    cached_at_raw = cached_payload.get("cached_at")
    if not cached_at_raw:
        return False

    try:
        cached_at = datetime.fromisoformat(str(cached_at_raw))
    except ValueError:
        return False

    return datetime.utcnow() - cached_at <= timedelta(hours=ttl_hours)


def get_with_cache(
    source_name: str,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    ttl_hours: int = 720,
) -> dict[str, Any] | list[Any] | None:
    """Consulta primero el cache y solo despues intenta una peticion GET."""
    payload = {
        "url": url,
        "params": params or {},
        "headers": headers or {},
    }
    cache_key = cache_key_from_payload(source_name, payload)
    cached_payload = load_cache(cache_key)

    if cached_payload and _is_cache_fresh(cached_payload, ttl_hours):
        return cached_payload.get("data")

    live_data = safe_get_json(url, params=params, headers=headers)
    if live_data is not None:
        save_cache(cache_key, live_data)
        return live_data

    if cached_payload:
        return cached_payload.get("data")

    return None


def post_with_cache(
    source_name: str,
    url: str,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    ttl_hours: int = 720,
) -> dict[str, Any] | list[Any] | None:
    """Variante con cache para peticiones POST, util en Overpass."""
    payload = {
        "url": url,
        "data": data or {},
        "headers": headers or {},
    }
    cache_key = cache_key_from_payload(source_name, payload)
    cached_payload = load_cache(cache_key)

    if cached_payload and _is_cache_fresh(cached_payload, ttl_hours):
        return cached_payload.get("data")

    live_data = safe_post_json(url, data=data, headers=headers)
    if live_data is not None:
        save_cache(cache_key, live_data)
        return live_data

    if cached_payload:
        return cached_payload.get("data")

    return None
