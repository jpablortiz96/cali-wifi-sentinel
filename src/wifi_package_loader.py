from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import DATA_RAW_DIR, PROJECT_ROOT
from src.data_loader import DataLoaderError, load_tabular_file
from src.external_sources import get_default_headers


EXPECTED_PACKAGE_FILES = {
    "events": "network_events_curated.csv",
    "clients": "clients_curated.csv",
    "access_points": "access_points_curated.csv",
    "hourly_metrics": "ap_hourly_metrics_curated.csv",
    "data_dictionary": "data_dictionary.csv",
}


def _empty_package(source: str = "") -> dict[str, object]:
    """Construye un paquete vacio sin romper el flujo de la app."""
    return {
        "events": pd.DataFrame(),
        "clients": pd.DataFrame(),
        "access_points": pd.DataFrame(),
        "hourly_metrics": pd.DataFrame(),
        "data_dictionary": pd.DataFrame(),
        "source": source,
        "is_official_package": False,
        "warnings": [],
    }


def list_wifi_package_files(base_path: str | Path) -> list[Path]:
    """Lista archivos candidatos del paquete oficial dentro de una carpeta o subcarpetas."""
    base_dir = Path(base_path).resolve()
    if not base_dir.exists():
        return []

    candidates: list[Path] = []
    for file_name in EXPECTED_PACKAGE_FILES.values():
        matches = list(base_dir.rglob(file_name))
        candidates.extend(matches)

    unique_paths = sorted({path.resolve() for path in candidates})
    return unique_paths


def detect_official_wifi_package(base_path: str | Path) -> dict[str, object]:
    """Detecta si una carpeta contiene el paquete Meraki esperado."""
    base_dir = Path(base_path).resolve()
    files = list_wifi_package_files(base_dir)
    file_map = {path.name.lower(): path for path in files}

    found_files = {
        logical_name: file_map.get(expected_name.lower())
        for logical_name, expected_name in EXPECTED_PACKAGE_FILES.items()
    }
    missing_files = [
        expected_name
        for logical_name, expected_name in EXPECTED_PACKAGE_FILES.items()
        if found_files.get(logical_name) is None
    ]

    return {
        "base_path": str(base_dir),
        "found_files": found_files,
        "missing_files": missing_files,
        "is_official_package": len(found_files) > 0 and len(missing_files) < len(EXPECTED_PACKAGE_FILES),
    }


def _load_file_path_to_dataframe(file_path: Path) -> pd.DataFrame:
    """Carga un archivo del paquete usando el loader tabular existente."""
    file_bytes = file_path.read_bytes()
    return load_tabular_file(file_path.name, file_bytes)


def load_wifi_package_from_folder(base_path: str | Path) -> dict[str, object]:
    """Carga el paquete Meraki oficial desde una carpeta local."""
    detection = detect_official_wifi_package(base_path)
    package = _empty_package(source=f"folder:{detection['base_path']}")
    warnings: list[str] = []

    for logical_name, expected_name in EXPECTED_PACKAGE_FILES.items():
        path = detection["found_files"].get(logical_name)
        if path is None:
            warnings.append(f"No se encontró `{expected_name}` en la carpeta seleccionada.")
            continue

        try:
            package[logical_name] = _load_file_path_to_dataframe(path)
        except (DataLoaderError, OSError) as error:
            warnings.append(f"No fue posible cargar `{expected_name}`: {error}")

    package["warnings"] = warnings
    package["is_official_package"] = validate_wifi_package(package)["is_valid"]
    return package


def _github_raw_candidates(repo_url: str) -> list[str]:
    """Construye posibles URLs raw para un repo de GitHub."""
    clean_url = str(repo_url).strip().rstrip("/")
    if "github.com" not in clean_url:
        return []

    repo_path = clean_url.split("github.com/", maxsplit=1)[-1].strip("/")
    if repo_path.endswith(".git"):
        repo_path = repo_path[:-4]

    branches = ["main", "master"]
    candidates = []
    for branch in branches:
        candidates.append(f"https://raw.githubusercontent.com/{repo_path}/{branch}")
    return candidates


def _download_raw_csv(url: str) -> pd.DataFrame:
    """Descarga un CSV raw desde GitHub sin asumir codificación especial."""
    response = requests.get(url, headers=get_default_headers(), timeout=30)
    response.raise_for_status()
    file_name = url.rsplit("/", maxsplit=1)[-1]
    return load_tabular_file(file_name, response.content)


def load_wifi_package_from_github(repo_url: str) -> dict[str, object]:
    """Carga el paquete oficial Meraki directamente desde GitHub."""
    package = _empty_package(source=f"github:{repo_url}")
    warnings: list[str] = []

    raw_roots = _github_raw_candidates(repo_url)
    if not raw_roots:
        package["warnings"] = ["La URL suministrada no parece corresponder a un repositorio válido de GitHub."]
        return package

    selected_root = None
    for raw_root in raw_roots:
        test_url = f"{raw_root}/{EXPECTED_PACKAGE_FILES['data_dictionary']}"
        try:
            response = requests.get(test_url, headers=get_default_headers(), timeout=15)
            if response.ok:
                selected_root = raw_root
                break
        except requests.RequestException:
            continue

    if selected_root is None:
        package["warnings"] = [
            "No fue posible encontrar el paquete en GitHub usando ramas `main` o `master`."
        ]
        return package

    for logical_name, file_name in EXPECTED_PACKAGE_FILES.items():
        file_url = f"{selected_root}/{file_name}"
        try:
            package[logical_name] = _download_raw_csv(file_url)
        except (requests.RequestException, DataLoaderError) as error:
            warnings.append(f"No fue posible descargar `{file_name}` desde GitHub: {error}")

    package["warnings"] = warnings
    package["is_official_package"] = validate_wifi_package(package)["is_valid"]
    return package


def validate_wifi_package(package: dict[str, object]) -> dict[str, object]:
    """Valida que el paquete tenga suficiente estructura para operar en modo Meraki."""
    warnings = list(package.get("warnings", [])) if isinstance(package.get("warnings"), list) else []
    table_status: dict[str, str] = {}

    for logical_name in EXPECTED_PACKAGE_FILES:
        table_df = package.get(logical_name)
        if isinstance(table_df, pd.DataFrame) and not table_df.empty:
            table_status[logical_name] = "ok"
        elif isinstance(table_df, pd.DataFrame):
            table_status[logical_name] = "empty"
            warnings.append(f"La tabla `{logical_name}` fue detectada pero está vacía.")
        else:
            table_status[logical_name] = "missing"

    is_valid = any(table_status.get(key) == "ok" for key in ["hourly_metrics", "access_points", "events"])
    return {
        "is_valid": is_valid,
        "table_status": table_status,
        "warnings": warnings,
    }


def get_package_summary(package: dict[str, object]) -> dict[str, Any]:
    """Resume el paquete oficial para UI, logs y diagnóstico."""
    summary_tables: list[dict[str, object]] = []
    for logical_name, file_name in EXPECTED_PACKAGE_FILES.items():
        table_df = package.get(logical_name)
        if isinstance(table_df, pd.DataFrame):
            summary_tables.append(
                {
                    "tabla": logical_name,
                    "archivo_esperado": file_name,
                    "filas": int(len(table_df)),
                    "columnas": int(table_df.shape[1]),
                    "columnas_detectadas": ", ".join(table_df.columns.astype(str).tolist()[:12]),
                }
            )
        else:
            summary_tables.append(
                {
                    "tabla": logical_name,
                    "archivo_esperado": file_name,
                    "filas": 0,
                    "columnas": 0,
                    "columnas_detectadas": "No disponible",
                }
            )

    validation = validate_wifi_package(package)
    return {
        "source": package.get("source", ""),
        "is_official_package": bool(package.get("is_official_package")),
        "warnings": validation["warnings"],
        "table_summary": pd.DataFrame(summary_tables),
        "available_tables": [
            logical_name
            for logical_name in EXPECTED_PACKAGE_FILES
            if isinstance(package.get(logical_name), pd.DataFrame) and not package[logical_name].empty
        ],
    }

