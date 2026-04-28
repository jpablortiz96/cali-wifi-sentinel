from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR, PROJECT_ROOT
from src.utils import build_storage_filename


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}


class DataLoaderError(Exception):
    """Error controlado para lectura o guardado de archivos."""


def detect_file_extension(file_name: str) -> str:
    """Valida y retorna la extensión del archivo cargado."""
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DataLoaderError(
            "Formato no soportado. Usa un archivo .csv, .xlsx, .xls o .txt."
        )

    return extension


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Intenta leer un archivo delimitado tipo CSV/TXT con codificaciones comunes."""
    candidate_encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_error: Exception | None = None

    for encoding in candidate_encodings:
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except Exception as error:  # noqa: BLE001
            last_error = error

    raise DataLoaderError(
        "No fue posible leer el archivo CSV/TXT con las configuraciones básicas "
        "de carga."
    ) from last_error


def load_excel(file_bytes: bytes, extension: str) -> pd.DataFrame:
    """Lee la primera hoja del archivo Excel según su extensión."""
    engine = "openpyxl" if extension == ".xlsx" else "xlrd"

    try:
        return pd.read_excel(BytesIO(file_bytes), engine=engine)
    except Exception as error:  # noqa: BLE001
        raise DataLoaderError(
            "No fue posible leer el archivo Excel. Verifica que no esté dañado."
        ) from error


def load_tabular_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Carga un archivo tabular sin asumir columnas específicas."""
    if not file_bytes:
        raise DataLoaderError("El archivo cargado está vacío.")

    extension = detect_file_extension(file_name)

    try:
        if extension in {".csv", ".txt"}:
            dataframe = load_csv(file_bytes)
        else:
            dataframe = load_excel(file_bytes, extension)
    except DataLoaderError:
        raise
    except Exception as error:  # noqa: BLE001
        raise DataLoaderError(
            "Ocurrió un error inesperado al cargar el archivo."
        ) from error

    return dataframe


def save_uploaded_file(file_name: str, file_bytes: bytes) -> Path:
    """Guarda una copia del archivo original en data/raw/."""
    output_path = DATA_RAW_DIR / build_storage_filename(file_name)

    try:
        output_path.write_bytes(file_bytes)
    except OSError as error:
        raise DataLoaderError(
            "No fue posible guardar una copia del archivo en data/raw/."
        ) from error

    return output_path


def list_local_datasets() -> list[Path]:
    """Lista datasets tabulares disponibles en data/raw/ y en la raíz del proyecto."""
    discovered_paths: list[Path] = []

    for search_dir in [DATA_RAW_DIR, PROJECT_ROOT]:
        if not search_dir.exists():
            continue

        iterator = search_dir.iterdir() if search_dir == PROJECT_ROOT else search_dir.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            discovered_paths.append(path.resolve())

    unique_paths = sorted(
        {path for path in discovered_paths},
        key=lambda item: item.as_posix().lower(),
    )
    return unique_paths


def load_local_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un dataset local disponible sin hardcodear su nombre."""
    dataset_path = Path(path).resolve()
    if not dataset_path.exists() or not dataset_path.is_file():
        raise DataLoaderError("El dataset local seleccionado no existe o no es un archivo.")

    if dataset_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DataLoaderError("El dataset local seleccionado no tiene una extensión soportada.")

    try:
        file_bytes = dataset_path.read_bytes()
    except OSError as error:
        raise DataLoaderError("No fue posible leer el dataset local seleccionado.") from error

    return load_tabular_file(dataset_path.name, file_bytes)
