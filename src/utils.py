from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path


def sanitize_filename(file_name: str) -> str:
    """Limpia el nombre del archivo para guardarlo de forma consistente."""
    clean_name = Path(file_name).stem.strip().lower()
    clean_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return clean_name or "archivo"


def get_timestamp() -> str:
    """Genera una marca de tiempo legible para versionar cargas."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_text(value: str) -> str:
    """Normaliza texto para comparaciones heuristicas simples."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def build_storage_filename(file_name: str) -> str:
    """Crea el nombre final del archivo que se guardará en data/raw/."""
    extension = Path(file_name).suffix.lower()
    return f"{get_timestamp()}_{sanitize_filename(file_name)}{extension}"
