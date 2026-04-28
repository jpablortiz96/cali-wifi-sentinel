from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_CACHE_DIR = PROJECT_ROOT / "data" / "external_cache"
DATA_OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


# Asegura que las carpetas existan incluso si el proyecto se mueve o se clona aparte.
for directory in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EXTERNAL_CACHE_DIR,
    DATA_OUTPUTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
