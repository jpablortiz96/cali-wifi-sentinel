from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.utils import get_timestamp, sanitize_filename


def convert_to_serializable(obj: object) -> object:
    """Convierte objetos de pandas y numpy a tipos JSON serializables."""
    if obj is None:
        return None

    if not isinstance(obj, (dict, list, tuple, set, pd.DataFrame, pd.Series, np.ndarray)):
        try:
            if pd.isna(obj):
                return None
        except TypeError:
            pass

    if isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(item) for item in obj]

    if isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict(orient="records"))

    if isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())

    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)

    if obj is pd.NA:
        return None

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def save_profile_json(profile: dict[str, object], original_filename: str) -> Path:
    """Guarda el perfil estructural del dataset como JSON en data/processed/."""
    filename = (
        f"profile_{get_timestamp()}_{sanitize_filename(original_filename)}.json"
    )
    output_path = DATA_PROCESSED_DIR / filename
    serializable_profile = convert_to_serializable(profile)

    output_path.write_text(
        json.dumps(serializable_profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path
