from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pandas as pd

from src.utils import get_timestamp


def create_audit_event(
    module: str,
    action: str,
    status: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Construye un evento de auditoria trazable."""
    return {
        "audit_id": f"AUD-{uuid4().hex[:10]}",
        "timestamp": get_timestamp(),
        "module": module,
        "action": action,
        "status": status,
        "message": message,
        "metadata": metadata or {},
    }


def append_audit_event(
    audit_log: list[dict[str, object]] | None,
    event: dict[str, object],
) -> list[dict[str, object]]:
    """Agrega un evento y devuelve la bitacora actualizada."""
    current_log = list(audit_log or [])
    current_log.append(event)
    return current_log


def audit_log_to_dataframe(audit_log: list[dict[str, object]] | None) -> pd.DataFrame:
    """Convierte la bitacora a tabla descargable."""
    if not audit_log:
        return pd.DataFrame(
            columns=["audit_id", "timestamp", "module", "action", "status", "message", "metadata"]
        )

    rows = []
    for event in audit_log:
        row = dict(event)
        metadata = row.get("metadata", {})
        row["metadata"] = json.dumps(metadata, ensure_ascii=False, default=str)
        rows.append(row)

    return pd.DataFrame(rows)


def build_operational_audit_summary(audit_log: list[dict[str, object]] | None) -> dict[str, object]:
    """Resume la bitacora operativa en indicadores simples."""
    if not audit_log:
        return {
            "eventos_totales": 0,
            "eventos_ok": 0,
            "advertencias": 0,
            "errores": 0,
            "ultimos_eventos": [],
            "modulos_ejecutados": [],
        }

    dataframe = audit_log_to_dataframe(audit_log)
    status_counts = dataframe["status"].fillna("warning").astype(str).value_counts()
    modules = sorted(dataframe["module"].dropna().astype(str).unique().tolist())

    last_events = []
    for _, row in dataframe.tail(5).iterrows():
        last_events.append(
            {
                "timestamp": row.get("timestamp"),
                "module": row.get("module"),
                "action": row.get("action"),
                "status": row.get("status"),
                "message": row.get("message"),
            }
        )

    return {
        "eventos_totales": int(len(dataframe)),
        "eventos_ok": int(status_counts.get("ok", 0)),
        "advertencias": int(status_counts.get("warning", 0)),
        "errores": int(status_counts.get("error", 0)),
        "ultimos_eventos": last_events,
        "modulos_ejecutados": modules,
    }
