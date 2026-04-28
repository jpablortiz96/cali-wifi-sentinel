from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils import get_timestamp


REVIEW_COLUMNS = [
    "order_id",
    "zona",
    "tipo_alerta",
    "prioridad",
    "evidencia",
    "accion_recomendada",
    "nivel_confianza",
    "estado_revision",
    "comentario_operador",
    "reviewed_at",
]

ALLOWED_REVIEW_STATUSES = {
    "pendiente",
    "aprobada",
    "rechazada",
    "requiere_visita",
    "cerrada",
}


def _empty_review_queue() -> pd.DataFrame:
    """Devuelve una cola vacia con el esquema esperado."""
    return pd.DataFrame(columns=REVIEW_COLUMNS)


def create_review_queue(work_orders: pd.DataFrame | Iterable[dict[str, object]] | None) -> pd.DataFrame:
    """Convierte ordenes en una cola de revision humana."""
    if work_orders is None:
        return _empty_review_queue()

    if isinstance(work_orders, pd.DataFrame):
        if work_orders.empty:
            return _empty_review_queue()
        source_df = work_orders.copy()
    else:
        source_df = pd.DataFrame(list(work_orders))
        if source_df.empty:
            return _empty_review_queue()

    queue_df = pd.DataFrame(
        {
            "order_id": source_df.get("id", pd.Series(dtype="object")).astype(str),
            "zona": source_df.get("zona", pd.Series(dtype="object")).astype(str),
            "tipo_alerta": source_df.get("tipo_alerta", pd.Series(dtype="object")).astype(str),
            "prioridad": source_df.get("prioridad", pd.Series(dtype="object")).astype(str),
            "evidencia": source_df.get("evidencia", pd.Series(dtype="object")).astype(str),
            "accion_recomendada": source_df.get("accion_recomendada", pd.Series(dtype="object")).astype(str),
            "nivel_confianza": source_df.get("nivel_confianza", pd.Series(dtype="object")).astype(str),
        }
    )

    queue_df["estado_revision"] = "pendiente"
    queue_df["comentario_operador"] = ""
    queue_df["reviewed_at"] = None

    queue_df = queue_df.drop_duplicates(subset=["order_id"]).reset_index(drop=True)
    return queue_df[REVIEW_COLUMNS]


def update_work_order_status(
    review_queue: pd.DataFrame,
    order_id: str,
    new_status: str,
    comment: str = "",
) -> pd.DataFrame:
    """Actualiza el estado de revision de una orden sin romper la cola."""
    if review_queue is None or review_queue.empty:
        return _empty_review_queue()

    normalized_status = str(new_status).strip().lower()
    if normalized_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(
            "Estado no permitido. Usa: pendiente, aprobada, rechazada, requiere_visita o cerrada."
        )

    updated_queue = review_queue.copy()
    mask = updated_queue["order_id"].astype(str) == str(order_id)
    if not mask.any():
        raise ValueError(f"No existe una orden con id '{order_id}' en la cola de revision.")

    updated_queue.loc[mask, "estado_revision"] = normalized_status
    updated_queue.loc[mask, "comentario_operador"] = str(comment).strip()
    updated_queue.loc[mask, "reviewed_at"] = get_timestamp()
    return updated_queue[REVIEW_COLUMNS]


def bulk_update_work_orders(
    review_queue: pd.DataFrame,
    new_status: str,
    comment: str = "",
    only_current_filter: bool = False,
    selected_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Aplica un cambio masivo de estado a todas o a un subconjunto de órdenes."""
    if review_queue is None or review_queue.empty:
        return _empty_review_queue()

    normalized_status = str(new_status).strip().lower()
    if normalized_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(
            "Estado no permitido. Usa: pendiente, aprobada, rechazada, requiere_visita o cerrada."
        )

    updated_queue = review_queue.copy()
    if selected_ids:
        selected_set = {str(order_id) for order_id in selected_ids}
        mask = updated_queue["order_id"].astype(str).isin(selected_set)
    else:
        mask = pd.Series([True] * len(updated_queue), index=updated_queue.index)

    if only_current_filter and not selected_ids:
        mask &= updated_queue["estado_revision"].astype(str).eq("pendiente")

    if not mask.any():
        return updated_queue[REVIEW_COLUMNS]

    updated_queue.loc[mask, "estado_revision"] = normalized_status
    updated_queue.loc[mask, "comentario_operador"] = str(comment).strip()
    updated_queue.loc[mask, "reviewed_at"] = get_timestamp()
    return updated_queue[REVIEW_COLUMNS]


def summarize_human_review(review_queue: pd.DataFrame) -> dict[str, object]:
    """Resume el estado actual de la revision humana."""
    if review_queue is None or review_queue.empty:
        return {
            "total_ordenes": 0,
            "pendientes": 0,
            "aprobadas": 0,
            "rechazadas": 0,
            "requiere_visita": 0,
            "cerradas": 0,
            "porcentaje_revisado": 0.0,
        }

    counts = review_queue["estado_revision"].fillna("pendiente").astype(str).value_counts()
    total_orders = int(len(review_queue))
    reviewed_count = total_orders - int(counts.get("pendiente", 0))
    reviewed_percentage = round((reviewed_count / total_orders) * 100, 2) if total_orders else 0.0

    return {
        "total_ordenes": total_orders,
        "pendientes": int(counts.get("pendiente", 0)),
        "aprobadas": int(counts.get("aprobada", 0)),
        "rechazadas": int(counts.get("rechazada", 0)),
        "requiere_visita": int(counts.get("requiere_visita", 0)),
        "cerradas": int(counts.get("cerrada", 0)),
        "porcentaje_revisado": reviewed_percentage,
    }


def export_review_log(review_queue: pd.DataFrame) -> pd.DataFrame:
    """Entrega una copia lista para descargar."""
    if review_queue is None or review_queue.empty:
        return _empty_review_queue()
    return review_queue.copy()[REVIEW_COLUMNS]
