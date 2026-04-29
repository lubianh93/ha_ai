"""User-editable model catalog helpers."""

from __future__ import annotations

import json
from typing import Any


def make_catalog(models: list[str], notes: dict[str, str] | None = None) -> str:
    """Return a pretty JSON catalog from model ids and optional notes."""
    notes = notes or {}
    return json.dumps(
        [{"id": model, "note": notes.get(model, "")} for model in models],
        ensure_ascii=False,
        indent=2,
    )


def parse_catalog(raw: Any, fallback: list[str]) -> list[dict[str, str]]:
    """Parse a user-editable model catalog.

    Accepted JSON forms:
    - [{"id": "model", "note": "price or status"}]
    - ["model-a", "model-b"]
    - {"model-a": "note"}
    """
    if not isinstance(raw, str) or not raw.strip():
        return [{"id": model, "note": ""} for model in fallback]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [{"id": model, "note": ""} for model in fallback]

    items: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                items.append({"id": item.strip(), "note": ""})
            elif isinstance(item, dict):
                model_id = str(item.get("id") or item.get("model") or "").strip()
                if model_id:
                    items.append({"id": model_id, "note": str(item.get("note") or "").strip()})
    elif isinstance(data, dict):
        for model_id, note in data.items():
            model_id = str(model_id).strip()
            if model_id:
                items.append({"id": model_id, "note": str(note or "").strip()})

    return items or [{"id": model, "note": ""} for model in fallback]


def validate_catalog(raw: Any) -> None:
    """Validate a user-editable model catalog.

    The parser is intentionally forgiving for display, but the config flow
    should reject malformed JSON before saving it to avoid silent surprises.
    """
    if raw is None or raw == "":
        return
    if not isinstance(raw, str):
        raise ValueError("model catalog must be a JSON string")

    data = json.loads(raw)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                if not item.strip():
                    raise ValueError("model id cannot be empty")
                continue
            if isinstance(item, dict):
                model_id = str(item.get("id") or item.get("model") or "").strip()
                if not model_id:
                    raise ValueError("model item must include id")
                continue
            raise ValueError("model catalog list items must be strings or objects")
        return

    if isinstance(data, dict):
        for model_id in data:
            if not str(model_id).strip():
                raise ValueError("model id cannot be empty")
        return

    raise ValueError("model catalog must be a list or object")


def selector_options(raw: Any, fallback: list[str]) -> list[dict[str, str]]:
    """Build Home Assistant select options with notes shown as labels."""
    options = []
    seen: set[str] = set()
    for item in parse_catalog(raw, fallback):
        model_id = item["id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        note = item.get("note", "")
        options.append({
            "value": model_id,
            "label": f"{model_id} - {note}" if note else model_id,
        })
    return options
