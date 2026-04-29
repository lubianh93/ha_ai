"""Helpers to resolve effective entry config for subentries."""
from __future__ import annotations

from typing import Any
import json

from .const import (
    CONF_API_KEYS,
    CONF_CUSTOM_API_KEY,
    CONF_PROVIDER_KEY,
)


def _get_subentry_by_type(
    entry: Any,
    subentry_type: str,
    subentry_id: str | None = None,
) -> Any | None:
    """Return the requested subentry, or the first matching type."""
    for subentry in getattr(entry, "subentries", {}).values():
        if subentry.subentry_type != subentry_type:
            continue
        if subentry_id and getattr(subentry, "subentry_id", None) != subentry_id:
            continue
        if subentry_id and getattr(subentry, "subentry_id", None) == subentry_id:
            return subentry
        if not subentry_id:
            return subentry
    return None


def _get_subentry_value(
    entry: Any,
    subentry_type: str,
    key: str,
    default: Any,
    subentry_id: str | None = None,
) -> Any:
    """Return a value from the first matching subentry, or the default."""
    subentry = _get_subentry_by_type(entry, subentry_type, subentry_id)
    if subentry is None:
        return default
    return subentry.data.get(key, default)


def resolve_entry_config(
    entry: Any,
    subentry_type: str,
    *values: tuple[str, Any],
    subentry_id: str | None = None,
) -> tuple[Any, ...]:
    """Return resolved subentry values followed by the effective API key."""
    effective_api_key = entry.runtime_data
    subentry = _get_subentry_by_type(entry, subentry_type, subentry_id)

    if subentry is not None:
        provider_key = str(subentry.data.get(CONF_PROVIDER_KEY, "") or "").strip()
        api_keys_raw = (
            getattr(entry, "options", {}).get(CONF_API_KEYS)
            or getattr(entry, "data", {}).get(CONF_API_KEYS, "")
        )
        if provider_key and isinstance(api_keys_raw, str) and api_keys_raw.strip():
            try:
                api_keys = json.loads(api_keys_raw)
            except json.JSONDecodeError:
                api_keys = {}
            if isinstance(api_keys, dict):
                mapped_key = str(api_keys.get(provider_key, "") or "").strip()
                if mapped_key:
                    effective_api_key = mapped_key

        custom_api_key = str(subentry.data.get(CONF_CUSTOM_API_KEY, "") or "").strip()
        if custom_api_key:
            effective_api_key = custom_api_key

    resolved_values = tuple(
        _get_subentry_value(entry, subentry_type, key, default, subentry_id)
        for key, default in values
    )
    return (*resolved_values, effective_api_key)
