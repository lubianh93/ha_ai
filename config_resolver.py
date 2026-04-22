"""Helpers to resolve effective entry config for subentries."""
from __future__ import annotations

from typing import Any

from .const import (
    CONF_CUSTOM_API_KEY,
)


def _get_subentry_by_type(entry: Any, subentry_type: str) -> Any | None:
    """Return the first subentry matching the requested type."""
    for subentry in getattr(entry, "subentries", {}).values():
        if subentry.subentry_type == subentry_type:
            return subentry
    return None


def _get_subentry_value(entry: Any, subentry_type: str, key: str, default: Any) -> Any:
    """Return a value from the first matching subentry, or the default."""
    subentry = _get_subentry_by_type(entry, subentry_type)
    if subentry is None:
        return default
    return subentry.data.get(key, default)


def resolve_entry_config(
    entry: Any,
    subentry_type: str,
    *values: tuple[str, Any],
) -> tuple[Any, ...]:
    """Return resolved subentry values followed by the effective API key."""
    effective_api_key = entry.runtime_data
    subentry = _get_subentry_by_type(entry, subentry_type)

    if subentry is not None:
        custom_api_key = subentry.data.get(CONF_CUSTOM_API_KEY, "").strip()
        if custom_api_key:
            effective_api_key = custom_api_key

    resolved_values = tuple(
        _get_subentry_value(entry, subentry_type, key, default)
        for key, default in values
    )
    return (*resolved_values, effective_api_key)