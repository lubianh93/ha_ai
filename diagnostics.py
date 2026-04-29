"""Diagnostics support for HA AI integration.

This module provides diagnostic information collection for troubleshooting and
debugging purposes. The diagnostics data is accessible through Home Assistant's
built-in diagnostics feature.

Features:
- API configuration diagnostics (with sensitive data redacted)
- Service status information
- Endpoint reachability checks based on effective subentry URLs
- Recent error logs
- Performance metrics
- Configuration validation results
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

try:
    from homeassistant.components.diagnostics import async_redact_data
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
except ModuleNotFoundError:  # pragma: no cover - used only in lightweight test environments
    ConfigEntry = Any  # type: ignore[assignment]
    HomeAssistant = Any  # type: ignore[assignment]

    def async_redact_data(data: dict[str, Any], to_redact: set[str]) -> dict[str, Any]:
        """Fallback redaction helper when Home Assistant is unavailable."""
        redacted = {}
        for key, value in data.items():
            if key in to_redact:
                redacted[key] = "REDACTED"
            else:
                redacted[key] = value
        return redacted

from .const import (
    CONF_API_KEYS,
    CONF_API_KEY,
    CONF_CHAT_URL,
    CONF_CUSTOM_API_KEY,
    CONF_IMAGE_URL,
    CONF_PROVIDER_KEY,
    CONF_STT_URL,
    CONF_TTS_PROVIDER,
    CONF_TTS_URL,
    DEFAULT_CHAT_URL,
    DEFAULT_IMAGE_URL,
    DEFAULT_STT_URL,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_URL,
    DOMAIN,
    RETRY_BASE_DELAY,
    RETRY_MAX_ATTEMPTS,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
    TIMEOUT_CHAT_API,
    TIMEOUT_IMAGE_API,
    TIMEOUT_STT_API,
    TIMEOUT_TTS_API,
)

EDGE_TTS_BASE_URL = "https://speech.platform.bing.com"

_LOGGER = logging.getLogger(__name__)

# Keys that contain sensitive data and should be redacted
TO_REDACT = {
    CONF_API_KEY,
    CONF_API_KEYS,
    CONF_CUSTOM_API_KEY,
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
}


def _is_configured(value: Any) -> bool:
    """Return whether a value should be treated as configured."""
    return isinstance(value, str) and bool(value.strip())


def _effective_api_key_for_subentry(entry: ConfigEntry, subentry: Any) -> str:
    """Return the effective API key for a subentry."""
    custom_key = subentry.data.get(CONF_CUSTOM_API_KEY, "")
    if _is_configured(custom_key):
        return custom_key

    provider_key = str(subentry.data.get(CONF_PROVIDER_KEY, "") or "").strip()
    api_keys_raw = entry.options.get(CONF_API_KEYS) or entry.data.get(CONF_API_KEYS, "")
    if provider_key and isinstance(api_keys_raw, str) and api_keys_raw.strip():
        try:
            api_keys = json.loads(api_keys_raw)
        except json.JSONDecodeError:
            api_keys = {}
        if isinstance(api_keys, dict):
            mapped_key = str(api_keys.get(provider_key, "") or "").strip()
            if mapped_key:
                return mapped_key

    runtime_key = getattr(entry, "runtime_data", None)
    if _is_configured(runtime_key):
        return runtime_key

    primary_key = entry.data.get(CONF_API_KEY, "")
    if _is_configured(primary_key):
        return primary_key

    return ""


def _effective_url_for_subentry(subentry: Any) -> str | None:
    """Return the effective endpoint URL for a subentry."""
    if subentry.subentry_type == SUBENTRY_CONVERSATION:
        return subentry.data.get(CONF_CHAT_URL, DEFAULT_CHAT_URL)
    if subentry.subentry_type == SUBENTRY_AI_TASK:
        return subentry.data.get(CONF_IMAGE_URL, DEFAULT_IMAGE_URL)
    if subentry.subentry_type == SUBENTRY_STT:
        return subentry.data.get(CONF_STT_URL, DEFAULT_STT_URL)
    if subentry.subentry_type == SUBENTRY_TTS:
        if subentry.data.get(CONF_TTS_PROVIDER, DEFAULT_TTS_PROVIDER) == DEFAULT_TTS_PROVIDER:
            return EDGE_TTS_BASE_URL
        return subentry.data.get(CONF_TTS_URL, DEFAULT_TTS_URL)
    return None


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    diagnostics: dict[str, Any] = {
        "ha_ai": {
            "integration": {
                "domain": DOMAIN,
                "title": entry.title,
                "entry_id": entry.entry_id,
                "version": entry.version,
                "minor_version": entry.minor_version,
            },
            "configuration": await _get_configuration_diagnostics(hass, entry),
            "subentries": await _get_subentries_diagnostics(hass, entry),
            "entities": await _get_entities_diagnostics(hass, entry),
            "api_status": await _get_api_status_diagnostics(hass, entry),
            "timeout_config": _get_timeout_config(),
            "retry_config": _get_retry_config(),
            "statistics": await _get_statistics_diagnostics(hass, entry),
            "system_info": _get_system_info(hass),
        }
    }
    return diagnostics


async def _get_configuration_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Get configuration diagnostics with sensitive data redacted."""
    primary_key = entry.data.get(CONF_API_KEY, "")
    runtime_key = getattr(entry, "runtime_data", None)

    config = {
        "data": async_redact_data(dict(entry.data), TO_REDACT),
        "options": async_redact_data(dict(entry.options), TO_REDACT),
        "primary_api_key_configured": _is_configured(primary_key),
        "primary_api_key_length": len(primary_key) if _is_configured(primary_key) else 0,
        "runtime_api_key_available": _is_configured(runtime_key),
        "runtime_api_key_length": len(runtime_key) if _is_configured(runtime_key) else 0,
    }

    return config


async def _get_subentries_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Get diagnostics for all subentries."""
    subentries_info: dict[str, Any] = {}

    if not entry.subentries:
        return {"count": 0, "entries": []}

    for subentry_id, subentry in entry.subentries.items():
        subentry_data = async_redact_data(dict(subentry.data), TO_REDACT)
        effective_key = _effective_api_key_for_subentry(entry, subentry)
        effective_url = _effective_url_for_subentry(subentry)

        subentries_info[subentry_id] = {
            "type": subentry.subentry_type,
            "title": subentry.title,
            "data": subentry_data,
            "custom_api_key_configured": _is_configured(
                subentry.data.get(CONF_CUSTOM_API_KEY, "")
            ),
            "effective_api_key_configured": _is_configured(effective_key),
            "effective_api_key_length": len(effective_key) if _is_configured(effective_key) else 0,
            "effective_url": effective_url,
        }

    return {
        "count": len(entry.subentries),
        "entries": subentries_info,
    }


async def _get_entities_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Get diagnostics for all entities created by this integration."""
    from homeassistant.helpers import entity_registry as er

    entity_registry = er.async_get(hass)
    entities_info: dict[str, list[dict[str, Any]]] = {
        "conversation": [],
        "ai_task": [],
        "tts": [],
        "stt": [],
        "button": [],
        "sensor": [],
        "other": [],
    }

    for entity_entry in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
        entity_data = {
            "entity_id": entity_entry.entity_id,
            "unique_id": entity_entry.unique_id,
            "name": entity_entry.name or entity_entry.original_name,
            "platform": entity_entry.platform,
            "disabled": entity_entry.disabled,
            "disabled_by": str(entity_entry.disabled_by) if entity_entry.disabled_by else None,
        }

        domain = entity_entry.domain
        if domain in entities_info:
            entities_info[domain].append(entity_data)
        else:
            entities_info["other"].append(entity_data)

    total_entities = sum(len(entities) for entities in entities_info.values())

    return {
        "total_count": total_entities,
        "by_type": {
            domain: {"count": len(entities), "entities": entities}
            for domain, entities in entities_info.items()
            if entities
        },
    }


async def _probe_url(session: Any, url: str, timeout_seconds: int = 10) -> dict[str, Any]:
    """Probe a URL and return reachability information."""
    import aiohttp

    try:
        start_time = datetime.now()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as response:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "reachable": True,
                "http_status": response.status,
                "latency_ms": round(latency, 2),
            }
    except aiohttp.ClientError as exc:
        return {"reachable": False, "error_type": "client", "error": str(exc)}
    except TimeoutError as exc:
        return {"reachable": False, "error_type": "timeout", "error": str(exc)}
    except Exception as exc:  # pragma: no cover
        return {"reachable": False, "error_type": "other", "error": str(exc)}


def _normalize_monitor_url(url: str) -> str | None:
    """Normalize a configured endpoint to a base URL for probing."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def collect_api_monitor_targets(entry: ConfigEntry) -> list[dict[str, Any]]:
    """Collect endpoint targets that should be monitored."""
    targets: dict[str, dict[str, Any]] = {}

    def add_target(configured_url: str, label: str, source: str) -> None:
        normalized = _normalize_monitor_url(configured_url)
        if normalized is None:
            return

        parsed = urlparse(configured_url)
        path_key = parsed.path.strip("/").replace("/", "_") or "root"
        key = (
            f"{label.lower().replace(' ', '_')}_"
            f"{parsed.netloc.replace('.', '_').replace(':', '_')}_{path_key}"
        )

        if configured_url not in targets:
            targets[configured_url] = {
                "key": key,
                "label": label,
                "url": configured_url,
                "monitor_url": normalized,
                "sources": [source],
            }
            return

        if source not in targets[configured_url]["sources"]:
            targets[configured_url]["sources"].append(source)

    for subentry in entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_CONVERSATION:
            add_target(
                subentry.data.get(CONF_CHAT_URL, DEFAULT_CHAT_URL),
                "Conversation API",
                subentry.title,
            )
        elif subentry.subentry_type == SUBENTRY_AI_TASK:
            add_target(
                subentry.data.get(CONF_IMAGE_URL, DEFAULT_IMAGE_URL),
                "AI Task Image API",
                subentry.title,
            )
        elif subentry.subentry_type == SUBENTRY_STT:
            add_target(
                subentry.data.get(CONF_STT_URL, DEFAULT_STT_URL),
                "STT API",
                subentry.title,
            )
        elif subentry.subentry_type == SUBENTRY_TTS:
            if subentry.data.get(CONF_TTS_PROVIDER, DEFAULT_TTS_PROVIDER) == DEFAULT_TTS_PROVIDER:
                add_target(EDGE_TTS_BASE_URL, "Edge TTS API", subentry.title)
            else:
                add_target(
                    subentry.data.get(CONF_TTS_URL, DEFAULT_TTS_URL),
                    "TTS API",
                    subentry.title,
                )

    return sorted(targets.values(), key=lambda target: target["label"])


async def _get_api_status_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Get API endpoint status diagnostics."""
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    session = async_get_clientsession(hass)
    status: dict[str, Any] = {}

    for target in collect_api_monitor_targets(entry):
        probe = await _probe_url(session, target["monitor_url"], timeout_seconds=10)
        status[target["key"]] = {
            "label": target["label"],
            "url": target["url"],
            "monitor_url": target["monitor_url"],
            "sources": target["sources"],
            "status": (
                "reachable"
                if probe.get("reachable")
                else "unreachable"
                if probe.get("error_type") in {"client", "timeout"}
                else "error"
            ),
        }

        if probe.get("reachable"):
            status[target["key"]]["http_status"] = probe["http_status"]
            status[target["key"]]["latency_ms"] = probe["latency_ms"]
        else:
            status[target["key"]]["error"] = probe.get("error")

    return status


def _get_timeout_config() -> dict[str, Any]:
    """Get current timeout configuration."""
    return {
        "chat_api_seconds": TIMEOUT_CHAT_API,
        "image_api_seconds": TIMEOUT_IMAGE_API,
        "stt_api_seconds": TIMEOUT_STT_API,
        "tts_api_seconds": TIMEOUT_TTS_API,
    }


def _get_retry_config() -> dict[str, Any]:
    """Get current retry configuration."""
    return {
        "max_attempts": RETRY_MAX_ATTEMPTS,
        "base_delay_seconds": RETRY_BASE_DELAY,
    }


async def _get_statistics_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Get usage statistics diagnostics."""
    integration_data = hass.data.get(DOMAIN, {})
    stats: dict[str, Any] = {
        "note": "Statistics tracking not yet implemented",
        "integration_data_available": bool(integration_data),
    }

    if entry.entry_id in integration_data:
        entry_data = integration_data[entry.entry_id]
        if isinstance(entry_data, dict) and "stats" in entry_data:
            stats["cached_stats"] = entry_data["stats"]

    return stats


def _get_system_info(hass: HomeAssistant) -> dict[str, Any]:
    """Get relevant system information."""
    import sys

    return {
        "python_version": sys.version,
        "home_assistant_version": hass.config.version,
        "timezone": str(hass.config.time_zone),
        "language": hass.config.language,
        "units": hass.config.units.name if hass.config.units else "unknown",
        "timestamp": datetime.now().isoformat(),
    }


class DiagnosticsCollector:
    """Helper class for collecting diagnostics data during runtime."""

    def __init__(self) -> None:
        """Initialize the diagnostics collector."""
        self._api_calls: dict[str, list[dict[str, Any]]] = {}
        self._errors: list[dict[str, Any]] = []
        self._start_time = datetime.now()

    def record_api_call(
        self,
        api_name: str,
        success: bool,
        latency_ms: float | None = None,
        **extra: Any,
    ) -> None:
        """Record an API call."""
        if api_name not in self._api_calls:
            self._api_calls[api_name] = []

        record = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "latency_ms": latency_ms,
            **extra,
        }
        self._api_calls[api_name].append(record)

        if len(self._api_calls[api_name]) > 100:
            self._api_calls[api_name] = self._api_calls[api_name][-100:]

    def record_error(
        self,
        context: str,
        error: str,
        **extra: Any,
    ) -> None:
        """Record an error."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error": error,
            **extra,
        }
        self._errors.append(record)

        if len(self._errors) > 50:
            self._errors = self._errors[-50:]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected diagnostics."""
        uptime = datetime.now() - self._start_time
        api_summary: dict[str, Any] = {}

        for api_name, calls in self._api_calls.items():
            total = len(calls)
            successful = sum(1 for c in calls if c.get("success"))
            latencies = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]

            api_summary[api_name] = {
                "total_calls": total,
                "successful_calls": successful,
                "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "min_latency_ms": round(min(latencies), 2) if latencies else None,
                "max_latency_ms": round(max(latencies), 2) if latencies else None,
            }

        return {
            "uptime_seconds": uptime.total_seconds(),
            "api_summary": api_summary,
            "recent_errors": self._errors[-10:],
            "total_errors": len(self._errors),
        }

    def clear(self) -> None:
        """Clear all collected data."""
        self._api_calls.clear()
        self._errors.clear()
        self._start_time = datetime.now()


_diagnostics_collector: DiagnosticsCollector | None = None


def get_diagnostics_collector() -> DiagnosticsCollector:
    """Get or create the global diagnostics collector."""
    global _diagnostics_collector
    if _diagnostics_collector is None:
        _diagnostics_collector = DiagnosticsCollector()
    return _diagnostics_collector
