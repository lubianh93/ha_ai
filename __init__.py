"""The HA AI integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import aiohttp

try:
    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.const import CONF_API_KEY, Platform
    from homeassistant.core import HomeAssistant
    from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
except ModuleNotFoundError:  # pragma: no cover - used only in lightweight test environments
    ConfigEntry = Any  # type: ignore[assignment]
    ConfigSubentry = Any  # type: ignore[assignment]
    HomeAssistant = Any  # type: ignore[assignment]
    CONF_API_KEY = "api_key"

    class ConfigEntryAuthFailed(Exception):
        """Fallback exception when Home Assistant is not installed."""

    class ConfigEntryNotReady(Exception):
        """Fallback exception when Home Assistant is not installed."""

    Platform = None  # type: ignore[assignment]

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

if Platform is None:
    PLATFORMS: list[Any] = []
else:
    PLATFORMS = [
        Platform.CONVERSATION,
        Platform.AI_TASK,
        Platform.TTS,
        Platform.STT,
        Platform.SENSOR,
    ]

HAAIConfigEntry: TypeAlias = ConfigEntry  # Store API key


@dataclass
class HAAIData:
    """Runtime data for HA AI integration.

    This class holds all runtime state for the integration,
    avoiding global variables and ensuring proper cleanup on reload.
    """

    api_key: str | None = None
    tts_cache: Any = None
    automation_manager: Any = None
    proactive_manager: Any = None
    provider_registry: Any = None
    diagnostics_collector: Any = None
    stats: dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.tts_cache is not None:
            self.tts_cache.clear()
        self.automation_manager = None
        self.proactive_manager = None
        self.provider_registry = None


def get_ha_ai_data(hass: HomeAssistant) -> HAAIData | None:
    """Get HA AI runtime data."""
    return hass.data.get(DOMAIN)


def get_or_create_ha_ai_data(hass: HomeAssistant) -> HAAIData:
    """Get or create HA AI runtime data."""
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = HAAIData()
    return hass.data[DOMAIN]


def get_provider_registry(hass: HomeAssistant):
    """Get or create the provider registry for this Home Assistant instance.

    This ensures proper lifecycle management - the registry is cleaned up
    when the integration is unloaded.

    Args:
        hass: Home Assistant instance

    Returns:
        UnifiedProviderRegistry instance
    """
    from .providers import get_registry

    ha_ai_data = get_or_create_ha_ai_data(hass)
    if ha_ai_data.provider_registry is None:
        ha_ai_data.provider_registry = get_registry()
    return ha_ai_data.provider_registry


async def async_setup_entry(hass: HomeAssistant, entry: HAAIConfigEntry) -> bool:
    """Set up HA AI from a config entry."""

    # Get API key (may be None if not provided)
    api_key = entry.data.get(CONF_API_KEY)

    # Store API key in runtime_data for subentries/entities
    entry.runtime_data = api_key

    from .proactive import get_proactive_manager
    from .services import async_setup_services

    await get_proactive_manager(hass).async_register_entry(entry)
    await async_setup_services(hass, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: HAAIConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload services first
    from .services import async_unload_services
    await async_unload_services(hass, entry.entry_id)

    from .proactive import get_proactive_manager
    await get_proactive_manager(hass).async_unregister_entry(entry.entry_id)

    # Clean up runtime data
    ha_ai_data = get_ha_ai_data(hass)
    if ha_ai_data is not None:
        ha_ai_data.cleanup()
        hass.data.pop(DOMAIN, None)

    # Unload all platforms
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating configuration from version %s.%s", entry.version, entry.minor_version)

    if entry.version == 1:
        # Migrate from version 1 to version 2
        # Version 2 uses subentries for conversation and ai_task
        new_data = {**entry.data}

        from .const import (
            DEFAULT_AI_TASK_NAME,
            DEFAULT_CONVERSATION_NAME,
            DEFAULT_STT_NAME,
            DEFAULT_TTS_NAME,
            RECOMMENDED_AI_TASK_OPTIONS,
            RECOMMENDED_CONVERSATION_OPTIONS,
            RECOMMENDED_STT_OPTIONS,
            RECOMMENDED_TTS_OPTIONS,
        )

        hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            options={},
            version=2,
            minor_version=3,
        )

        defaults = (
            ("conversation", DEFAULT_CONVERSATION_NAME, RECOMMENDED_CONVERSATION_OPTIONS),
            ("ai_task_data", DEFAULT_AI_TASK_NAME, RECOMMENDED_AI_TASK_OPTIONS),
            ("tts", DEFAULT_TTS_NAME, RECOMMENDED_TTS_OPTIONS),
            ("stt", DEFAULT_STT_NAME, RECOMMENDED_STT_OPTIONS),
        )
        for subentry_type, title, data in defaults:
            hass.config_entries.async_add_subentry(
                entry,
                ConfigSubentry(
                    data=dict(data),
                    subentry_type=subentry_type,
                    title=title,
                    unique_id=None,
                ),
            )

        _LOGGER.debug("Migration to version %s.%s successful", entry.version, entry.minor_version)

    if entry.version == 2 and entry.minor_version == 1:
        # Migrate from version 2.1 to 2.2
        # Update subentry titles
        from .const import DEFAULT_AI_TASK_NAME, DEFAULT_CONVERSATION_NAME

        for subentry in entry.subentries.values():
            # Update old titles to new format
            if subentry.subentry_type == "conversation":
                if subentry.title in ("对话助手", "Conversation Agent"):
                    hass.config_entries.async_update_subentry(
                        entry, subentry.subentry_id, title=DEFAULT_CONVERSATION_NAME
                    )
            elif subentry.subentry_type == "ai_task_data":
                if subentry.title in ("AI任务", "AI Task"):
                    hass.config_entries.async_update_subentry(
                        entry, subentry.subentry_id, title=DEFAULT_AI_TASK_NAME
                    )

        hass.config_entries.async_update_entry(
            entry,
            minor_version=2,
        )

        _LOGGER.debug("Migration to version %s.%s successful", entry.version, entry.minor_version)

    if entry.version == 2 and entry.minor_version < 3:
        from .const import (
            DEFAULT_AI_TASK_NAME,
            DEFAULT_CONVERSATION_NAME,
            DEFAULT_STT_NAME,
            DEFAULT_TTS_NAME,
            RECOMMENDED_AI_TASK_OPTIONS,
            RECOMMENDED_CONVERSATION_OPTIONS,
            RECOMMENDED_STT_OPTIONS,
            RECOMMENDED_TTS_OPTIONS,
        )

        defaults = {
            "conversation": (DEFAULT_CONVERSATION_NAME, RECOMMENDED_CONVERSATION_OPTIONS),
            "ai_task_data": (DEFAULT_AI_TASK_NAME, RECOMMENDED_AI_TASK_OPTIONS),
            "tts": (DEFAULT_TTS_NAME, RECOMMENDED_TTS_OPTIONS),
            "stt": (DEFAULT_STT_NAME, RECOMMENDED_STT_OPTIONS),
        }
        existing = {subentry.subentry_type for subentry in entry.subentries.values()}
        for subentry_type, (title, data) in defaults.items():
            if subentry_type in existing:
                continue
            hass.config_entries.async_add_subentry(
                entry,
                ConfigSubentry(
                    data=dict(data),
                    subentry_type=subentry_type,
                    title=title,
                    unique_id=None,
                ),
            )

        hass.config_entries.async_update_entry(entry, minor_version=3)
        _LOGGER.debug("Migration to version %s.%s successful", entry.version, entry.minor_version)

    return True
