"""Health check sensor for HA AI integration.

This module provides a sensor that monitors the health and status
of the HA AI integration and its connected APIs.

Features:
- API connectivity monitoring
- Latency tracking
- Error rate monitoring
- Automatic status updates
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    TIMEOUT_HEALTH_CHECK,
)
from .diagnostics import collect_api_monitor_targets, get_diagnostics_collector
from .proactive import get_proactive_manager

_LOGGER = logging.getLogger(__name__)

# Update interval for health checks (5 minutes)
SCAN_INTERVAL = timedelta(minutes=5)

# Diagnostic device identifier
DIAGNOSTIC_DEVICE_ID = "diagnostic"


def _get_diagnostic_device_info(entry: ConfigEntry) -> dr.DeviceInfo:
    """Get device info for the diagnostic service."""
    return dr.DeviceInfo(
        identifiers={(DOMAIN, f"{entry.entry_id}_{DIAGNOSTIC_DEVICE_ID}")},
        name="HA AI Diagnostic",
        manufacturer="Fork自老王杂谈说",
        model="Diagnostic Service",
        entry_type=dr.DeviceEntryType.SERVICE,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up health check sensor entities."""
    entities = []

    # Main integration health sensor (always added)
    entities.append(HAAIHealthCheckSensor(hass, entry))

    # Edge TTS health sensor (free provider, commonly used)
    entities.append(EdgeTTSHealthSensor(hass, entry))

    # Proactive assistant and habit learning status
    entities.append(HAAIProactiveStatusSensor(hass, entry))

    async_add_entities(entities)


class HAAIHealthCheckSensor(SensorEntity):
    """Sensor for overall HA AI health status."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["unknown", "disabled", "active"]
    _attr_icon = "mdi:heart-pulse"
    _attr_should_poll = True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the health sensor."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_health_sensor"
        self._attr_name = "Health Status"
        self._attr_device_info = _get_diagnostic_device_info(entry)

        self._api_statuses: dict[str, dict[str, Any]] = {}
        self._last_check: datetime | None = None
        self._diagnostics = get_diagnostics_collector()

    async def async_added_to_hass(self) -> None:
        """Run initial update when entity is added."""
        await super().async_added_to_hass()
        # Trigger immediate update on startup
        await self.async_update()

    @property
    def native_value(self) -> str:
        """Return the health status."""
        if not self._api_statuses:
            return "unknown"

        # Check if all APIs are healthy
        all_healthy = all(
            status.get("status") == "healthy"
            for status in self._api_statuses.values()
        )

        if all_healthy:
            return "healthy"

        # Check if any API is down
        any_down = any(
            status.get("status") == "unreachable"
            for status in self._api_statuses.values()
        )

        if any_down:
            return "degraded"

        return "unknown"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        attrs: dict[str, Any] = {
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "apis": self._api_statuses,
        }

        # Add diagnostics summary
        try:
            summary = self._diagnostics.get_summary()
            attrs["uptime_seconds"] = summary.get("uptime_seconds")
            attrs["total_errors"] = summary.get("total_errors", 0)
            attrs["api_summary"] = summary.get("api_summary", {})
        except Exception as e:
            _LOGGER.debug("Failed to get diagnostics summary: %s", e)

        return attrs

    async def async_update(self) -> None:
        """Update the health status."""
        session = async_get_clientsession(self.hass)

        self._api_statuses = {}
        for target in collect_api_monitor_targets(self._entry):
            self._api_statuses[target["key"]] = await self._check_api(
                session,
                target["monitor_url"],
                target["label"],
            )

        self._last_check = datetime.now()

    async def _check_api(
        self,
        session: aiohttp.ClientSession,
        url: str,
        name: str,
    ) -> dict[str, Any]:
        """Check if an API endpoint is reachable.

        Args:
            session: aiohttp session
            url: URL to check
            name: Name of the API for logging

        Returns:
            Status dictionary
        """
        try:
            start_time = datetime.now()
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_HEALTH_CHECK),
            ) as response:
                latency = (datetime.now() - start_time).total_seconds() * 1000

                return {
                    "status": "healthy" if response.status < 500 else "degraded",
                    "http_status": response.status,
                    "latency_ms": round(latency, 2),
                    "checked_at": datetime.now().isoformat(),
                }

        except aiohttp.ClientError as e:
            _LOGGER.debug("%s health check failed: %s", name, e)
            return {
                "status": "unreachable",
                "error": str(e),
                "checked_at": datetime.now().isoformat(),
            }

        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"Timeout after {TIMEOUT_HEALTH_CHECK}s",
                "checked_at": datetime.now().isoformat(),
            }

        except aiohttp.ClientError as err:
            _LOGGER.warning("%s health check error: %s", name, err)
            return {
                "status": "error",
                "error": str(err),
                "checked_at": datetime.now().isoformat(),
            }


class _BaseHealthSensor(SensorEntity):
    """Base class for API health check sensors.

    This class provides common functionality for all health sensors that check
    API endpoints and measure latency. Subclasses only need to define the
    check URL and optional customizations.
    """

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "ms"
    _attr_icon = "mdi:api"
    _attr_should_poll = True

    # Subclasses should override these
    _check_url: str
    _name_suffix: str

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_{self._name_suffix}_latency"
        self._attr_name = f"{self._name_suffix.replace('_', ' ').title()} Latency"
        self._attr_device_info = _get_diagnostic_device_info(entry)

        self._latency: float | None = None
        self._status: str = "unknown"
        self._last_check: datetime | None = None

    async def async_added_to_hass(self) -> None:
        """Run initial update when entity is added."""
        await super().async_added_to_hass()
        await self.async_update()

    @property
    def native_value(self) -> float | None:
        """Return the latency in milliseconds."""
        return self._latency

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        return {
            "status": self._status,
            "last_check": self._last_check.isoformat() if self._last_check else None,
        }

    async def async_update(self) -> None:
        """Update the sensor."""
        session = async_get_clientsession(self.hass)

        try:
            start_time = datetime.now()
            async with session.get(
                self._check_url,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_HEALTH_CHECK),
            ) as response:
                self._latency = round(
                    (datetime.now() - start_time).total_seconds() * 1000,
                    2,
                )
                self._status = "healthy" if response.status < 500 else "degraded"

        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            _LOGGER.debug("%s latency check failed: %s", self._name_suffix, err)
            self._latency = None
            self._status = "unreachable"

        self._last_check = datetime.now()


class EdgeTTSHealthSensor(_BaseHealthSensor):
    """Sensor for Edge TTS API health and latency."""

    _check_url = "https://speech.platform.bing.com"
    _name_suffix = "edge_tts"
    _attr_icon = "mdi:text-to-speech"


class HAAIProactiveStatusSensor(SensorEntity):
    """Sensor showing proactive follow-up and habit learning status."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_icon = "mdi:account-voice"
    _attr_should_poll = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_proactive_status"
        self._attr_name = "Proactive Status"
        self._attr_device_info = _get_diagnostic_device_info(entry)
        self._status: dict[str, Any] = {}

    @property
    def native_value(self) -> str:
        """Return the proactive feature state."""
        settings = self._status.get("settings", {})
        if not settings:
            return "unknown"
        if settings.get("follow_up_enabled") or settings.get("habit_learning_enabled"):
            return "active"
        return "disabled"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return proactive settings and current counters."""
        settings = self._status.get("settings", {})
        return {
            "follow_up_enabled": settings.get("follow_up_enabled", False),
            "follow_up_timeout_seconds": settings.get("follow_up_timeout_seconds"),
            "follow_up_max_attempts": settings.get("follow_up_max_attempts"),
            "habit_learning_enabled": settings.get("habit_learning_enabled", False),
            "habit_min_observations": settings.get("habit_min_observations"),
            "habit_confidence_threshold": settings.get("habit_confidence_threshold"),
            "habit_temperature_sensors": settings.get("habit_temperature_sensors", []),
            "habit_presence_entities": settings.get("habit_presence_entities", []),
            "habit_action_domains": settings.get("habit_action_domains", []),
            "pending_count": self._status.get("pending_count", 0),
            "habit_candidate_count": self._status.get("habit_candidate_count", 0),
            "recent_habit_events": self._status.get("recent_habit_events", [])[-5:],
        }

    async def async_update(self) -> None:
        """Update proactive status."""
        self._status = await get_proactive_manager(self.hass).async_status()
