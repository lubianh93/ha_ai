"""Proactive follow-up and habit learning helpers for HA AI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    CONF_FOLLOW_UP_ENABLED,
    CONF_FOLLOW_UP_MAX_ATTEMPTS,
    CONF_FOLLOW_UP_TIMEOUT_SECONDS,
    CONF_HABIT_ACTION_DOMAINS,
    CONF_HABIT_CONFIDENCE_THRESHOLD,
    CONF_HABIT_LEARNING_ENABLED,
    CONF_HABIT_MIN_OBSERVATIONS,
    CONF_HABIT_PRESENCE_ENTITIES,
    CONF_HABIT_TEMPERATURE_SENSORS,
    DEFAULT_FOLLOW_UP_MAX_ATTEMPTS,
    DEFAULT_FOLLOW_UP_TIMEOUT_SECONDS,
    DEFAULT_HABIT_ACTION_DOMAINS,
    DEFAULT_HABIT_CONFIDENCE_THRESHOLD,
    DEFAULT_HABIT_MIN_OBSERVATIONS,
    DOMAIN,
    EVENT_FOLLOW_UP_LISTEN_REQUESTED,
    EVENT_HABIT_CANDIDATE_UPDATED,
)

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_proactive"
_MANAGER_KEY = f"{DOMAIN}_proactive_manager"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt_value: datetime) -> str:
    return dt_value.isoformat()


def _parse_csv(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


@dataclass
class ProactiveSettings:
    """Effective proactive assistant settings."""

    follow_up_enabled: bool = False
    follow_up_timeout_seconds: int = DEFAULT_FOLLOW_UP_TIMEOUT_SECONDS
    follow_up_max_attempts: int = DEFAULT_FOLLOW_UP_MAX_ATTEMPTS
    habit_learning_enabled: bool = False
    habit_min_observations: int = DEFAULT_HABIT_MIN_OBSERVATIONS
    habit_confidence_threshold: float = DEFAULT_HABIT_CONFIDENCE_THRESHOLD
    habit_temperature_sensors: list[str] | None = None
    habit_presence_entities: list[str] | None = None
    habit_action_domains: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable settings snapshot."""
        return {
            "follow_up_enabled": self.follow_up_enabled,
            "follow_up_timeout_seconds": self.follow_up_timeout_seconds,
            "follow_up_max_attempts": self.follow_up_max_attempts,
            "habit_learning_enabled": self.habit_learning_enabled,
            "habit_min_observations": self.habit_min_observations,
            "habit_confidence_threshold": self.habit_confidence_threshold,
            "habit_temperature_sensors": self.habit_temperature_sensors or [],
            "habit_presence_entities": self.habit_presence_entities or [],
            "habit_action_domains": self.habit_action_domains or [],
        }


class HAAIProactiveManager:
    """Manage short-lived follow-up questions and habit candidates."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._store = Store[dict[str, Any]](hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._data: dict[str, Any] | None = None
        self._entries: dict[str, ConfigEntry] = {}
        self._remove_call_service_listener = None

    async def async_register_entry(self, entry: ConfigEntry) -> None:
        """Register a config entry for proactive settings."""
        self._entries[entry.entry_id] = entry
        await self.async_load()
        self._refresh_listeners()

    async def async_unregister_entry(self, entry_id: str) -> None:
        """Unregister a config entry."""
        self._entries.pop(entry_id, None)
        self._refresh_listeners()

    async def async_load(self) -> dict[str, Any]:
        """Load persistent proactive data."""
        if self._data is not None:
            return self._data
        data = await self._store.async_load()
        if not isinstance(data, dict):
            data = {}
        data.setdefault("pending_followups", {})
        data.setdefault("habit_candidates", {})
        data.setdefault("recent_habit_events", [])
        self._data = data
        return data

    async def async_save(self) -> None:
        """Persist proactive data."""
        await self._store.async_save(await self.async_load())

    def settings(self) -> ProactiveSettings:
        """Return effective settings from the first registered entry."""
        options: dict[str, Any] = {}
        for entry in self._entries.values():
            options = dict(getattr(entry, "options", {}) or {})
            break

        follow_timeout = options.get(
            CONF_FOLLOW_UP_TIMEOUT_SECONDS,
            DEFAULT_FOLLOW_UP_TIMEOUT_SECONDS,
        )
        max_attempts = options.get(
            CONF_FOLLOW_UP_MAX_ATTEMPTS,
            DEFAULT_FOLLOW_UP_MAX_ATTEMPTS,
        )
        min_observations = options.get(
            CONF_HABIT_MIN_OBSERVATIONS,
            DEFAULT_HABIT_MIN_OBSERVATIONS,
        )
        confidence = options.get(
            CONF_HABIT_CONFIDENCE_THRESHOLD,
            DEFAULT_HABIT_CONFIDENCE_THRESHOLD,
        )

        try:
            follow_timeout = int(follow_timeout)
        except (TypeError, ValueError):
            follow_timeout = DEFAULT_FOLLOW_UP_TIMEOUT_SECONDS
        try:
            max_attempts = int(max_attempts)
        except (TypeError, ValueError):
            max_attempts = DEFAULT_FOLLOW_UP_MAX_ATTEMPTS
        try:
            min_observations = int(min_observations)
        except (TypeError, ValueError):
            min_observations = DEFAULT_HABIT_MIN_OBSERVATIONS
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = DEFAULT_HABIT_CONFIDENCE_THRESHOLD

        return ProactiveSettings(
            follow_up_enabled=_as_bool(options.get(CONF_FOLLOW_UP_ENABLED), False),
            follow_up_timeout_seconds=max(5, follow_timeout),
            follow_up_max_attempts=max(1, max_attempts),
            habit_learning_enabled=_as_bool(
                options.get(CONF_HABIT_LEARNING_ENABLED),
                False,
            ),
            habit_min_observations=max(1, min_observations),
            habit_confidence_threshold=min(1.0, max(0.0, confidence)),
            habit_temperature_sensors=_parse_csv(options.get(CONF_HABIT_TEMPERATURE_SENSORS)),
            habit_presence_entities=_parse_csv(options.get(CONF_HABIT_PRESENCE_ENTITIES)),
            habit_action_domains=_parse_csv(
                options.get(CONF_HABIT_ACTION_DOMAINS, DEFAULT_HABIT_ACTION_DOMAINS)
            ),
        )

    def _refresh_listeners(self) -> None:
        """Enable or disable event listeners based on settings."""
        enabled = self.settings().habit_learning_enabled
        if enabled and self._remove_call_service_listener is None:
            self._remove_call_service_listener = self.hass.bus.async_listen(
                "call_service",
                self._handle_call_service_event,
            )
            return
        if not enabled and self._remove_call_service_listener is not None:
            self._remove_call_service_listener()
            self._remove_call_service_listener = None

    async def async_create_pending_follow_up(
        self,
        *,
        original_text: str,
        question_text: str,
        device_id: str | None = None,
        conversation_id: str | None = None,
        missing_slot: str | None = None,
        source: str = "conversation",
    ) -> dict[str, Any] | None:
        """Create a pending follow-up transaction."""
        settings = self.settings()
        if not settings.follow_up_enabled:
            return None

        data = await self.async_load()
        pending_id = uuid4().hex
        expires_at = _utcnow() + timedelta(seconds=settings.follow_up_timeout_seconds)
        pending = {
            "pending_id": pending_id,
            "state": "waiting_tts_done",
            "source": source,
            "original_text": original_text,
            "question_text": question_text,
            "missing_slot": missing_slot or "",
            "device_id": device_id or "",
            "conversation_id": conversation_id or "",
            "attempt_count": 0,
            "max_attempts": settings.follow_up_max_attempts,
            "created_at": _iso(_utcnow()),
            "expires_at": _iso(expires_at),
        }
        data["pending_followups"][pending_id] = pending
        await self.async_save()
        return pending

    async def async_handle_playback_done(
        self,
        *,
        pending_id: str | None = None,
        device_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark TTS playback as done and request follow-up listening."""
        pending = await self._find_pending(
            pending_id=pending_id,
            device_id=device_id,
            conversation_id=conversation_id,
        )
        if pending is None:
            return {"success": False, "error": "No matching pending follow-up"}

        if self._is_expired(pending):
            await self.async_resolve_pending(pending["pending_id"], "expired")
            return {"success": False, "error": "Pending follow-up expired"}

        pending["state"] = "waiting_user_reply"
        pending["attempt_count"] = int(pending.get("attempt_count", 0)) + 1
        pending["playback_done_at"] = _iso(_utcnow())
        await self.async_save()

        event_data = dict(pending)
        self.hass.bus.async_fire(EVENT_FOLLOW_UP_LISTEN_REQUESTED, event_data)
        return {
            "success": True,
            "pending": dict(pending),
            "event": EVENT_FOLLOW_UP_LISTEN_REQUESTED,
        }

    async def async_match_pending_reply(self, user_input: Any) -> dict[str, Any] | None:
        """Return a pending follow-up that can consume the next user reply."""
        settings = self.settings()
        if not settings.follow_up_enabled:
            return None

        device_id = str(getattr(user_input, "device_id", "") or "")
        conversation_id = str(getattr(user_input, "conversation_id", "") or "")
        pending = await self._find_pending(device_id=device_id, conversation_id=conversation_id)
        if pending is None or self._is_expired(pending):
            return None
        if pending.get("state") not in ("waiting_user_reply", "waiting_tts_done"):
            return None
        return pending

    async def async_resolve_pending(self, pending_id: str, state: str = "resolved") -> None:
        """Resolve or expire a pending follow-up."""
        data = await self.async_load()
        pending = data["pending_followups"].get(pending_id)
        if pending:
            pending["state"] = state
            pending["resolved_at"] = _iso(_utcnow())
            await self.async_save()

    def expand_follow_up_reply(self, pending: dict[str, Any], reply_text: str) -> str:
        """Expand a short follow-up answer into an actionable command."""
        reply = reply_text.strip()
        original = str(pending.get("original_text", "") or "")
        if not reply:
            return reply_text

        open_words = ("开", "打开", "开启", "帮我开")
        close_words = ("关", "关闭", "关掉", "帮我关")
        if any(word in original for word in open_words):
            if not any(word in reply for word in open_words):
                return f"打开{reply}"
        if any(word in original for word in close_words):
            if not any(word in reply for word in close_words):
                return f"关闭{reply}"
        return f"{original}，具体是{reply}"

    async def async_record_habit_event(
        self,
        *,
        domain: str,
        service: str,
        entity_id: str,
        device_id: str | None = None,
        source: str = "service",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a habit-learning observation."""
        settings = self.settings()
        if not settings.habit_learning_enabled:
            return {"success": False, "error": "Habit learning is disabled"}

        data = await self.async_load()
        key = f"{domain}.{service}:{entity_id}"
        candidate = data["habit_candidates"].setdefault(
            key,
            {
                "key": key,
                "domain": domain,
                "service": service,
                "entity_id": entity_id,
                "observations": 0,
                "last_seen": "",
                "temperature_samples": [],
                "presence_samples": {},
                "time_buckets": {},
                "suggestion_enabled": False,
                "created_switch": False,
            },
        )

        now = _utcnow()
        hour_bucket = f"{now.hour:02d}:00"
        candidate["observations"] = int(candidate.get("observations", 0)) + 1
        candidate["last_seen"] = _iso(now)
        candidate["time_buckets"][hour_bucket] = int(
            candidate["time_buckets"].get(hour_bucket, 0)
        ) + 1

        temperatures = self._snapshot_temperature(settings)
        if temperatures:
            candidate["temperature_samples"].append(temperatures)
            candidate["temperature_samples"] = candidate["temperature_samples"][-20:]

        presence = self._snapshot_presence(settings)
        if presence:
            for presence_entity, state in presence.items():
                samples = candidate["presence_samples"].setdefault(presence_entity, {})
                samples[state] = int(samples.get(state, 0)) + 1

        if extra:
            candidate["last_extra"] = extra
        if device_id:
            candidate["last_device_id"] = device_id
        candidate["source"] = source

        data["recent_habit_events"].append({
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "device_id": device_id or "",
            "source": source,
            "recorded_at": _iso(now),
        })
        data["recent_habit_events"] = data["recent_habit_events"][-50:]

        await self.async_save()
        self.hass.bus.async_fire(EVENT_HABIT_CANDIDATE_UPDATED, dict(candidate))
        return {"success": True, "candidate": dict(candidate)}

    async def async_status(self) -> dict[str, Any]:
        """Return current proactive status."""
        data = await self.async_load()
        pending = {
            key: value
            for key, value in data["pending_followups"].items()
            if value.get("state") not in ("resolved", "expired")
        }
        return {
            "settings": self.settings().to_dict(),
            "pending_followups": pending,
            "pending_count": len(pending),
            "habit_candidate_count": len(data["habit_candidates"]),
            "habit_candidates": data["habit_candidates"],
            "recent_habit_events": data["recent_habit_events"],
        }

    async def _find_pending(
        self,
        *,
        pending_id: str | None = None,
        device_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any] | None:
        data = await self.async_load()
        if pending_id:
            pending = data["pending_followups"].get(pending_id)
            return pending if isinstance(pending, dict) else None

        candidates = [
            item for item in data["pending_followups"].values()
            if isinstance(item, dict)
            and item.get("state") not in ("resolved", "expired")
        ]
        candidates.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        for pending in candidates:
            if device_id and pending.get("device_id") == device_id:
                return pending
            if conversation_id and pending.get("conversation_id") == conversation_id:
                return pending
        return None

    def _is_expired(self, pending: dict[str, Any]) -> bool:
        try:
            expires_at = datetime.fromisoformat(str(pending.get("expires_at", "")))
        except ValueError:
            return True
        return expires_at < _utcnow()

    def _snapshot_temperature(self, settings: ProactiveSettings) -> dict[str, float]:
        samples: dict[str, float] = {}
        for entity_id in settings.habit_temperature_sensors or []:
            state = self.hass.states.get(entity_id)
            if state is None:
                continue
            try:
                samples[entity_id] = round(float(state.state), 1)
            except (TypeError, ValueError):
                continue
        return samples

    def _snapshot_presence(self, settings: ProactiveSettings) -> dict[str, str]:
        samples: dict[str, str] = {}
        for entity_id in settings.habit_presence_entities or []:
            state = self.hass.states.get(entity_id)
            if state is not None:
                samples[entity_id] = str(state.state)
        return samples

    def _handle_call_service_event(self, event: Event) -> None:
        """Capture user-facing service calls as habit observations."""
        settings = self.settings()
        domain = str(event.data.get("domain", "") or "")
        service = str(event.data.get("service", "") or "")
        if domain not in (settings.habit_action_domains or []):
            return

        service_data = event.data.get("service_data") or {}
        entity_ids = service_data.get("entity_id")
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        if not isinstance(entity_ids, list):
            return

        device_id = ""
        context = getattr(event, "context", None)
        if context is not None:
            device_id = str(getattr(context, "id", "") or "")

        for entity_id in entity_ids:
            self.hass.async_create_task(
                self.async_record_habit_event(
                    domain=domain,
                    service=service,
                    entity_id=str(entity_id),
                    device_id=device_id,
                    source="call_service_event",
                    extra={"service_data": dict(service_data)},
                )
            )


def get_proactive_manager(hass: HomeAssistant) -> HAAIProactiveManager:
    """Get or create the proactive assistant manager."""
    manager = hass.data.get(_MANAGER_KEY)
    if manager is None:
        manager = HAAIProactiveManager(hass)
        hass.data[_MANAGER_KEY] = manager
    return manager
