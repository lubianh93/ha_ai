"""Services for the HA AI integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall

from .config_resolver import resolve_entry_config
from .const import (
    CONF_CHAT_URL,
    CONF_IMAGE_URL,
    CONF_TTS_MODEL,
    CONF_TTS_PROVIDER,
    CONF_TTS_URL,
    CONF_STT_URL,
    CONF_SUBENTRY_ID,
    DEFAULT_CHAT_URL,
    DEFAULT_IMAGE_URL,
    DEFAULT_STT_URL,
    DEFAULT_TTS_URL,
    DEFAULT_TTS_PROVIDER,
    DOMAIN,
    SERVICE_FOLLOW_UP_PLAYBACK_DONE,
    SERVICE_GET_PROACTIVE_STATUS,
    SERVICE_ANALYZE_IMAGE,
    SERVICE_GENERATE_IMAGE,
    SERVICE_RECORD_HABIT_EVENT,
    SERVICE_STT_TRANSCRIBE,
    SERVICE_TTS_SAY,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)
from .services_lib import (
    FOLLOW_UP_PLAYBACK_DONE_SCHEMA,
    GET_PROACTIVE_STATUS_SCHEMA,
    IMAGE_ANALYZER_SCHEMA,
    IMAGE_GENERATOR_SCHEMA,
    RECORD_HABIT_EVENT_SCHEMA,
    STT_SCHEMA,
    TTS_SCHEMA,
    handle_analyze_image,
    handle_generate_image,
    handle_stt_transcribe,
    handle_tts_speech,
    handle_tts_stream,
)
from .proactive import get_proactive_manager

_LOGGER = logging.getLogger(__name__)

_REGISTERED_HASS: HomeAssistant | None = None
_REGISTERED_ENTRY = None
_SERVICE_CONTEXTS_KEY = f"{DOMAIN}_service_contexts"
_SERVICES_REGISTERED_KEY = f"{DOMAIN}_services_registered"


def _has_configured_api_key(api_key: Any) -> bool:
    return isinstance(api_key, str) and bool(api_key.strip())


def _register_service(
    hass: HomeAssistant,
    service_name: str,
    handler,
    schema: dict[str, Any],
) -> None:
    hass.services.async_register(
        DOMAIN,
        service_name,
        handler,
        schema=vol.Schema(schema),
        supports_response=True,
    )


def _get_service_contexts(hass: HomeAssistant) -> dict[str, object]:
    return hass.data.setdefault(_SERVICE_CONTEXTS_KEY, {})


def _entry_has_subentry_type(
    config_entry: Any,
    subentry_type: str,
    subentry_id: str | None = None,
) -> bool:
    return any(
        subentry.subentry_type == subentry_type
        and (not subentry_id or subentry.subentry_id == subentry_id)
        for subentry in getattr(config_entry, "subentries", {}).values()
    )


def _resolve_service_entry(
    call: ServiceCall,
    subentry_type: str | None = None,
) -> tuple[HomeAssistant | None, Any | None, str | None]:
    hass = _REGISTERED_HASS
    fallback_entry = _REGISTERED_ENTRY

    if hass is None:
        return None, None, "HA AI 未初始化"

    contexts = _get_service_contexts(hass)
    explicit_entry_id = call.data.get("config_entry_id")
    explicit_subentry_id = call.data.get(CONF_SUBENTRY_ID)

    if explicit_entry_id:
        config_entry = contexts.get(explicit_entry_id)
        if config_entry is None:
            return hass, None, f"未找到配置项: {explicit_entry_id}"
        if subentry_type and not _entry_has_subentry_type(
            config_entry,
            subentry_type,
            explicit_subentry_id,
        ):
            return hass, None, f"指定配置项不包含 {subentry_type} 子项"
        return hass, config_entry, None

    candidates = list(contexts.values())
    if subentry_type is not None:
        candidates = [
            entry for entry in candidates
            if _entry_has_subentry_type(entry, subentry_type, explicit_subentry_id)
        ]

    if not candidates:
        if fallback_entry is not None and (
            subentry_type is None
            or _entry_has_subentry_type(fallback_entry, subentry_type, explicit_subentry_id)
        ):
            return hass, fallback_entry, None
        return hass, None, "没有可用配置"

    if len(candidates) > 1:
        return hass, None, "检测到多个配置，请在服务中指定 config_entry_id"

    return hass, candidates[0], None


def _resolve_service_config(
    call: ServiceCall,
    subentry_type: str,
    *values: tuple[str, Any],
) -> tuple[HomeAssistant | None, Any | None, str | None]:
    hass, config_entry, error = _resolve_service_entry(call, subentry_type)
    if error is not None or hass is None or config_entry is None:
        return hass, None, error or "API密钥未配置"

    return hass, resolve_entry_config(
        config_entry,
        subentry_type,
        *values,
        subentry_id=call.data.get(CONF_SUBENTRY_ID),
    ), None


async def _handle_service_with_api_key(
    call: ServiceCall,
    subentry_type: str,
    *values: tuple[str, Any],
    config_mapper,
    handler,
) -> dict:
    hass, config, error = _resolve_service_config(call, subentry_type, *values)
    if error is not None or hass is None or config is None:
        return {"success": False, "error": error or "API密钥未配置"}

    handler_args = config_mapper(config)
    effective_key = handler_args[0]
    if not _has_configured_api_key(effective_key):
        return {"success": False, "error": "API密钥未配置"}

    return await handler(hass, call, *handler_args)


async def _handle_analyze_image(call: ServiceCall) -> dict:
    return await _handle_service_with_api_key(
        call,
        SUBENTRY_CONVERSATION,
        (CONF_CHAT_URL, DEFAULT_CHAT_URL),
        config_mapper=lambda config: (config[1], config[0]),
        handler=handle_analyze_image,
    )


async def _handle_generate_image(call: ServiceCall) -> dict:
    return await _handle_service_with_api_key(
        call,
        SUBENTRY_AI_TASK,
        (CONF_IMAGE_URL, DEFAULT_IMAGE_URL),
        config_mapper=lambda config: (config[1], config[0]),
        handler=handle_generate_image,
    )


async def _handle_tts_say(call: ServiceCall) -> dict:
    stream = call.data.get("stream", False)
    hass, config, error = _resolve_service_config(
        call,
        SUBENTRY_TTS,
        (CONF_TTS_PROVIDER, DEFAULT_TTS_PROVIDER),
        (CONF_TTS_URL, DEFAULT_TTS_URL),
        (CONF_TTS_MODEL, ""),
    )
    if error is not None or hass is None or config is None:
        return {"success": False, "error": error or "TTS configuration is not available"}

    provider, tts_url, tts_model, effective_key = config
    handler = handle_tts_stream if stream else handle_tts_speech
    return await handler(hass, call, effective_key, provider, tts_url, tts_model)


async def _handle_stt_transcribe(call: ServiceCall) -> dict:
    return await _handle_service_with_api_key(
        call,
        SUBENTRY_STT,
        (CONF_STT_URL, DEFAULT_STT_URL),
        config_mapper=lambda config: (config[1], config[0]),
        handler=handle_stt_transcribe,
    )


async def _handle_follow_up_playback_done(call: ServiceCall) -> dict:
    hass = _REGISTERED_HASS
    if hass is None:
        return {"success": False, "error": "HA AI is not initialized"}
    return await get_proactive_manager(hass).async_handle_playback_done(
        pending_id=call.data.get("pending_id"),
        device_id=call.data.get("device_id"),
        conversation_id=call.data.get("conversation_id"),
    )


async def _handle_record_habit_event(call: ServiceCall) -> dict:
    hass = _REGISTERED_HASS
    if hass is None:
        return {"success": False, "error": "HA AI is not initialized"}
    return await get_proactive_manager(hass).async_record_habit_event(
        domain=str(call.data["domain"]),
        service=str(call.data["service"]),
        entity_id=str(call.data["entity_id"]),
        device_id=call.data.get("device_id"),
        source=str(call.data.get("source", "manual")),
    )


async def _handle_get_proactive_status(call: ServiceCall) -> dict:
    hass = _REGISTERED_HASS
    if hass is None:
        return {"success": False, "error": "HA AI is not initialized"}
    return {"success": True, **await get_proactive_manager(hass).async_status()}


async def async_setup_services(hass: HomeAssistant, config_entry) -> None:
    contexts = _get_service_contexts(hass)
    contexts[config_entry.entry_id] = config_entry

    global _REGISTERED_HASS, _REGISTERED_ENTRY
    _REGISTERED_HASS = hass
    _REGISTERED_ENTRY = next(reversed(contexts.values()), None)

    if hass.data.get(_SERVICES_REGISTERED_KEY):
        _LOGGER.debug("services already registered; updated context only")
        return

    _register_service(hass, SERVICE_ANALYZE_IMAGE, _handle_analyze_image, IMAGE_ANALYZER_SCHEMA)
    _register_service(hass, SERVICE_GENERATE_IMAGE, _handle_generate_image, IMAGE_GENERATOR_SCHEMA)
    _register_service(hass, SERVICE_TTS_SAY, _handle_tts_say, TTS_SCHEMA)
    _register_service(hass, SERVICE_STT_TRANSCRIBE, _handle_stt_transcribe, STT_SCHEMA)
    _register_service(
        hass,
        SERVICE_FOLLOW_UP_PLAYBACK_DONE,
        _handle_follow_up_playback_done,
        FOLLOW_UP_PLAYBACK_DONE_SCHEMA,
    )
    _register_service(
        hass,
        SERVICE_RECORD_HABIT_EVENT,
        _handle_record_habit_event,
        RECORD_HABIT_EVENT_SCHEMA,
    )
    _register_service(
        hass,
        SERVICE_GET_PROACTIVE_STATUS,
        _handle_get_proactive_status,
        GET_PROACTIVE_STATUS_SCHEMA,
    )

    hass.data[_SERVICES_REGISTERED_KEY] = True
    _LOGGER.info("HA AI services registered successfully")


async def async_unload_services(hass: HomeAssistant, entry_id: str | None = None) -> None:
    global _REGISTERED_HASS, _REGISTERED_ENTRY

    contexts = _get_service_contexts(hass)
    if entry_id is not None:
        contexts.pop(entry_id, None)
        _REGISTERED_ENTRY = next(reversed(contexts.values()), None) if contexts else None
        if contexts:
            return

    if not hass.data.get(_SERVICES_REGISTERED_KEY):
        return

    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_IMAGE)
    hass.services.async_remove(DOMAIN, SERVICE_GENERATE_IMAGE)
    hass.services.async_remove(DOMAIN, SERVICE_TTS_SAY)
    hass.services.async_remove(DOMAIN, SERVICE_STT_TRANSCRIBE)
    hass.services.async_remove(DOMAIN, SERVICE_FOLLOW_UP_PLAYBACK_DONE)
    hass.services.async_remove(DOMAIN, SERVICE_RECORD_HABIT_EVENT)
    hass.services.async_remove(DOMAIN, SERVICE_GET_PROACTIVE_STATUS)

    if _REGISTERED_HASS is hass:
        _REGISTERED_HASS = None
        _REGISTERED_ENTRY = None

    hass.data.pop(_SERVICES_REGISTERED_KEY, None)
    hass.data.pop(_SERVICE_CONTEXTS_KEY, None)
