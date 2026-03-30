"""Services for AI Hub integration - 精简版，实际处理委托给 services_lib 模块."""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall

from .const import (
    CONF_API_KEY,
    CONF_CHAT_URL,
    CONF_CUSTOM_API_KEY,
    CONF_IMAGE_URL,
    DOMAIN,
    SERVICE_ANALYZE_IMAGE,
    SERVICE_GENERATE_IMAGE,
    SERVICE_STT_TRANSCRIBE,

    SERVICE_TTS_SAY,
)
from .services_lib import (

    # Schemas
    IMAGE_ANALYZER_SCHEMA,
    IMAGE_GENERATOR_SCHEMA,
    STT_SCHEMA,

    TTS_SCHEMA,

    # Handlers
    handle_analyze_image,
    handle_generate_image,
    handle_stt_transcribe,
    handle_tts_speech,
    handle_tts_stream,
)

_LOGGER = logging.getLogger(__name__)


def _get_conversation_config(config_entry) -> tuple[str, str]:
    """Get chat URL, model and API key from Conversation Agent subentry."""
    from .const import AI_HUB_CHAT_URL
    chat_url = AI_HUB_CHAT_URL    
    custom_api_key = ""

    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == "conversation":
            chat_url = subentry.data.get(CONF_CHAT_URL, chat_url)
           
            custom_api_key = subentry.data.get(CONF_CUSTOM_API_KEY, "").strip()
            break

    # Use custom key if provided, otherwise use main key
    api_key = custom_api_key if custom_api_key else config_entry.runtime_data
    return chat_url, api_key


def _get_image_config(config_entry) -> tuple[str, str]:
    """Get image URL and API key from AI Task subentry."""
    from .const import AI_HUB_IMAGE_GEN_URL
    image_url = AI_HUB_IMAGE_GEN_URL
    custom_api_key = ""

    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == "ai_task_data":
            image_url = subentry.data.get(CONF_IMAGE_URL, image_url)
            custom_api_key = subentry.data.get(CONF_CUSTOM_API_KEY, "").strip()
            break

    # Use custom key if provided, otherwise use main key
    api_key = custom_api_key if custom_api_key else config_entry.runtime_data
    return image_url, api_key


async def async_setup_services(hass: HomeAssistant, config_entry) -> None:
    """Set up services for AI Hub integration."""

    api_key = config_entry.runtime_data
    def has_api_key() -> bool:
        """Check if any API key is available (main or custom)."""
        return api_key is not None and api_key.strip() != ""

    # ========== 图像分析服务 ==========
    async def _handle_analyze_image(call: ServiceCall) -> dict:
        chat_url, effective_key = _get_conversation_config(config_entry)
        if not effective_key or not effective_key.strip():
            return {"success": False, "error": "API密钥未配置"}
        return await handle_analyze_image(hass, call, effective_key, chat_url)

    # ========== 图像生成服务 ==========
    async def _handle_generate_image(call: ServiceCall) -> dict:
        image_url, effective_key = _get_image_config(config_entry)
        if not effective_key or not effective_key.strip():
            return {"success": False, "error": "API密钥未配置"}
        return await handle_generate_image(hass, call, effective_key, image_url)

    # ========== TTS 语音合成服务（统一） ==========
    async def _handle_tts_say(call: ServiceCall) -> dict:
        """Handle TTS service with optional streaming support."""
        stream = call.data.get("stream", False)
        if stream:
            return await handle_tts_stream(hass, call)
        else:
            if not has_api_key():
                return {"success": False, "error": "API密钥未配置"}
            return await handle_tts_speech(hass, call, api_key)

    # ========== STT 语音转文字服务 ==========
    async def _handle_stt_transcribe(call: ServiceCall) -> dict:
        api_key = config_entry.data.get(CONF_API_KEY) if hasattr(config_entry, 'data') else None
        if not api_key or not api_key.strip():
            return {"success": False, "error": "API密钥未配置"}
        return await handle_stt_transcribe(hass, call, api_key)


    # ========== 注册所有服务 ==========
    hass.services.async_register(
        DOMAIN, SERVICE_ANALYZE_IMAGE, _handle_analyze_image,
        schema=vol.Schema(IMAGE_ANALYZER_SCHEMA), supports_response=True
    )

    hass.services.async_register(
        DOMAIN, SERVICE_GENERATE_IMAGE, _handle_generate_image,
        schema=vol.Schema(IMAGE_GENERATOR_SCHEMA), supports_response=True
    )

    hass.services.async_register(
        DOMAIN, SERVICE_TTS_SAY, _handle_tts_say,
        schema=vol.Schema(TTS_SCHEMA), supports_response=True
    )

    hass.services.async_register(
        DOMAIN, SERVICE_STT_TRANSCRIBE, _handle_stt_transcribe,
        schema=vol.Schema(STT_SCHEMA), supports_response=True
    )


    _LOGGER.info("HA AI services registered successfully")


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload all services for HA AI integration.

    Args:
        hass: Home Assistant instance
    """
    hass.services.async_remove(DOMAIN, SERVICE_ANALYZE_IMAGE)
    hass.services.async_remove(DOMAIN, SERVICE_GENERATE_IMAGE)
    hass.services.async_remove(DOMAIN, SERVICE_TTS_SAY)
    hass.services.async_remove(DOMAIN, SERVICE_STT_TRANSCRIBE)

    _LOGGER.info("HA AI services unloaded successfully")
