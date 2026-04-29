"""Text to speech support for AI Hub using Edge TTS.

Features:
- Caching support for frequently used phrases
- Prosody parameter support (pitch, rate, volume)
- Quality parameter support (HA 2025.10+)
- Streaming input support
- 400+ voice models from Microsoft Edge TTS"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from propcache.api import cached_property

try:
    import edge_tts
    import edge_tts.exceptions
except ImportError:
    try:
        import edgeTTS  # noqa: F401 - intentional check for wrong package
        raise Exception('Please uninstall edgeTTS and install edge_tts instead.')
    except ImportError:
        raise Exception('edge_tts is required. Please install edge_tts.')

from .const import (
    ALIYUN_BAILIAN_TTS_PROVIDER,
    CONF_TTS_LANG,
    CONF_TTS_MODEL,
    CONF_TTS_PROVIDER,
    CONF_TTS_URL,
    CONF_TTS_VOICE,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_URL,
    DOMAIN,
    EDGE_TTS_VOICES,
    TTS_CACHE_MAX_SIZE,
    TTS_CACHE_TTL,
    TTS_DEFAULT_LANG,
    TTS_DEFAULT_VOICE,
    TTS_DEFAULT_VOICES,
)
from .entity import AIHubEntityBase
from .services_lib.tts import _aliyun_bailian_tts_audio
from .utils.tts_cache import TTSCache

# Create supported languages dynamically
SUPPORTED_LANGUAGES = {
    **dict(zip(EDGE_TTS_VOICES.values(), EDGE_TTS_VOICES.keys())),
    'zh-CN': 'zh-CN-XiaoxiaoNeural',
}

_LOGGER = logging.getLogger(__name__)

# Prosody options and quality (HA 2025.10+)
PROSODY_OPTIONS = ['pitch', 'rate', 'volume', 'quality']


def _get_tts_cache(hass: HomeAssistant) -> TTSCache:
    """Get or create TTS cache instance from hass.data."""
    from . import get_or_create_ai_hub_data

    ai_hub_data = get_or_create_ai_hub_data(hass)
    if ai_hub_data.tts_cache is None:
        ai_hub_data.tts_cache = TTSCache(
            max_size=TTS_CACHE_MAX_SIZE,
            ttl_seconds=TTS_CACHE_TTL
        )
    return ai_hub_data.tts_cache


def _generate_cache_key(
    message: str, voice: str, pitch: str, rate: str, volume: str
) -> str:
    """Generate cache key for TTS request."""
    key_data = f"{message}|{voice}|{pitch}|{rate}|{volume}"
    return hashlib.md5(key_data.encode()).hexdigest()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        async_add_entities(
            [AIHubTTSEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class AIHubTTSEntity(TextToSpeechEntity, AIHubEntityBase):
    """AI Hub text-to-speech entity using Edge TTS."""

    _attr_has_entity_name = False
    _attr_supported_options = ['voice', 'quality'] + PROSODY_OPTIONS
    _attr_supports_streaming_input = True

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the TTS entity."""
        super().__init__(config_entry, subentry, TTS_DEFAULT_VOICE)
        self._attr_available = True
        self._request_lock = asyncio.Lock()

        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="老王杂谈说",
            model=self._get_provider_name(),
            sw_version=edge_tts.__version__,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def options(self) -> dict[str, Any]:
        """Return the options for this entity."""
        return self.subentry.data

    def _get_provider_name(self) -> str:
        """Return configured TTS provider."""
        provider = self.subentry.data.get(CONF_TTS_PROVIDER, DEFAULT_TTS_PROVIDER)
        return str(provider or DEFAULT_TTS_PROVIDER).strip() or DEFAULT_TTS_PROVIDER

    @cached_property
    def default_options(self) -> dict[str, Any]:
        """Return default options."""
        return {
            ATTR_VOICE: TTS_DEFAULT_VOICE,
            'quality': 'default',
            'pitch': '+0Hz',
            'rate': '+0%',
            'volume': '+0%',
        }

    @property
    def default_language(self) -> str:
        """Return the default language from configured voice."""
        if CONF_TTS_LANG in self.subentry.data:
            return self.subentry.data[CONF_TTS_LANG]
        voice = self.subentry.data.get(CONF_TTS_VOICE, TTS_DEFAULT_VOICE)
        return EDGE_TTS_VOICES.get(voice, TTS_DEFAULT_LANG)

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return list([*SUPPORTED_LANGUAGES.keys(), *EDGE_TTS_VOICES.keys()])

    @property
    def _supported_voices(self) -> list[Voice]:
        """Return supported voices."""
        return [Voice(voice_id, voice_id) for voice_id in EDGE_TTS_VOICES.keys()]

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return a list of supported voices for a language."""
        if language is None:
            return self._supported_voices
        return [
            Voice(voice_id, voice_id)
            for voice_id, voice_lang in EDGE_TTS_VOICES.items()
            if voice_lang == language
        ]

    def _get_default_voice_for_language(self, language: str) -> str:
        """Get default voice for a language from centralized config."""
        return TTS_DEFAULT_VOICES.get(language, TTS_DEFAULT_VOICE)

    def _resolve_voice(self, language: str, options: dict[str, Any] | None) -> str:
        """Resolve voice from options, config, or language default."""
        voice = None
        if options and options.get('voice'):
            voice = options['voice']
        else:
            voice = self.subentry.data.get(CONF_TTS_VOICE, TTS_DEFAULT_VOICE)

        if self._get_provider_name() != DEFAULT_TTS_PROVIDER:
            return str(voice or TTS_DEFAULT_VOICE).strip() or TTS_DEFAULT_VOICE

        # If voice is a language code, convert to corresponding default voice
        if voice in SUPPORTED_LANGUAGES:
            voice = SUPPORTED_LANGUAGES[voice]

        if voice not in EDGE_TTS_VOICES:
            voice = self._get_default_voice_for_language(language)
            if voice not in EDGE_TTS_VOICES:
                voice = TTS_DEFAULT_VOICE

        return voice

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> TtsAudioType:
        """Load TTS audio."""
        return "mp3", await self._process_tts_audio(message, language, options or {})

    async def _process_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> bytes:
        """Process TTS with prosody support and caching."""
        if not message or not message.strip():
            raise HomeAssistantError("Text content cannot be empty")

        voice = self._resolve_voice(language, options)

        # Prosody parameters
        pitch = options.get('pitch', '+0Hz')
        rate = options.get('rate', '+0%')
        volume = options.get('volume', '+0%')

        # Check cache first
        cache = _get_tts_cache(self.hass)
        cache_key = _generate_cache_key(message, voice, pitch, rate, volume)
        cached_audio = cache.get(cache_key)
        if cached_audio is not None:
            _LOGGER.debug("TTS cache hit for message: %s", message[:50])
            return cached_audio

        _LOGGER.debug(
            'TTS: message="%s", voice="%s", pitch=%s, rate=%s, volume=%s',
            message[:50], voice, pitch, rate, volume
        )

        provider = self._get_provider_name()
        if provider != DEFAULT_TTS_PROVIDER:
            async with self._request_lock:
                audio_bytes = await self._process_remote_tts(
                    message,
                    voice,
                    options,
                )
            cache.set(cache_key, audio_bytes)
            return audio_bytes

        start_time = time.perf_counter()

        try:
            communicate = edge_tts.Communicate(
                text=message,
                voice=voice,
                pitch=pitch,
                rate=rate,
                volume=volume,
            )

            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _LOGGER.debug(
                "TTS generation time: %.1fms, audio size: %d bytes",
                elapsed_ms, len(audio_bytes)
            )

            if not audio_bytes:
                raise HomeAssistantError("No audio data generated")

            # Store in cache
            cache.set(cache_key, audio_bytes)
            _LOGGER.debug("TTS audio cached for message: %s", message[:50])

            return audio_bytes

        except edge_tts.exceptions.NoAudioReceived as exc:
            _LOGGER.warning("Edge TTS received no audio: %s", message[:50])
            raise HomeAssistantError(f"TTS received no audio: {message[:50]}") from exc
        except Exception as exc:
            _LOGGER.error("Edge TTS generation failed: %s", exc)
            # Retry with default voice
            if voice != TTS_DEFAULT_VOICE:
                _LOGGER.warning("Attempting retry with default voice...")
                try:
                    communicate = edge_tts.Communicate(text=message, voice=TTS_DEFAULT_VOICE)
                    audio_bytes = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_bytes += chunk["data"]
                    if audio_bytes:
                        _LOGGER.info("Default voice retry successful")
                        # Cache the retry result as well
                        retry_cache_key = _generate_cache_key(
                            message, TTS_DEFAULT_VOICE, pitch, rate, volume
                        )
                        cache.set(retry_cache_key, audio_bytes)
                        return audio_bytes
                except Exception as retry_exc:
                    _LOGGER.error("Default voice retry failed: %s", retry_exc)
            raise HomeAssistantError(f"TTS generation failed: {exc}") from exc

    async def _process_openai_compatible_tts(
        self,
        message: str,
        voice: str,
        options: dict[str, Any],
    ) -> bytes:
        """Process TTS through an OpenAI-compatible HTTP speech endpoint.

        This intentionally avoids realtime websocket/session APIs. Some Qwen
        realtime TTS integrations can emit session.updated after the client has
        already started finishing a request; serial HTTP requests are less
        interactive but much more predictable for Home Assistant TTS bursts.
        """
        if not self._api_key:
            raise HomeAssistantError("TTS API key is not configured")

        tts_url = str(
            self.subentry.data.get(CONF_TTS_URL)
            or DEFAULT_TTS_URL
        ).strip()
        model = str(self.subentry.data.get(CONF_TTS_MODEL) or "tts-1").strip()

        payload: dict[str, Any] = {
            "model": model,
            "input": message,
            "voice": voice,
            "response_format": options.get("response_format", "mp3"),
        }
        speed = options.get("speed")
        if speed is not None:
            payload["speed"] = speed

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        session = async_get_clientsession(self.hass)
        async with session.post(
            tts_url,
            headers=headers,
            json=payload,
            timeout=60,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HomeAssistantError(
                    f"TTS request failed: HTTP {response.status} {error_text}"
                )
            audio_bytes = await response.read()

        if not audio_bytes:
            raise HomeAssistantError("No audio data generated")
        return audio_bytes

    async def _process_remote_tts(
        self,
        message: str,
        voice: str,
        options: dict[str, Any],
    ) -> bytes:
        """Process non-Edge TTS providers through a serialized request."""
        provider = self._get_provider_name()
        if provider == ALIYUN_BAILIAN_TTS_PROVIDER:
            return await _aliyun_bailian_tts_audio(
                self.hass,
                message,
                voice,
                self._api_key,
                self.subentry.data.get(CONF_TTS_URL),
                self.subentry.data.get(CONF_TTS_MODEL),
            )
        return await self._process_openai_compatible_tts(message, voice, options)

    # ========== 流式输出支持 ==========

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        """Stream TTS audio - Home Assistant 2024.1+ support."""
        return TTSAudioResponse("mp3", self._stream_tts_audio(request))

    async def _stream_tts_audio(self, request: TTSAudioRequest) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio generation."""
        _LOGGER.debug("Starting streaming TTS, options: %s", request.options)

        if self._get_provider_name() != DEFAULT_TTS_PROVIDER:
            buffer = ""
            async for message in request.message_gen:
                buffer += message
            if buffer.strip():
                async with self._request_lock:
                    yield await self._process_remote_tts(
                        buffer.strip(),
                        self._resolve_voice(request.language, request.options),
                        request.options or {},
                    )
            return

        separators = "\n。.，,；;！!？?、"
        buffer = ""
        count = 0

        async for message in request.message_gen:
            _LOGGER.debug("Streaming TTS received: %s", message)
            count += 1
            min_len = 2 ** count * 10

            for char in message:
                buffer += char
                msg = buffer.strip()
                if len(msg) >= min_len and char in separators:
                    audio = await self._generate_stream_audio(msg, request.language, request.options)
                    if audio:
                        yield audio
                    buffer = ""

        if msg := buffer.strip():
            audio = await self._generate_stream_audio(msg, request.language, request.options)
            if audio:
                yield audio

    async def _generate_stream_audio(
        self, message: str, language: str, options: dict[str, Any] | None
    ) -> bytes | None:
        """Generate audio for streaming."""
        if not message:
            return None

        voice = self._resolve_voice(language, options)
        pitch = options.get('pitch', '+0Hz') if options else '+0Hz'
        rate = options.get('rate', '+0%') if options else '+0%'
        volume = options.get('volume', '+0%') if options else '+0%'

        try:
            communicate = edge_tts.Communicate(
                text=message,
                voice=voice,
                pitch=pitch,
                rate=rate,
                volume=volume,
            )

            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            return audio_bytes if audio_bytes else None

        except edge_tts.exceptions.NoAudioReceived:
            _LOGGER.warning("Streaming TTS received no audio: %s", message[:30])
            return None
        except Exception as exc:
            _LOGGER.error("Streaming TTS failed: %s", exc)
            return None
