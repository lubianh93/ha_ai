"""Speech to Text support for AI Hub using Silicon Flow ASR.

Features:
- High-precision recognition using SenseVoice model
- Automatic language detection (Chinese, English, Japanese, Korean)
- Support for multiple audio formats (WAV/MP3/FLAC/OGG/WebM)
- Dynamic timeout calculation based on audio size
- Retry mechanism with exponential backoff"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from homeassistant.components import stt
from homeassistant.components.stt import SpeechResultState, SpeechToTextEntity
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    DOMAIN,
    SILICONFLOW_ASR_URL,
    SILICONFLOW_STT_MODELS,
    STT_DEFAULT_MODEL,
    STT_MAX_AUDIO_SIZE,
    STT_MIN_AUDIO_SIZE,
    STT_WARNING_AUDIO_SIZE,
)
from .entity import AIHubEntityBase
from .markdown_filter import filter_markdown_content
from .utils.retry import RetryConfig, RetryError, async_retry

_LOGGER = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "pcm", "opus", "webm"]

# Retry configuration for STT API calls
STT_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    retryable_exceptions=(
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ConnectionError,
    ),
    retryable_status_codes=(408, 429, 500, 502, 503, 504),
)


def _calculate_dynamic_timeout(audio_size_bytes: int) -> aiohttp.ClientTimeout:
    """Calculate dynamic timeout based on audio size.

    Logic:
    - Base timeout: 15s (for short voice commands)
    - Add 3s per 100KB
    - Range: 15-60 seconds
    """
    audio_size_kb = audio_size_bytes / 1024
    base_timeout = 15
    timeout_per_100kb = 3
    calculated_timeout = base_timeout + (audio_size_kb / 100) * timeout_per_100kb
    total_timeout = min(max(calculated_timeout, 15), 60)
    connect_timeout = 5
    sock_read_timeout = max(total_timeout * 0.8, 10)

    _LOGGER.debug(
        "Dynamic timeout calculation: audio_size=%dKB, total=%.1fs, connect=%.1fs, read=%.1fs",
        int(audio_size_kb), total_timeout, connect_timeout, sock_read_timeout
    )

    return aiohttp.ClientTimeout(
        total=total_timeout,
        connect=connect_timeout,
        sock_read=sock_read_timeout
    )


def _create_wav_header(audio_data: bytes, metadata: stt.SpeechMetadata) -> bytes:
    """Create WAV header for raw PCM audio data."""
    sample_rate = metadata.sample_rate
    channels = metadata.channel
    bits_per_sample = metadata.bit_rate
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    wav_header = bytearray()
    # RIFF header
    wav_header.extend(b'RIFF')
    wav_header.extend((36 + len(audio_data)).to_bytes(4, 'little'))
    wav_header.extend(b'WAVE')
    # fmt chunk
    wav_header.extend(b'fmt ')
    wav_header.extend((16).to_bytes(4, 'little'))
    wav_header.extend((1).to_bytes(2, 'little'))  # PCM format
    wav_header.extend(channels.to_bytes(2, 'little'))
    wav_header.extend(sample_rate.to_bytes(4, 'little'))
    wav_header.extend(byte_rate.to_bytes(4, 'little'))
    wav_header.extend(block_align.to_bytes(2, 'little'))
    wav_header.extend(bits_per_sample.to_bytes(2, 'little'))
    # data chunk
    wav_header.extend(b'data')
    wav_header.extend(len(audio_data).to_bytes(4, 'little'))

    return bytes(wav_header) + audio_data


def _extract_transcription(response_data: dict[str, Any]) -> str | None:
    """Extract transcription text from API response."""
    # OpenAI-style response
    if "text" in response_data:
        return response_data["text"]
    if "transcription" in response_data:
        return response_data["transcription"]

    # Silicon Flow API format
    if "code" in response_data:
        code = response_data.get("code")
        if code != 20000:
            error_msg = response_data.get("message", "Unknown API error")
            _LOGGER.error("Silicon Flow API error: code=%s, message=%s", code, error_msg)
            raise HomeAssistantError(f"API error: {error_msg}")
        data = response_data.get("data")
        if data:
            return data.get("text") or data.get("transcription")

    # Result field format
    if "result" in response_data:
        result = response_data["result"]
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if isinstance(result, str):
            return result

    # Last resort: find any string field
    if isinstance(response_data, dict):
        for key, value in response_data.items():
            if isinstance(value, str) and len(value.strip()) > 0 and key not in ["message", "msg"]:
                return value

    return None


def _handle_timeout_error(exc: asyncio.TimeoutError) -> HomeAssistantError:
    """Convert timeout error to user-friendly HomeAssistantError."""
    _LOGGER.error("Silicon Flow ASR request timeout: %s", exc)
    exc_str = str(exc)

    if "SocketTimeoutError" in exc_str or "Timeout on reading data" in exc_str:
        _LOGGER.warning("SiliconFlow server response delayed, speech recognition timeout")
        return HomeAssistantError(
            "Speech recognition service is temporarily busy. Please try again."
        )
    if "Timeout on connect" in exc_str:
        return HomeAssistantError("Cannot connect to speech recognition service. Please check network.")

    return HomeAssistantError("Speech recognition timeout. Please try again.")


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [AIHubSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class AIHubSTTEntity(SpeechToTextEntity, AIHubEntityBase):
    """AI Hub speech-to-text entity using Silicon Flow ASR."""

    _attr_has_entity_name = False
    _attr_supported_options = ["model", "language"]

    def __init__(self, config_entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the STT entity."""
        super().__init__(config_entry, subentry, STT_DEFAULT_MODEL)
        self._attr_available = True
        self._hass: HomeAssistant | None = None

        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="AI Hub",
            model="Silicon Flow ASR",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        self._api_key = config_entry.data.get(CONF_API_KEY)

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self._hass = self.hass

    @property
    def options(self) -> dict[str, Any]:
        """Return the options for this entity."""
        return self.subentry.data

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options."""
        return {
            "model": STT_DEFAULT_MODEL,
            "language": "auto"  # HA 2025.12+ auto detection
        }

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return [
            "zh-CN", "zh-TW", "zh-HK", "en-US", "ja-JP", "ko-KR",
            "fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "ru-RU",
        ]

    @property
    def supported_formats(self) -> list[str]:
        """Return a list of supported audio formats."""
        return ["wav", "mp3", "opus", "webm"]

    @property
    def supported_codecs(self) -> list[str]:
        """Return a list of supported audio codecs."""
        return ["pcm", "mp3", "wav", "flac", "aac", "ogg"]

    @property
    def supported_sample_rates(self) -> list[int]:
        """Return a list of supported sample rates."""
        return [8000, 11025, 16000, 22050, 44100, 48000]

    @property
    def supported_bit_rates(self) -> list[int]:
        """Return a list of supported bit rates."""
        return [8, 16, 24, 32, 64, 128, 256, 320]

    @property
    def supported_channels(self) -> list[int]:
        """Return a list of supported audio channels."""
        return [1, 2]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream
    ) -> stt.SpeechResult:
        """Process an audio stream and return the transcription result."""
        _LOGGER.debug(
            "Starting STT processing: format=%s, sample_rate=%d, channel=%d",
            metadata.format, metadata.sample_rate, metadata.channel
        )

        self._validate_api_key()
        audio_data = await self._collect_audio_stream(stream)

        # Check for empty audio
        if len(audio_data) < STT_MIN_AUDIO_SIZE:
            _LOGGER.debug("Audio data too small: %d bytes, returning empty result", len(audio_data))
            return stt.SpeechResult("", SpeechResultState.SUCCESS)

        if len(audio_data) > STT_WARNING_AUDIO_SIZE:
            _LOGGER.warning("Voice assistant audio is quite large: %d bytes", len(audio_data))

        try:
            model = self._validate_model()
            audio_data = self._prepare_audio_data(audio_data, metadata)
            response_data = await self._send_asr_request(audio_data, model)
            return self._process_asr_response(response_data)

        except HomeAssistantError:
            raise
        except Exception as exc:
            _LOGGER.error("Silicon Flow ASR transcription failed: %s", exc, exc_info=True)
            raise HomeAssistantError(f"ASR transcription failed: {exc}") from exc

    def _validate_api_key(self) -> None:
        """Validate that API key is configured."""
        if not self._api_key:
            _LOGGER.error("Silicon Flow API key not configured")
            raise HomeAssistantError(
                "Silicon Flow API key not configured. Please add the API key in the integration settings."
            )

    async def _collect_audio_stream(self, stream) -> bytes:
        """Collect audio data from stream."""
        audio_data = b""
        chunk_count = 0
        async for chunk in stream:
            audio_data += chunk
            chunk_count += 1

        _LOGGER.debug("Audio data collected: chunks=%d, total_size=%d bytes", chunk_count, len(audio_data))
        return audio_data

    def _validate_model(self) -> str:
        """Validate and return the STT model."""
        model = self.options.get("model", STT_DEFAULT_MODEL)
        if model not in SILICONFLOW_STT_MODELS:
            raise HomeAssistantError(f"Unsupported model: {model}")
        _LOGGER.debug("Using STT model: %s", model)
        return model

    def _prepare_audio_data(self, audio_data: bytes, metadata: stt.SpeechMetadata) -> bytes:
        """Prepare audio data for API request."""
        # Convert to WAV if needed
        if len(audio_data) < 12 or audio_data[:4] != b'RIFF':
            _LOGGER.debug("Converting raw PCM data to WAV format")
            audio_data = _create_wav_header(audio_data, metadata)
            _LOGGER.debug("Created WAV header, total size: %d bytes", len(audio_data))
        else:
            _LOGGER.debug("Audio data already has WAV format")

        # Validate size
        if len(audio_data) > STT_MAX_AUDIO_SIZE:
            _LOGGER.error("Audio file too large: %d bytes (max: %d bytes)", len(audio_data), STT_MAX_AUDIO_SIZE)
            raise HomeAssistantError("Audio file too large. Please use a file smaller than 10MB.")

        # Validate format
        if metadata.format.lower() not in SUPPORTED_AUDIO_FORMATS:
            _LOGGER.error("Unsupported audio format: %s", metadata.format)
            raise HomeAssistantError(
                f"Unsupported audio format: {metadata.format}. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )

        return audio_data

    async def _send_asr_request(self, audio_data: bytes, model: str) -> dict[str, Any]:
        """Send audio data to Silicon Flow ASR API with retry logic."""
        timeout = _calculate_dynamic_timeout(len(audio_data))

        _LOGGER.debug("Sending request to Silicon Flow ASR: model=%s, size=%d bytes", model, len(audio_data))

        if len(audio_data) >= 8:
            _LOGGER.debug("Audio header: %s", audio_data[:8].hex())

        async def _make_request() -> dict[str, Any]:
            """Inner function for retry wrapper."""
            headers = {"Authorization": f"Bearer {self._api_key}"}

            # Create new FormData for each attempt (FormData can only be consumed once)
            form_data = aiohttp.FormData()
            form_data.add_field('model', model)
            form_data.add_field('file', audio_data, filename='recording.wav', content_type='audio/wav')

            session = async_get_clientsession(self._hass or self.hass)
            async with session.post(
                SILICONFLOW_ASR_URL,
                headers=headers,
                data=form_data,
                timeout=timeout
            ) as response:
                _LOGGER.debug("HTTP response: status=%d", response.status)

                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("HTTP error: %s - %s", response.status, error_text)
                    # Raise ClientResponseError for retry mechanism to handle
                    if response.status in STT_RETRY_CONFIG.retryable_status_codes:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=error_text,
                        )
                    raise HomeAssistantError(f"HTTP request failed: {response.status}")

                try:
                    response_data = await response.json()
                    _LOGGER.debug("Silicon Flow ASR response: %s", response_data)
                    return response_data
                except Exception as e:
                    _LOGGER.error("Failed to parse Silicon Flow ASR response: %s", e)
                    raise HomeAssistantError(f"Failed to parse response: {e}") from e

        try:
            return await async_retry(_make_request, config=STT_RETRY_CONFIG)
        except RetryError as exc:
            _LOGGER.error("Silicon Flow ASR failed after %d retries: %s", exc.attempts, exc.last_exception)
            if exc.last_exception:
                if isinstance(exc.last_exception, asyncio.TimeoutError):
                    raise _handle_timeout_error(exc.last_exception) from exc
                if isinstance(exc.last_exception, aiohttp.ClientConnectorError):
                    raise HomeAssistantError(
                        "Cannot connect to Silicon Flow server after retries. Please check network."
                    ) from exc
            raise HomeAssistantError("Speech recognition failed after multiple retries.") from exc
        except asyncio.TimeoutError as exc:
            raise _handle_timeout_error(exc) from exc
        except aiohttp.ClientConnectorError as exc:
            _LOGGER.error("Silicon Flow ASR connection failed: %s", exc)
            raise HomeAssistantError("Cannot connect to Silicon Flow server. Please check network.") from exc
        except aiohttp.ClientError as exc:
            _LOGGER.error("Silicon Flow ASR network error: %s", exc)
            raise HomeAssistantError("Speech recognition network request failed. Please try again.") from exc

    def _process_asr_response(self, response_data: dict[str, Any]) -> stt.SpeechResult:
        """Process ASR response and return SpeechResult."""
        transcribed_text = _extract_transcription(response_data)

        if transcribed_text is None:
            _LOGGER.error("Cannot extract transcription from response: %s", response_data)
            raise HomeAssistantError("API response format error. Cannot find transcription text.")

        if not transcribed_text.strip():
            _LOGGER.debug("STT returned empty text (user may not have spoken)")
            return stt.SpeechResult("", SpeechResultState.SUCCESS)

        _LOGGER.info("STT recognition successful: '%s'", transcribed_text)

        cleaned_text = filter_markdown_content(transcribed_text)
        return stt.SpeechResult(cleaned_text.strip(), stt.SpeechResultState.SUCCESS)
