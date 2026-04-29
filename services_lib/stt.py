"""STT service handlers for HA AI."""

from __future__ import annotations

import logging
import os

import aiohttp
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError

from ..const import (
    CONF_STT_FILE,
    DEFAULT_REQUEST_TIMEOUT,
    RECOMMENDED_STT_MODEL,
    STT_AUDIO_FORMATS,
    STT_MAX_FILE_SIZE_MB,
)

_LOGGER = logging.getLogger(__name__)


async def handle_stt_transcribe(
    hass: HomeAssistant,
    call: ServiceCall,
    api_key: str,
    stt_url: str,
) -> dict:
    """Handle an STT service call."""
    try:
        if not api_key or not api_key.strip():
            return {
                "success": False,
                "error": "API key is not configured. Please set a provider key or custom API key.",
            }
        if not stt_url or not str(stt_url).strip():
            return {"success": False, "error": "STT API URL is not configured"}

        audio_file = call.data[CONF_STT_FILE]
        model = call.data.get("model", RECOMMENDED_STT_MODEL)

        if not audio_file or not audio_file.strip():
            raise ServiceValidationError("Audio file path cannot be empty")
        if not isinstance(model, str) or not model.strip():
            raise ServiceValidationError("STT model is required")

        if not os.path.isabs(audio_file):
            audio_file = os.path.join(hass.config.config_dir, audio_file)
        if not os.path.exists(audio_file):
            raise ServiceValidationError(f"Audio file does not exist: {audio_file}")
        if os.path.isdir(audio_file):
            raise ServiceValidationError(f"Provided path is a directory: {audio_file}")

        file_size = os.path.getsize(audio_file)
        if file_size > STT_MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ServiceValidationError(
                f"Audio file is too large; maximum supported size is {STT_MAX_FILE_SIZE_MB}MB"
            )

        file_ext = os.path.splitext(audio_file)[1].lower().lstrip(".")
        if file_ext not in STT_AUDIO_FORMATS:
            raise ServiceValidationError(
                f"Unsupported audio format: {file_ext}. Supported formats: "
                f"{', '.join(STT_AUDIO_FORMATS)}"
            )

        with open(audio_file, "rb") as file_obj:
            audio_data = file_obj.read()

        headers = {"Authorization": f"Bearer {api_key}"}
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            audio_data,
            filename=os.path.basename(audio_file),
            content_type=f"audio/{file_ext}",
        )
        form_data.add_field("model", model.strip())

        timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT / 1000)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(stt_url, headers=headers, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("STT API error: %s - %s", response.status, error_text)
                    return {"success": False, "error": f"STT API request failed: {response.status}"}

                response_data = await response.json()

        text = response_data.get("text") or response_data.get("transcription")
        if not text and isinstance(response_data.get("data"), dict):
            text = response_data["data"].get("text") or response_data["data"].get("transcription")
        if not text:
            _LOGGER.error("STT API response format error: %s", response_data)
            return {"success": False, "error": "API response format error"}

        return {
            "success": True,
            "text": text,
            "model": model,
            "audio_file": audio_file,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }

    except ServiceValidationError as exc:
        _LOGGER.error("STT service validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        _LOGGER.error("STT service error: %s", exc, exc_info=True)
        return {"success": False, "error": f"STT transcription failed: {exc}"}
