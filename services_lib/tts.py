"""TTS services for HA AI."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import (
    ALIYUN_BAILIAN_TTS_PROVIDER,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_URL,
    DOMAIN,
    EDGE_TTS_VOICES,
    TTS_DEFAULT_VOICE,
)

_LOGGER = logging.getLogger(__name__)
_HTTP_TTS_LOCK = asyncio.Lock()


def _check_edge_tts():
    """Return edge_tts module if installed."""
    try:
        import edge_tts
        return edge_tts
    except ImportError:
        return None


async def _edge_tts_audio(text: str, voice: str) -> bytes:
    """Generate audio with Edge TTS."""
    edge_tts = _check_edge_tts()
    if not edge_tts:
        raise ServiceValidationError("edge_tts is not installed. Please install edge-tts.")
    if voice not in EDGE_TTS_VOICES:
        raise ServiceValidationError(f"Unsupported Edge TTS voice: {voice}")

    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


async def _http_tts_audio(
    hass: HomeAssistant,
    text: str,
    voice: str,
    api_key: str | None,
    tts_url: str | None,
    tts_model: str | None,
) -> bytes:
    """Generate audio with an OpenAI-compatible HTTP speech endpoint."""
    if not api_key or not api_key.strip():
        raise ServiceValidationError("TTS API key is not configured")

    endpoint = str(tts_url or DEFAULT_TTS_URL).strip()
    if not endpoint:
        raise ServiceValidationError("TTS API URL is not configured")

    payload = {
        "model": str(tts_model or "tts-1").strip(),
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    session = async_get_clientsession(hass)
    async with _HTTP_TTS_LOCK:
        async with session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ServiceValidationError(
                    f"TTS request failed: HTTP {response.status} {error_text}"
                )
            return await response.read()


def _decode_audio_value(value: Any) -> bytes:
    """Decode a possible base64 audio value from a realtime event."""
    if not isinstance(value, str) or not value.strip():
        return b""
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return b""


def _audio_bytes_from_event(event: dict[str, Any]) -> bytes:
    """Extract audio bytes from common OpenAI/DashScope realtime shapes."""
    audio = b""
    event_type = str(event.get("type") or "")

    for key in ("audio", "audio_data", "audio_chunk", "output_audio"):
        audio += _decode_audio_value(event.get(key))

    delta = event.get("delta")
    if isinstance(delta, str) and "audio" in event_type:
        audio += _decode_audio_value(delta)
    elif isinstance(delta, dict):
        audio += _audio_bytes_from_event(delta)

    for key in ("response", "item", "output", "data"):
        nested = event.get(key)
        if isinstance(nested, dict):
            audio += _audio_bytes_from_event(nested)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    audio += _audio_bytes_from_event(item)

    return audio


async def _aliyun_bailian_tts_audio(
    hass: HomeAssistant,
    text: str,
    voice: str,
    api_key: str | None,
    tts_url: str | None,
    tts_model: str | None,
) -> bytes:
    """Generate TTS audio through Alibaba Cloud Bailian.

    Realtime requests are serialized and use a small state machine that ignores
    late session events. If the URL is HTTP(S), the provider falls back to the
    OpenAI-compatible speech payload.
    """
    endpoint = str(tts_url or "").strip()
    if not endpoint:
        raise ServiceValidationError("Aliyun Bailian TTS URL is not configured")
    if not endpoint.startswith(("ws://", "wss://")):
        return await _http_tts_audio(hass, text, voice, api_key, endpoint, tts_model or "qwen-tts")
    if not api_key or not api_key.strip():
        raise ServiceValidationError("Aliyun Bailian API key is not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
        "X-DashScope-DataInspection": "enable",
    }
    model = str(tts_model or "qwen-tts-realtime").strip()
    session = async_get_clientsession(hass)

    async with _HTTP_TTS_LOCK:
        async with session.ws_connect(
            endpoint,
            headers=headers,
            timeout=90,
        ) as ws:
            await ws.send_json({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": voice,
                    "model": model,
                    "output_audio_format": "mp3",
                },
            })

            session_ready = False
            for _ in range(20):
                msg = await ws.receive(timeout=5)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = msg.json()
                    event_type = str(event.get("type") or "")
                    if event_type in ("session.updated", "session.created"):
                        session_ready = True
                        break
                    if "error" in event_type or event.get("error"):
                        raise ServiceValidationError(f"Aliyun Bailian TTS error: {event}")
                    _LOGGER.debug("Aliyun Bailian TTS setup event ignored: %s", event_type)
                    continue
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    raise ServiceValidationError("Aliyun Bailian TTS websocket closed during setup")
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise ServiceValidationError(f"Aliyun Bailian TTS websocket error: {ws.exception()}")

            if not session_ready:
                raise ServiceValidationError("Aliyun Bailian TTS session was not ready")

            await ws.send_json({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            })
            await ws.send_json({
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "voice": voice,
                    "output_audio_format": "mp3",
                },
            })

            audio_bytes = b""
            done_events = {
                "response.done",
                "response.audio.done",
                "response.output_audio.done",
                "session.closed",
            }
            while True:
                msg = await ws.receive(timeout=20)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = msg.json()
                    event_type = str(event.get("type") or "")
                    if "error" in event_type or event.get("error"):
                        raise ServiceValidationError(f"Aliyun Bailian TTS error: {event}")
                    audio_bytes += _audio_bytes_from_event(event)
                    if event_type in done_events:
                        break
                    if event_type in ("session.updated", "session.created"):
                        _LOGGER.debug("Aliyun Bailian TTS late session event ignored: %s", event_type)
                    continue
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise ServiceValidationError(f"Aliyun Bailian TTS websocket error: {ws.exception()}")

    if not audio_bytes:
        raise ServiceValidationError("No audio data generated by Aliyun Bailian TTS")
    return audio_bytes


async def _generate_audio(
    hass: HomeAssistant,
    text: str,
    voice: str,
    api_key: str | None,
    provider: str,
    tts_url: str | None,
    tts_model: str | None,
) -> bytes:
    """Generate audio using the configured provider."""
    if provider == DEFAULT_TTS_PROVIDER:
        audio_bytes = await _edge_tts_audio(text, voice)
    elif provider == ALIYUN_BAILIAN_TTS_PROVIDER:
        audio_bytes = await _aliyun_bailian_tts_audio(
            hass,
            text,
            voice,
            api_key,
            tts_url,
            tts_model,
        )
    else:
        audio_bytes = await _http_tts_audio(hass, text, voice, api_key, tts_url, tts_model)

    if not audio_bytes:
        raise ServiceValidationError("No audio data generated")
    return audio_bytes


async def handle_tts_speech(
    hass: HomeAssistant,
    call: ServiceCall,
    api_key: str | None = None,
    provider: str = DEFAULT_TTS_PROVIDER,
    tts_url: str | None = None,
    tts_model: str | None = None,
) -> dict:
    """Handle a TTS service call."""
    try:
        text = call.data["text"]
        voice = call.data.get("voice", TTS_DEFAULT_VOICE)
        media_player_entity = call.data.get("media_player_entity")

        if not text or not text.strip():
            raise ServiceValidationError("Text content cannot be empty")

        audio_bytes = await _generate_audio(
            hass,
            text,
            voice,
            api_key,
            provider,
            tts_url,
            tts_model,
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        if media_player_entity:
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file_path = temp_file.name

                await hass.services.async_call(
                    "media_player",
                    "play_media",
                    {
                        "entity_id": media_player_entity,
                        "media_content_id": f"file://{temp_file_path}",
                        "media_content_type": "audio/mpeg",
                    },
                    blocking=True,
                )

                await asyncio.sleep(1)
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

                return {
                    "success": True,
                    "message": "Speech playback succeeded",
                    "media_player": media_player_entity,
                    "voice": voice,
                    "provider": provider,
                }
            except Exception as exc:
                _LOGGER.error("Media playback failed: %s", exc)
                return {
                    "success": False,
                    "error": f"Media playback failed: {exc}",
                    "audio_data": audio_base64,
                    "provider": provider,
                }

        return {
            "success": True,
            "audio_data": audio_base64,
            "audio_format": "mp3",
            "voice": voice,
            "provider": provider,
        }

    except ServiceValidationError as exc:
        _LOGGER.error("TTS service validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        _LOGGER.error("TTS service error: %s", exc, exc_info=True)
        return {"success": False, "error": f"TTS generation failed: {exc}"}


async def handle_tts_stream(
    hass: HomeAssistant,
    call: ServiceCall,
    api_key: str | None = None,
    provider: str = DEFAULT_TTS_PROVIDER,
    tts_url: str | None = None,
    tts_model: str | None = None,
) -> dict:
    """Handle a streaming TTS service call through the event bus."""
    try:
        text = call.data["text"]
        voice = call.data.get("voice", TTS_DEFAULT_VOICE)
        chunk_size = call.data.get("chunk_size", 4096)

        if not text or not text.strip():
            raise ServiceValidationError("Text content cannot be empty")

        audio_bytes = await _generate_audio(
            hass,
            text,
            voice,
            api_key,
            provider,
            tts_url,
            tts_model,
        )

        total_chunks = 0
        total_bytes = 0
        for index, start in enumerate(range(0, len(audio_bytes), chunk_size), start=1):
            chunk = audio_bytes[start:start + chunk_size]
            total_chunks = index
            total_bytes += len(chunk)
            hass.bus.async_fire(
                f"{DOMAIN}_tts_stream_chunk",
                {
                    "voice": voice,
                    "provider": provider,
                    "chunk_index": index,
                    "chunk_size": len(chunk),
                    "total_bytes": total_bytes,
                    "audio_chunk": base64.b64encode(chunk).decode("utf-8"),
                    "content_type": "audio/mpeg",
                },
            )

        hass.bus.async_fire(
            f"{DOMAIN}_tts_stream_complete",
            {
                "voice": voice,
                "provider": provider,
                "total_chunks": total_chunks,
                "total_bytes": total_bytes,
                "text": text,
            },
        )

        return {
            "success": True,
            "method": "stream",
            "voice": voice,
            "provider": provider,
            "total_chunks": total_chunks,
            "total_bytes": total_bytes,
            "message": "Audio stream was pushed through the event bus",
        }

    except ServiceValidationError as exc:
        _LOGGER.error("Streaming TTS validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        _LOGGER.error("Streaming TTS error: %s", exc, exc_info=True)
        return {"success": False, "error": f"Streaming TTS generation failed: {exc}"}
