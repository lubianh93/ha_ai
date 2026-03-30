"""TTS services for AI Hub - Edge TTS 语音合成功能.

本模块提供基于 Edge TTS 的文本转语音服务：
- handle_tts_speech: 生成语音并可选播放到媒体播放器
- handle_tts_stream: 流式生成语音，通过事件总线推送音频块

Edge TTS 特性:
- 微软免费 TTS 服务，无需 API Key
- 支持多种语音和语言
- 流式输出，适合长文本
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError

from ..const import (
    DOMAIN,
    EDGE_TTS_VOICES,
    TTS_DEFAULT_VOICE,
)

_LOGGER = logging.getLogger(__name__)


def _check_edge_tts():
    """检查 edge_tts 是否已安装."""
    try:
        import edge_tts
        return edge_tts
    except ImportError:
        return None


async def handle_tts_speech(
    hass: HomeAssistant,
    call: ServiceCall,
    api_key: str | None = None  # 保留参数兼容性，但不使用
) -> dict:
    """Handle Edge TTS service call - 生成语音."""
    try:
        edge_tts = _check_edge_tts()
        if not edge_tts:
            return {
                "success": False,
                "error": "edge_tts 库未安装，请先安装: pip install edge-tts"
            }

        text = call.data["text"]
        voice = call.data.get("voice", TTS_DEFAULT_VOICE)
        media_player_entity = call.data.get("media_player_entity")

        if not text or not text.strip():
            raise ServiceValidationError("文本内容不能为空")

        if voice not in EDGE_TTS_VOICES:
            raise ServiceValidationError(f"不支持的语音类型: {voice}")

        _LOGGER.debug("Edge TTS: text='%s', voice='%s'", text[:50], voice)

        # 使用 Edge TTS 生成音频
        communicate = edge_tts.Communicate(text=text, voice=voice)

        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        if not audio_bytes:
            return {"success": False, "error": "未生成音频数据"}

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # 如果指定了媒体播放器，直接播放
        if media_player_entity:
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file_path = temp_file.name

                await hass.services.async_call(
                    "media_player", "play_media",
                    {
                        "entity_id": media_player_entity,
                        "media_content_id": f"file://{temp_file_path}",
                        "media_content_type": "audio/mpeg",
                    },
                    blocking=True,
                )

                # 延迟删除临时文件
                await asyncio.sleep(1)
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

                return {
                    "success": True,
                    "message": "语音播放成功",
                    "media_player": media_player_entity,
                    "voice": voice,
                }

            except Exception as exc:
                _LOGGER.error("Media playback failed: %s", exc)
                return {
                    "success": False,
                    "error": f"Media playback failed: {exc}",
                    "audio_data": audio_base64,
                }

        return {
            "success": True,
            "audio_data": audio_base64,
            "audio_format": "mp3",
            "voice": voice,
        }

    except ServiceValidationError as exc:
        _LOGGER.error("TTS service validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        _LOGGER.error("TTS service error: %s", exc, exc_info=True)
        return {"success": False, "error": f"TTS 生成失败: {exc}"}


async def handle_tts_stream(
    hass: HomeAssistant,
    call: ServiceCall
) -> dict:
    """Handle streaming Edge TTS service call - 流式生成语音."""
    try:
        edge_tts = _check_edge_tts()
        if not edge_tts:
            return {
                "success": False,
                "error": "edge_tts 库未安装，请先安装: pip install edge-tts"
            }

        text = call.data["text"]
        voice = call.data.get("voice", TTS_DEFAULT_VOICE)
        chunk_size = call.data.get("chunk_size", 4096)

        if not text or not text.strip():
            raise ServiceValidationError("文本内容不能为空")

        if voice not in EDGE_TTS_VOICES:
            raise ServiceValidationError(f"不支持的语音类型: {voice}")

        _LOGGER.info("Starting streaming TTS: text='%s', voice='%s'", text[:50], voice)

        communicate = edge_tts.Communicate(text=text, voice=voice)

        audio_chunks = []
        total_bytes = 0
        chunk_count = 0

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data = chunk["data"]
                audio_chunks.append(audio_data)
                total_bytes += len(audio_data)
                chunk_count += 1

                if total_bytes >= chunk_size:
                    combined_chunk = b"".join(audio_chunks)
                    hass.bus.async_fire(
                        f"{DOMAIN}_tts_stream_chunk",
                        {
                            "voice": voice,
                            "chunk_index": len(audio_chunks),
                            "chunk_size": len(combined_chunk),
                            "total_bytes": total_bytes,
                            "audio_chunk": base64.b64encode(combined_chunk).decode("utf-8"),
                            "content_type": "audio/mpeg",
                        }
                    )
                    audio_chunks = []

        if audio_chunks:
            final_chunk = b"".join(audio_chunks)
            hass.bus.async_fire(
                f"{DOMAIN}_tts_stream_chunk",
                {
                    "voice": voice,
                    "chunk_index": chunk_count + 1,
                    "chunk_size": len(final_chunk),
                    "total_bytes": total_bytes,
                    "audio_chunk": base64.b64encode(final_chunk).decode("utf-8"),
                    "content_type": "audio/mpeg",
                }
            )

        hass.bus.async_fire(
            f"{DOMAIN}_tts_stream_complete",
            {"voice": voice, "total_chunks": chunk_count, "total_bytes": total_bytes, "text": text}
        )

        _LOGGER.info("Streaming TTS completed: %d chunks, %d bytes", chunk_count, total_bytes)

        return {
            "success": True,
            "method": "stream",
            "voice": voice,
            "total_chunks": chunk_count,
            "total_bytes": total_bytes,
            "message": "音频流已通过事件总线推送，请监听 ai_hub_tts_stream_chunk 事件",
        }

    except ServiceValidationError as exc:
        _LOGGER.error("Streaming TTS validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        _LOGGER.error("Streaming TTS error: %s", exc, exc_info=True)
        return {"success": False, "error": f"流式 TTS 生成失败: {exc}"}
