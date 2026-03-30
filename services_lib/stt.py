"""STT services for AI Hub - 语音转文字功能.

本模块提供语音识别服务，使用硅基流动 (SiliconFlow) SenseVoice API。

主要函数:
- handle_stt_transcribe: 处理语音转文字服务调用

特性:
- 自动语言检测，无需手动指定语言
- 支持中文、英文、日文、韩文等多种语言
- 支持的音频格式: WAV, MP3, FLAC, M4A, OGG, WebM
- 最大文件大小: 25MB
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError

from ..const import (
    CONF_STT_FILE,
    DEFAULT_REQUEST_TIMEOUT,
    RECOMMENDED_STT_MODEL,
    SILICONFLOW_ASR_URL,
    SILICONFLOW_STT_AUDIO_FORMATS,
    SILICONFLOW_STT_MODELS,
    STT_MAX_FILE_SIZE_MB,
)

_LOGGER = logging.getLogger(__name__)


async def handle_stt_transcribe(
    hass: HomeAssistant,
    call: ServiceCall,
    siliconflow_api_key: str
) -> dict:
    """Handle Silicon Flow STT service call."""
    try:
        if not siliconflow_api_key or not siliconflow_api_key.strip():
            return {
                "success": False,
                "error": "硅基流动API密钥未配置，请先在集成配置中设置"
            }

        audio_file = call.data[CONF_STT_FILE]
        model = call.data.get("model", RECOMMENDED_STT_MODEL)

        if not audio_file or not audio_file.strip():
            raise ServiceValidationError("音频文件路径不能为空")

        if model not in SILICONFLOW_STT_MODELS:
            raise ServiceValidationError(f"不支持的模型: {model}")

        # 处理相对路径
        if not os.path.isabs(audio_file):
            audio_file = os.path.join(hass.config.config_dir, audio_file)

        if not os.path.exists(audio_file):
            raise ServiceValidationError(f"音频文件不存在: {audio_file}")

        if os.path.isdir(audio_file):
            raise ServiceValidationError(f"提供的路径是一个目录: {audio_file}")

        file_size = os.path.getsize(audio_file)
        if file_size > STT_MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ServiceValidationError(f"音频文件过大，最大支持 {STT_MAX_FILE_SIZE_MB}MB")

        file_ext = os.path.splitext(audio_file)[1].lower().lstrip('.')
        if file_ext not in SILICONFLOW_STT_AUDIO_FORMATS:
            raise ServiceValidationError(
                f"不支持的音频格式: {file_ext}，支持的格式: {', '.join(SILICONFLOW_STT_AUDIO_FORMATS)}"
            )

        with open(audio_file, "rb") as f:
            audio_data = f.read()

        headers = {"Authorization": f"Bearer {siliconflow_api_key}"}

        form_data = aiohttp.FormData()
        form_data.add_field(
            "file", audio_data,
            filename=os.path.basename(audio_file),
            content_type=f"audio/{file_ext}"
        )
        form_data.add_field("model", model)

        timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT / 1000)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(SILICONFLOW_ASR_URL, headers=headers, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("STT API error: %s - %s", response.status, error_text)
                    return {"success": False, "error": f"STT API 请求失败: {response.status}"}

                response_data = await response.json()

                if "text" not in response_data:
                    _LOGGER.error("STT API response format error: %s", response_data)
                    return {"success": False, "error": "API 响应格式错误"}

                return {
                    "success": True,
                    "text": response_data["text"],
                    "model": model,
                    "audio_file": audio_file,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                }

    except ServiceValidationError as exc:
        _LOGGER.error("STT service validation error: %s", exc)
        return {"success": False, "error": str(exc)}
    except aiohttp.ClientError as exc:
        _LOGGER.error("STT service network error: %s", exc)
        return {"success": False, "error": f"网络请求失败: {exc}"}
    except asyncio.TimeoutError:
        _LOGGER.error("STT service timeout")
        return {"success": False, "error": "请求超时"}
    except Exception as exc:
        _LOGGER.error("STT service error: %s", exc, exc_info=True)
        return {"success": False, "error": f"STT 转录失败: {exc}"}
