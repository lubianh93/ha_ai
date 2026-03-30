"""Services library module for AI Hub integration - 模块化服务.

本模块提供 AI Hub 集成的所有服务功能，包括：

模块结构:
- schemas.py: 服务调用的数据验证模式
- image.py: 图像分析和生成服务
- tts.py: 文本转语音服务
- stt.py: 语音转文本服务

使用示例:
    from .services_lib import (
        handle_analyze_image,
        handle_generate_image,
        handle_tts_speech,
        handle_stt_transcribe,
    )

    # 在服务注册中使用
    result = await handle_analyze_image(hass, call, api_key)
"""

from __future__ import annotations


# Image services
from .image import (
    handle_analyze_image,
    handle_generate_image,
    handle_stream_response,
    load_image_from_camera,
    load_image_from_file,
    process_image,
)

# Schemas
from .schemas import (
    IMAGE_ANALYZER_SCHEMA,
    IMAGE_GENERATOR_SCHEMA,
    STT_SCHEMA,
    TTS_SCHEMA,
    TTS_STREAM_SCHEMA,
)

# STT services
from .stt import handle_stt_transcribe


# TTS services
from .tts import (
    handle_tts_speech,
    handle_tts_stream,
)

__all__ = [
    # Schemas
    "IMAGE_ANALYZER_SCHEMA",
    "IMAGE_GENERATOR_SCHEMA",
    "TTS_SCHEMA",
    "TTS_STREAM_SCHEMA",
    "STT_SCHEMA",
    # Image
    "handle_analyze_image",
    "handle_generate_image",
    "load_image_from_file",
    "load_image_from_camera",
    "process_image",
    "handle_stream_response",
    # TTS
    "handle_tts_speech",
    "handle_tts_stream",
    # STT
    "handle_stt_transcribe",    
]
