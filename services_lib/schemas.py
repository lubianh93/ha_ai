"""Service schemas for AI Hub integration - 服务调用数据验证模式.

本模块定义所有 AI Hub 服务的数据验证模式 (Schema)。

可用模式:
- IMAGE_ANALYZER_SCHEMA: 图像分析服务
- IMAGE_GENERATOR_SCHEMA: 图像生成服务
- TTS_SCHEMA: 文本转语音服务
- TTS_STREAM_SCHEMA: 流式文本转语音服务
- STT_SCHEMA: 语音转文本服务
- TRANSLATION_SCHEMA: 组件翻译服务
- BLUEPRINTS_TRANSLATION_SCHEMA: 蓝图翻译服务
"""

import voluptuous as vol
from homeassistant.helpers import config_validation as cv

from ..const import (
    CONF_STT_FILE,
    EDGE_TTS_VOICES,
    IMAGE_SIZES,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TEMPERATURE,
    SILICONFLOW_STT_MODELS,
    TTS_DEFAULT_VOICE,
)

# Schema for image analysis service
IMAGE_ANALYZER_SCHEMA = {
    vol.Optional("image_file"): cv.string,
    vol.Optional("image_entity"): cv.entity_id,
    vol.Required("message"): cv.string,
    vol.Optional("model", default=RECOMMENDED_IMAGE_ANALYSIS_MODEL): cv.string,
    vol.Optional("temperature", default=RECOMMENDED_TEMPERATURE): vol.Coerce(float),
    vol.Optional("max_tokens", default=RECOMMENDED_MAX_TOKENS): cv.positive_int,
    vol.Optional("stream", default=False): cv.boolean,
}

# Schema for image generation service
IMAGE_GENERATOR_SCHEMA = {
    vol.Required("prompt"): cv.string,
    vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
    vol.Optional("model", default=RECOMMENDED_IMAGE_MODEL): cv.string,
}

# Schema for Edge TTS service
TTS_SCHEMA = {
    vol.Required("text"): cv.string,
    vol.Optional("voice", default=TTS_DEFAULT_VOICE): vol.In(list(EDGE_TTS_VOICES.keys())),
    vol.Optional("media_player_entity"): cv.entity_id,
}

# Schema for streaming Edge TTS service
TTS_STREAM_SCHEMA = {
    vol.Required("text"): cv.string,
    vol.Optional("voice", default=TTS_DEFAULT_VOICE): vol.In(list(EDGE_TTS_VOICES.keys())),
    vol.Optional("chunk_size", default=4096): vol.Coerce(int),
}

# Schema for Silicon Flow STT service
STT_SCHEMA = {
    vol.Required(CONF_STT_FILE): cv.string,
    vol.Optional("model", default=RECOMMENDED_STT_MODEL): vol.In(SILICONFLOW_STT_MODELS),
}

