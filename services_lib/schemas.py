"""Service schemas for HA AI integration - 服务调用数据验证模式.

本模块定义所有 HA AI 服务的数据验证模式 (Schema)。

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
    CONF_SUBENTRY_ID,
    CONF_STT_FILE,
    IMAGE_SIZES,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_TEMPERATURE,
    TTS_DEFAULT_VOICE,
)

SERVICE_TARGET_SCHEMA = {
    vol.Optional("config_entry_id"): cv.string,
    vol.Optional(CONF_SUBENTRY_ID): cv.string,
}

# Schema for image analysis service
IMAGE_ANALYZER_SCHEMA = {
    **SERVICE_TARGET_SCHEMA,
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
    **SERVICE_TARGET_SCHEMA,
    vol.Required("prompt"): cv.string,
    vol.Optional("size", default="1024x1024"): vol.In(IMAGE_SIZES),
    vol.Optional("model", default=RECOMMENDED_IMAGE_MODEL): cv.string,
}

# Schema for TTS service
TTS_SCHEMA = {
    **SERVICE_TARGET_SCHEMA,
    vol.Required("text"): cv.string,
    vol.Optional("voice", default=TTS_DEFAULT_VOICE): cv.string,
    vol.Optional("media_player_entity"): cv.entity_id,
    vol.Optional("stream", default=False): cv.boolean,
    vol.Optional("chunk_size", default=4096): vol.Coerce(int),
}

# Schema for streaming TTS service
TTS_STREAM_SCHEMA = {
    **SERVICE_TARGET_SCHEMA,
    vol.Required("text"): cv.string,
    vol.Optional("voice", default=TTS_DEFAULT_VOICE): cv.string,
    vol.Optional("chunk_size", default=4096): vol.Coerce(int),
}

# Schema for STT service
STT_SCHEMA = {
    **SERVICE_TARGET_SCHEMA,
    vol.Required(CONF_STT_FILE): cv.string,
    vol.Optional("model", default=RECOMMENDED_STT_MODEL): cv.string,
}

FOLLOW_UP_PLAYBACK_DONE_SCHEMA = {
    vol.Optional("pending_id"): cv.string,
    vol.Optional("device_id"): cv.string,
    vol.Optional("conversation_id"): cv.string,
}

RECORD_HABIT_EVENT_SCHEMA = {
    vol.Required("domain"): cv.string,
    vol.Required("service"): cv.string,
    vol.Required("entity_id"): cv.entity_id,
    vol.Optional("device_id"): cv.string,
    vol.Optional("source", default="manual"): cv.string,
}

GET_PROACTIVE_STATUS_SCHEMA = {}
