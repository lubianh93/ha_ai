"""Constants for the HA AI integration.

This module contains all constants organized by category:

SECTION 1: Domain and API URLs
SECTION 2: Timeouts and Retry Configuration
SECTION 3: Cache and Audio Limits
SECTION 4: Configuration Keys
SECTION 5: Default Values and Recommended Options
SECTION 6: Model Lists
SECTION 7: Error Messages
SECTION 8: Services

For better organization, consider extracting these to separate modules:
- const/urls.py - API endpoints
- const/timeouts.py - Timeout configurations
- const/models.py - Model lists
- const/defaults.py - Default values
- const/errors.py - Error messages
"""

from __future__ import annotations

import logging
from typing import Any, Final

try:
    from homeassistant.core import HomeAssistant
except ModuleNotFoundError:  # pragma: no cover - used only in lightweight test environments
    HomeAssistant = Any  # type: ignore[assignment]

# Import llm for API constants
try:
    from homeassistant.helpers import llm
    LLM_API_ASSIST = llm.LLM_API_ASSIST
    DEFAULT_INSTRUCTIONS_PROMPT = llm.DEFAULT_INSTRUCTIONS_PROMPT
except ImportError:
    LLM_API_ASSIST = "assist"
    DEFAULT_INSTRUCTIONS_PROMPT = "You are a helpful AI assistant."

_LOGGER = logging.getLogger(__name__)
LOGGER = _LOGGER  # Backwards compatibility


def get_localized_name(hass: HomeAssistant, zh_name: str, en_name: str) -> str:
    """Return localized name based on Home Assistant language setting."""
    language = hass.config.language
    chinese_languages = ["zh", "zh-cn", "zh-hans", "zh-hant", "zh-tw", "zh-hk"]
    if language and language.lower() in chinese_languages:
        return zh_name
    return en_name


# =============================================================================
# Domain and API URLs
# =============================================================================

DOMAIN: Final = "ha_ai"

# API Endpoints
API_URLS: Final = {
    "chat": "https://api.siliconflow.cn/v1/chat/completions",
    "image": "https://api.siliconflow.cn/v1/images/generations",
    "siliconflow_base": "https://api.siliconflow.cn/v1",
    "siliconflow_asr": "https://api.siliconflow.cn/v1/audio/transcriptions",
}

AI_HUB_CHAT_URL: Final = API_URLS["chat"]
AI_HUB_IMAGE_GEN_URL: Final = API_URLS["image"]
SILICONFLOW_API_BASE: Final = API_URLS["siliconflow_base"]
SILICONFLOW_ASR_URL: Final = API_URLS["siliconflow_asr"]

CONF_STT_URL: Final = "stt_url"

SUBENTRY_CONVERSATION: Final = "conversation"
SUBENTRY_AI_TASK: Final = "ai_task_data"
SUBENTRY_STT: Final = "stt"
SUBENTRY_TRANSLATION: Final = "translation"
# =============================================================================
# Timeouts Configuration (in seconds)
# =============================================================================

TIMEOUTS: Final = {
    "default": 30.0,
    "chat_api": 60.0,
    "image_api": 120.0,
    "stt_api": 30.0,
    "tts_api": 30.0,
    "media_download": 30.0,
    "health_check": 10.0,
}

DEFAULT_REQUEST_TIMEOUT: Final = 30000  # milliseconds
TIMEOUT_DEFAULT: Final = TIMEOUTS["default"]
TIMEOUT_CHAT_API: Final = TIMEOUTS["chat_api"]
TIMEOUT_IMAGE_API: Final = TIMEOUTS["image_api"]
TIMEOUT_STT_API: Final = TIMEOUTS["stt_api"]
TIMEOUT_TTS_API: Final = TIMEOUTS["tts_api"]
TIMEOUT_MEDIA_DOWNLOAD: Final = TIMEOUTS["media_download"]
TIMEOUT_HEALTH_CHECK: Final = TIMEOUTS["health_check"]


# =============================================================================
# Retry Configuration
# =============================================================================

RETRY_CONFIG: Final = {
    "max_attempts": 3,
    "base_delay": 1.0,
    "max_delay": 30.0,
    "exponential_base": 2.0,
}

# Legacy retry constants
RETRY_MAX_ATTEMPTS: Final = RETRY_CONFIG["max_attempts"]
RETRY_BASE_DELAY: Final = RETRY_CONFIG["base_delay"]
RETRY_MAX_DELAY: Final = RETRY_CONFIG["max_delay"]
RETRY_EXPONENTIAL_BASE: Final = RETRY_CONFIG["exponential_base"]


# =============================================================================
# Cache Configuration
# =============================================================================

CACHE_CONFIG: Final = {
    "tts_max_size": 100,
    "tts_ttl": 3600,  # 1 hour
}

TTS_CACHE_MAX_SIZE: Final = CACHE_CONFIG["tts_max_size"]
TTS_CACHE_TTL: Final = CACHE_CONFIG["tts_ttl"]


# =============================================================================
# Audio Size Limits
# =============================================================================

AUDIO_LIMITS: Final = {
    "stt_min_size": 1000,  # 1KB
    "stt_max_size": 10 * 1024 * 1024,  # 10MB
    "stt_warning_size": 500 * 1024,  # 500KB
    "stt_max_file_size_mb": 25,
}

STT_MIN_AUDIO_SIZE: Final = AUDIO_LIMITS["stt_min_size"]
STT_MAX_AUDIO_SIZE: Final = AUDIO_LIMITS["stt_max_size"]
STT_WARNING_AUDIO_SIZE: Final = AUDIO_LIMITS["stt_warning_size"]
STT_MAX_FILE_SIZE_MB: Final = AUDIO_LIMITS["stt_max_file_size_mb"]


# =============================================================================
# Configuration Keys
# =============================================================================

# API Keys
CONF_API_KEY: Final = "api_key"
CONF_CUSTOM_API_KEY: Final = "custom_api_key"

# Model Configuration
CONF_CHAT_MODEL: Final = "chat_model"
CONF_CHAT_URL: Final = "chat_url"
CONF_IMAGE_MODEL: Final = "image_model"
CONF_IMAGE_URL: Final = "image_url"
CONF_STT_MODEL: Final = "model"
CONF_STT_FILE: Final = "file"
CONF_LLM_PROVIDER: Final = "llm_provider"
# LLM Parameters
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_PROMPT: Final = "prompt"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_TOP_K: Final = "top_k"
CONF_LLM_HASS_API: Final = "llm_hass_api"
CONF_RECOMMENDED: Final = "recommended"
CONF_MAX_HISTORY_MESSAGES: Final = "max_history_messages"
CONF_LONG_MEMORY_ENABLED: Final = "long_memory_enabled"
CONF_LONG_MEMORY_UPDATE_TURNS: Final = "long_memory_update_turns"
CONF_LONG_MEMORY_MAX_CHARS: Final = "long_memory_max_chars"
CONF_LONG_MEMORY_PINNED: Final = "long_memory_pinned"
CONF_LONG_MEMORY_GLOBAL: Final = "long_memory_global"
CONF_LONG_MEMORY_CONVERSATION: Final = "long_memory_conversation"

# TTS Configuration
CONF_TTS_VOICE: Final = "voice"
CONF_TTS_LANG: Final = "lang"

# =============================================================================
# Recommended Values
# =============================================================================

RECOMMENDED: Final[dict[str, Any]] = {
    # Conversation
    "chat_model": "Qwen/Qwen3-8B",
    "temperature": 0.3,
    "top_p": 0.5,
    "top_k": 1,
    "max_tokens": 250,
    "max_history_messages": 30,
    "long_memory_enabled": True,
    "long_memory_update_turns": 8,
    "long_memory_max_chars": 1200,
    "long_memory_pinned": "",
    # AI Task
    "ai_task_model": "Qwen/Qwen3-8B",
    "ai_task_temperature": 0.95,
    "ai_task_top_p": 0.7,
    "ai_task_max_tokens": 2000,
    # Image (Free models on SiliconFlow)
    "image_model": "Kwai-Kolors/Kolors",
    "image_analysis_model": "THUDM/GLM-4.1V-9B-Thinking",
    # TTS
    "tts_voice": "zh-CN-XiaoxiaoNeural",
    # STT
    "stt_model": "FunAudioLLM/SenseVoiceSmall",
}

RECOMMENDED_CHAT_MODEL: Final = RECOMMENDED["chat_model"]
RECOMMENDED_TEMPERATURE: Final = RECOMMENDED["temperature"]
RECOMMENDED_TOP_P: Final = RECOMMENDED["top_p"]
RECOMMENDED_TOP_K: Final = RECOMMENDED["top_k"]
RECOMMENDED_MAX_TOKENS: Final = RECOMMENDED["max_tokens"]
RECOMMENDED_LONG_MEMORY_ENABLED: Final = RECOMMENDED["long_memory_enabled"]
RECOMMENDED_LONG_MEMORY_UPDATE_TURNS: Final = RECOMMENDED["long_memory_update_turns"]
RECOMMENDED_LONG_MEMORY_MAX_CHARS: Final = RECOMMENDED["long_memory_max_chars"]
RECOMMENDED_LONG_MEMORY_PINNED: Final = RECOMMENDED["long_memory_pinned"]
RECOMMENDED_MAX_HISTORY_MESSAGES: Final = RECOMMENDED["max_history_messages"]
RECOMMENDED_AI_TASK_MODEL: Final = RECOMMENDED["ai_task_model"]
RECOMMENDED_AI_TASK_TEMPERATURE: Final = RECOMMENDED["ai_task_temperature"]
RECOMMENDED_AI_TASK_TOP_P: Final = RECOMMENDED["ai_task_top_p"]
RECOMMENDED_AI_TASK_MAX_TOKENS: Final = RECOMMENDED["ai_task_max_tokens"]
RECOMMENDED_IMAGE_MODEL: Final = RECOMMENDED["image_model"]
RECOMMENDED_IMAGE_ANALYSIS_MODEL: Final = RECOMMENDED["image_analysis_model"]
RECOMMENDED_TTS_MODEL: Final = RECOMMENDED["tts_voice"]
RECOMMENDED_STT_MODEL: Final = RECOMMENDED["stt_model"]
LLM_PROVIDER_OPTIONS: Final = [
    "openai_compatible",
    "anthropic_compatible",
]
DEFAULT_LLM_PROVIDER: Final = "openai_compatible"

# =============================================================================
# Default Names
# =============================================================================

DEFAULT_NAMES: Final = {
    "title": "HA AI",
    "conversation": {"zh": "HA AI对话助手", "en": "HA AI Assistant"},
    "ai_task": {"zh": "HA AI AI任务", "en": "HA AIb Task"},
    "tts": {"zh": "HA AI TTS语音", "en": "HA AI TTS"},
    "stt": {"zh": "HA AI STT语音", "en": "HA AI STT"},
}

# Legacy default name constants
DEFAULT_TITLE: Final = DEFAULT_NAMES["title"]  # type: ignore[index]
DEFAULT_CONVERSATION_NAME: Final = DEFAULT_NAMES["conversation"]["zh"]  # type: ignore[index]
DEFAULT_AI_TASK_NAME: Final = DEFAULT_NAMES["ai_task"]["zh"]  # type: ignore[index]
DEFAULT_TTS_NAME: Final = DEFAULT_NAMES["tts"]["zh"]  # type: ignore[index]
DEFAULT_STT_NAME: Final = DEFAULT_NAMES["stt"]["zh"]  # type: ignore[index]


# =============================================================================
# TTS Default Values
# =============================================================================

TTS_DEFAULT_VOICE: Final = "zh-CN-XiaoxiaoNeural"
TTS_DEFAULT_LANG: Final = "zh-CN"
STT_DEFAULT_MODEL: Final = "FunAudioLLM/SenseVoiceSmall"

# Default voice per language
TTS_DEFAULT_VOICES: Final = {
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "zh-HK": "zh-HK-HiuMaanNeural",
    "en-US": "en-US-JennyNeural",
    "en-GB": "en-GB-LibbyNeural",
    "en-AU": "en-AU-NatashaNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "ko-KR": "ko-KR-SunHiNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "it-IT": "it-IT-ElsaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "ru-RU": "ru-RU-SvetlanaNeural",
}


# =============================================================================
# Model Lists
# =============================================================================

# Chat models (SiliconFlow)
AI_HUB_CHAT_MODELS: Final = [
    # Qwen series (recommended)
    "Qwen/Qwen3-8B",  # Free (recommended)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    # DeepSeek series
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V2.5",
    # Other popular models
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "01-ai/Yi-1.5-34B-Chat",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "THUDM/glm-4-9b-chat",
]

# Image generation models
AI_HUB_IMAGE_MODELS: Final = [
    "Kwai-Kolors/Kolors",  # Free (recommended)
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX-pro",
    "stabilityai/stable-diffusion-3-5-large",
    "stabilityai/stable-diffusion-3-medium",
]

# Vision models (support image analysis)
VISION_MODELS: Final = [
    "THUDM/GLM-4.1V-9B-Thinking",  # Free (recommended)
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
]

# Image sizes
IMAGE_SIZES: Final = [
    "1024x1024",
    "768x1344",
    "864x1152",
    "1344x768",
    "1152x864",
    "1440x720",
    "720x1440",
]

# SiliconFlow STT models
SILICONFLOW_STT_MODELS: Final = [
    "TeleAI/TeleSpeechASR",
    "FunAudioLLM/SenseVoiceSmall",  # Recommended
]

# SiliconFlow audio formats
SILICONFLOW_STT_AUDIO_FORMATS: Final = [
    "mp3", "wav", "flac", "m4a", "ogg", "webm",
]


# =============================================================================
# Error Messages
# =============================================================================

ERRORS: Final = {
    "getting_response": "Error getting response",
    "invalid_api_key": "Invalid API key",
    "cannot_connect": "Cannot connect to HA AI service",
}

ERROR_GETTING_RESPONSE: Final = ERRORS["getting_response"]
ERROR_INVALID_API_KEY: Final = ERRORS["invalid_api_key"]
ERROR_CANNOT_CONNECT: Final = ERRORS["cannot_connect"]


# =============================================================================
# Services
# =============================================================================

SERVICES: Final = {
    "generate_image": "generate_image",
    "analyze_image": "analyze_image",
    "tts_say": "tts_say",
    "stt_transcribe": "stt_transcribe",
}

# Legacy service constants
SERVICE_GENERATE_IMAGE: Final = SERVICES["generate_image"]
SERVICE_ANALYZE_IMAGE: Final = SERVICES["analyze_image"]
SERVICE_TTS_SAY: Final = SERVICES["tts_say"]
# Deprecated: use SERVICE_TTS_SAY instead
SERVICE_TTS_SPEECH: Final = "tts_speech"
SERVICE_TTS_STREAM: Final = "tts_stream"
SERVICE_STT_TRANSCRIBE: Final = SERVICES["stt_transcribe"]


# =============================================================================
# Recommended Options (Pre-built configurations)
# =============================================================================

RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: LLM_API_ASSIST,
    CONF_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_CHAT_URL: AI_HUB_CHAT_URL,
    CONF_LLM_PROVIDER: DEFAULT_LLM_PROVIDER,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_TOP_K: RECOMMENDED_TOP_K,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_MAX_HISTORY_MESSAGES: RECOMMENDED_MAX_HISTORY_MESSAGES,
    CONF_LONG_MEMORY_ENABLED: RECOMMENDED_LONG_MEMORY_ENABLED,
    CONF_LONG_MEMORY_UPDATE_TURNS: RECOMMENDED_LONG_MEMORY_UPDATE_TURNS,
    CONF_LONG_MEMORY_MAX_CHARS: RECOMMENDED_LONG_MEMORY_MAX_CHARS,
    CONF_LONG_MEMORY_PINNED: RECOMMENDED_LONG_MEMORY_PINNED,
}

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
    CONF_IMAGE_URL: AI_HUB_IMAGE_GEN_URL,
    CONF_TEMPERATURE: RECOMMENDED_AI_TASK_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_AI_TASK_TOP_P,
    CONF_MAX_TOKENS: RECOMMENDED_AI_TASK_MAX_TOKENS,
}

RECOMMENDED_TTS_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_TTS_VOICE: TTS_DEFAULT_VOICE,
    CONF_TTS_LANG: TTS_DEFAULT_LANG,
}

RECOMMENDED_STT_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_STT_MODEL: RECOMMENDED_STT_MODEL,
    CONF_STT_URL: SILICONFLOW_ASR_URL,
}




# =============================================================================
# Edge TTS Voices (moved to separate file for cleanliness)
# =============================================================================

# Import from separate file to keep this file manageable
from .voices import EDGE_TTS_VOICES  # noqa: E402, F401
