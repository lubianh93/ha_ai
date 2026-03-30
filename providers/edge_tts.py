"""Edge TTS provider implementation.

This module provides TTS using Microsoft Edge's free TTS service.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from .base import ProviderType
from .tts_base import TTSProvider, TTSResult

_LOGGER = logging.getLogger(__name__)

# Check for edge_tts availability
try:
    import edge_tts
    import edge_tts.exceptions

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    _LOGGER.warning("edge_tts not installed, EdgeTTSProvider will not be available")


# Default voices for each language
DEFAULT_VOICES = {
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "zh-HK": "zh-HK-HiuMaanNeural",
    "en-US": "en-US-JennyNeural",
    "en-GB": "en-GB-SoniaNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "ko-KR": "ko-KR-SunHiNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "it-IT": "it-IT-ElsaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "ru-RU": "ru-RU-SvetlanaNeural",
}

# Voice to language mapping (commonly used voices)
EDGE_TTS_VOICES = {
    # Chinese
    "zh-CN-XiaoxiaoNeural": "zh-CN",
    "zh-CN-YunxiNeural": "zh-CN",
    "zh-CN-YunjianNeural": "zh-CN",
    "zh-CN-XiaoyiNeural": "zh-CN",
    "zh-CN-YunyangNeural": "zh-CN",
    "zh-CN-XiaochenNeural": "zh-CN",
    "zh-CN-XiaohanNeural": "zh-CN",
    "zh-CN-XiaomengNeural": "zh-CN",
    "zh-CN-XiaomoNeural": "zh-CN",
    "zh-CN-XiaoqiuNeural": "zh-CN",
    "zh-CN-XiaoruiNeural": "zh-CN",
    "zh-CN-XiaoshuangNeural": "zh-CN",
    "zh-CN-XiaoxuanNeural": "zh-CN",
    "zh-CN-XiaoyanNeural": "zh-CN",
    "zh-CN-XiaoyouNeural": "zh-CN",
    "zh-CN-XiaozhenNeural": "zh-CN",
    "zh-CN-YunfengNeural": "zh-CN",
    "zh-CN-YunhaoNeural": "zh-CN",
    "zh-CN-YunxiaNeural": "zh-CN",
    "zh-CN-YunzeNeural": "zh-CN",
    "zh-CN-liaoning-XiaobeiNeural": "zh-CN",
    "zh-CN-shaanxi-XiaoniNeural": "zh-CN",
    # Taiwan
    "zh-TW-HsiaoChenNeural": "zh-TW",
    "zh-TW-YunJheNeural": "zh-TW",
    "zh-TW-HsiaoYuNeural": "zh-TW",
    # Hong Kong
    "zh-HK-HiuMaanNeural": "zh-HK",
    "zh-HK-WanLungNeural": "zh-HK",
    "zh-HK-HiuGaaiNeural": "zh-HK",
    # English
    "en-US-JennyNeural": "en-US",
    "en-US-GuyNeural": "en-US",
    "en-US-AriaNeural": "en-US",
    "en-US-DavisNeural": "en-US",
    "en-US-AmberNeural": "en-US",
    "en-US-AnaNeural": "en-US",
    "en-US-AshleyNeural": "en-US",
    "en-US-BrandonNeural": "en-US",
    "en-US-ChristopherNeural": "en-US",
    "en-US-CoraNeural": "en-US",
    "en-US-ElizabethNeural": "en-US",
    "en-US-EricNeural": "en-US",
    "en-US-JacobNeural": "en-US",
    "en-US-MichelleNeural": "en-US",
    "en-US-MonicaNeural": "en-US",
    "en-US-SaraNeural": "en-US",
    "en-GB-SoniaNeural": "en-GB",
    "en-GB-RyanNeural": "en-GB",
    # Japanese
    "ja-JP-NanamiNeural": "ja-JP",
    "ja-JP-KeitaNeural": "ja-JP",
    # Korean
    "ko-KR-SunHiNeural": "ko-KR",
    "ko-KR-InJoonNeural": "ko-KR",
    # European
    "fr-FR-DeniseNeural": "fr-FR",
    "fr-FR-HenriNeural": "fr-FR",
    "de-DE-KatjaNeural": "de-DE",
    "de-DE-ConradNeural": "de-DE",
    "es-ES-ElviraNeural": "es-ES",
    "es-ES-AlvaroNeural": "es-ES",
    "it-IT-ElsaNeural": "it-IT",
    "it-IT-DiegoNeural": "it-IT",
    "pt-BR-FranciscaNeural": "pt-BR",
    "pt-BR-AntonioNeural": "pt-BR",
    "ru-RU-SvetlanaNeural": "ru-RU",
    "ru-RU-DmitryNeural": "ru-RU",
}


class EdgeTTSProvider(TTSProvider):
    """Edge TTS provider using Microsoft's free TTS service.

    Features:
    - Free to use
    - High quality neural voices
    - Support for multiple languages
    - Prosody control (pitch, rate, volume)
    - No API key required

    Example:
        config = TTSConfig(
            voice="zh-CN-XiaoxiaoNeural",
            pitch="+0Hz",
            rate="+0%",
        )
        provider = EdgeTTSProvider(config)

        result = await provider.synthesize("Hello, world!")
    """

    # Class-level attributes for registration
    _name = "edge_tts"
    _provider_type = ProviderType.TTS

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "edge_tts"

    @property
    def display_name(self) -> str:
        """Return a human-readable display name."""
        return "Microsoft Edge TTS"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        return list(set(EDGE_TTS_VOICES.values()))

    @property
    def supported_voices(self) -> dict[str, str]:
        """Return mapping of voice IDs to their language codes."""
        return EDGE_TTS_VOICES

    def _get_default_voice(self) -> str:
        """Get the default voice based on config language."""
        language = self.config.language or "zh-CN"
        return DEFAULT_VOICES.get(language, "zh-CN-XiaoxiaoNeural")

    def _resolve_voice(self, voice: str | None) -> str:
        """Resolve the voice to use.

        Args:
            voice: Requested voice or None

        Returns:
            Valid voice identifier
        """
        if voice and voice in EDGE_TTS_VOICES:
            return voice

        # Try config voice
        if self.config.voice and self.config.voice in EDGE_TTS_VOICES:
            return self.config.voice

        # Fall back to default for language
        return self._get_default_voice()

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (uses default if not specified)
            **kwargs: Additional parameters (pitch, rate, volume)

        Returns:
            TTSResult containing the audio data
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge_tts package is not installed")

        if not text or not text.strip():
            raise ValueError("Text content cannot be empty")

        resolved_voice = self._resolve_voice(voice)

        # Get prosody parameters
        pitch = kwargs.get("pitch", self.config.pitch)
        rate = kwargs.get("rate", self.config.rate)
        volume = kwargs.get("volume", self.config.volume)

        _LOGGER.debug(
            "EdgeTTS: text=%s..., voice=%s, pitch=%s, rate=%s, volume=%s",
            text[:30],
            resolved_voice,
            pitch,
            rate,
            volume,
        )

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=resolved_voice,
                pitch=pitch,
                rate=rate,
                volume=volume,
            )

            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if not audio_bytes:
                raise RuntimeError("No audio data generated")

            return TTSResult(
                audio_data=audio_bytes,
                audio_format="mp3",
            )

        except edge_tts.exceptions.NoAudioReceived as exc:
            _LOGGER.warning("Edge TTS received no audio for: %s", text[:50])
            raise RuntimeError(f"TTS received no audio: {text[:50]}") from exc

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        voice: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech from streaming text input.

        Buffers text until sentence boundaries for natural speech.
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge_tts package is not installed")

        resolved_voice = self._resolve_voice(voice)
        pitch = kwargs.get("pitch", self.config.pitch)
        rate = kwargs.get("rate", self.config.rate)
        volume = kwargs.get("volume", self.config.volume)

        separators = "\n。.，,；;！!？?、"
        buffer = ""
        count = 0

        async for chunk in text_stream:
            count += 1
            min_len = 2**count * 10

            for char in chunk:
                buffer += char
                msg = buffer.strip()
                if len(msg) >= min_len and char in separators:
                    try:
                        communicate = edge_tts.Communicate(
                            text=msg,
                            voice=resolved_voice,
                            pitch=pitch,
                            rate=rate,
                            volume=volume,
                        )
                        audio_bytes = b""
                        async for audio_chunk in communicate.stream():
                            if audio_chunk["type"] == "audio":
                                audio_bytes += audio_chunk["data"]
                        if audio_bytes:
                            yield audio_bytes
                    except Exception as exc:
                        _LOGGER.warning("Stream TTS chunk failed: %s", exc)
                    buffer = ""

        # Process remaining buffer
        if msg := buffer.strip():
            try:
                communicate = edge_tts.Communicate(
                    text=msg,
                    voice=resolved_voice,
                    pitch=pitch,
                    rate=rate,
                    volume=volume,
                )
                audio_bytes = b""
                async for audio_chunk in communicate.stream():
                    if audio_chunk["type"] == "audio":
                        audio_bytes += audio_chunk["data"]
                if audio_bytes:
                    yield audio_bytes
            except Exception as exc:
                _LOGGER.warning("Stream TTS final chunk failed: %s", exc)

    async def health_check(self) -> bool:
        """Check if Edge TTS is available."""
        return EDGE_TTS_AVAILABLE

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for recommended mode."""
        return {
            "voice": "zh-CN-XiaoxiaoNeural",
            "language": "zh-CN",
            "pitch": "+0Hz",
            "rate": "+0%",
            "volume": "+0%",
        }
