"""TTS Provider abstraction for AI Hub integration.

This module provides the abstract base class for TTS (Text-to-Speech) providers.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from .base import BaseProvider, BaseProviderConfig, ProviderType

_LOGGER = logging.getLogger(__name__)


@dataclass
class TTSConfig(BaseProviderConfig):
    """Configuration for TTS providers.

    Attributes:
        voice: Voice identifier
        language: Language code (e.g., "zh-CN", "en-US")
        pitch: Voice pitch adjustment
        rate: Speaking rate adjustment
        volume: Volume adjustment
    """

    voice: str = ""
    language: str = "zh-CN"
    pitch: str = "+0Hz"
    rate: str = "+0%"
    volume: str = "+0%"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSResult:
    """Result from TTS generation.

    Attributes:
        audio_data: Generated audio bytes
        audio_format: Audio format (e.g., "mp3", "wav")
        duration_ms: Audio duration in milliseconds (optional)
    """

    audio_data: bytes
    audio_format: str = "mp3"
    duration_ms: int | None = None


class TTSProvider(BaseProvider[TTSConfig]):
    """Abstract base class for TTS providers.

    All TTS providers should inherit from this class and implement
    the required methods.

    Example:
        class MyTTSProvider(TTSProvider):
            async def synthesize(self, text, **kwargs):
                # Implementation
                pass

            async def synthesize_stream(self, text_stream, **kwargs):
                # Implementation
                pass
    """

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.TTS

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        pass

    @property
    @abstractmethod
    def supported_voices(self) -> dict[str, str]:
        """Return mapping of voice IDs to their language codes."""
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            TTSResult containing the audio data
        """
        pass

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        voice: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech from a streaming text input.

        Args:
            text_stream: Async generator yielding text chunks
            voice: Voice identifier
            **kwargs: Additional parameters

        Yields:
            Audio data chunks
        """
        # Default implementation: buffer and synthesize
        buffer = ""
        async for chunk in text_stream:
            buffer += chunk

        if buffer.strip():
            result = await self.synthesize(buffer, voice, **kwargs)
            yield result.audio_data

    def get_voice_for_language(self, language: str) -> str | None:
        """Get the default voice for a language.

        Args:
            language: Language code

        Returns:
            Voice ID or None if not found
        """
        for voice_id, voice_lang in self.supported_voices.items():
            if voice_lang == language:
                return voice_id
        return None

    def is_voice_valid(self, voice: str) -> bool:
        """Check if a voice ID is valid.

        Args:
            voice: Voice identifier

        Returns:
            True if the voice is supported
        """
        return voice in self.supported_voices

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for recommended mode."""
        return {
            "voice": "",  # Will use provider default
            "language": "zh-CN",
            "pitch": "+0Hz",
            "rate": "+0%",
            "volume": "+0%",
        }
