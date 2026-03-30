"""STT Provider abstraction for AI Hub integration.

This module provides the abstract base class for STT (Speech-to-Text) providers.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .base import BaseProvider, BaseProviderConfig, ProviderType

_LOGGER = logging.getLogger(__name__)


@dataclass
class STTConfig(BaseProviderConfig):
    """Configuration for STT providers.

    Attributes:
        model: STT model identifier
        language: Expected language of the audio
        sample_rate: Audio sample rate in Hz
    """

    model: str = ""
    language: str = "zh-CN"
    sample_rate: int = 16000
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioMetadata:
    """Metadata about audio being transcribed.

    Attributes:
        format: Audio format (wav, mp3, etc.)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        bit_rate: Bits per sample
    """

    format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    bit_rate: int = 16


@dataclass
class STTResult:
    """Result from STT transcription.

    Attributes:
        text: Transcribed text
        confidence: Confidence score (0-1, optional)
        language: Detected language (optional)
        duration_ms: Audio duration in milliseconds (optional)
    """

    text: str
    confidence: float | None = None
    language: str | None = None
    duration_ms: int | None = None

    @property
    def is_empty(self) -> bool:
        """Check if the result is empty."""
        return not self.text or not self.text.strip()


class STTProvider(BaseProvider[STTConfig]):
    """Abstract base class for STT providers.

    All STT providers should inherit from this class and implement
    the required methods.

    Example:
        class MySTTProvider(STTProvider):
            async def transcribe(self, audio_data, metadata, **kwargs):
                # Implementation
                pass
    """

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.STT

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported audio formats."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of supported model identifiers."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        metadata: AudioMetadata,
        **kwargs: Any,
    ) -> STTResult:
        """Transcribe audio to text.

        Args:
            audio_data: Audio bytes to transcribe
            metadata: Audio metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            STTResult containing the transcribed text
        """
        pass

    def validate_audio(
        self,
        audio_data: bytes,
        metadata: AudioMetadata,
    ) -> tuple[bool, str | None]:
        """Validate audio data before transcription.

        Args:
            audio_data: Audio bytes
            metadata: Audio metadata

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check format
        if metadata.format.lower() not in self.supported_formats:
            return False, f"Unsupported format: {metadata.format}"

        # Check minimum size (100 bytes is too small for meaningful audio)
        if len(audio_data) < 100:
            return False, "Audio data too small"

        return True, None

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for recommended mode."""
        return {
            "model": "",  # Will use provider default
            "language": "zh-CN",
        }
