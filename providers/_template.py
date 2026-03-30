"""Provider Development Guide for AI Hub Integration.

This file serves as a template and guide for adding new service providers.
Each provider type (LLM, TTS, STT) has its own base class and requirements.

## Quick Start

1. Choose the right base class:
   - LLMProvider: For chat/completion AI models
   - TTSProvider: For text-to-speech services
   - STTProvider: For speech-to-text services

2. Create a new file in the providers/ directory

3. Implement the required abstract methods

4. Add the _name class attribute for registration

5. The provider will be auto-discovered and registered

## Example: Adding a New TTS Provider
"""

from __future__ import annotations


# Example: Creating a new TTS provider
# Uncomment and modify this template for your provider

# from .tts_base import TTSConfig, TTSProvider, TTSResult
# from .base import ProviderType
#
#
# class MyNewTTSProvider(TTSProvider):
#     '''My custom TTS provider.
#
#     Example:
#         config = TTSConfig(api_key="xxx", voice="default")
#         provider = MyNewTTSProvider(config)
#         result = await provider.synthesize("Hello!")
#     '''
#
#     # Required: class-level name for auto-registration
#     _name = "my_new_tts"
#     _provider_type = ProviderType.TTS
#
#     @property
#     def name(self) -> str:
#         return "my_new_tts"
#
#     @property
#     def display_name(self) -> str:
#         return "My New TTS Service"
#
#     @property
#     def supported_languages(self) -> list[str]:
#         return ["zh-CN", "en-US"]
#
#     @property
#     def supported_voices(self) -> dict[str, str]:
#         return {
#             "voice-1": "zh-CN",
#             "voice-2": "en-US",
#         }
#
#     async def synthesize(
#         self,
#         text: str,
#         voice: str | None = None,
#         **kwargs: Any,
#     ) -> TTSResult:
#         '''Synthesize speech from text.'''
#         # Your implementation here
#         # Call your API, process the audio, return TTSResult
#         audio_bytes = b""  # Get from API
#         return TTSResult(
#             audio_data=audio_bytes,
#             audio_format="mp3",
#         )
#
#     @classmethod
#     def get_default_config(cls) -> dict[str, Any]:
#         '''Default config for recommended mode.'''
#         return {
#             "voice": "voice-1",
#             "language": "zh-CN",
#         }


# Example: Creating a new LLM provider

# from . import LLMConfig, LLMMessage, LLMProvider, LLMResponse
# from .base import ProviderType
# from collections.abc import AsyncGenerator
#
#
# class MyNewLLMProvider(LLMProvider):
#     '''My custom LLM provider.'''
#
#     _name = "my_new_llm"
#     _provider_type = ProviderType.LLM
#
#     @property
#     def name(self) -> str:
#         return "my_new_llm"
#
#     @property
#     def supported_models(self) -> list[str]:
#         return ["model-a", "model-b"]
#
#     async def complete(
#         self,
#         messages: list[LLMMessage],
#         **kwargs: Any,
#     ) -> LLMResponse:
#         '''Generate a completion.'''
#         # Call your API
#         content = "Response from API"
#         return LLMResponse(content=content)
#
#     async def complete_stream(
#         self,
#         messages: list[LLMMessage],
#         **kwargs: Any,
#     ) -> AsyncGenerator[str, None]:
#         '''Generate streaming completion.'''
#         # Stream from your API
#         yield "Hello "
#         yield "World!"


# Example: Creating a new STT provider

# from .stt_base import AudioMetadata, STTConfig, STTProvider, STTResult
# from .base import ProviderType
#
#
# class MyNewSTTProvider(STTProvider):
#     '''My custom STT provider.'''
#
#     _name = "my_new_stt"
#     _provider_type = ProviderType.STT
#
#     @property
#     def name(self) -> str:
#         return "my_new_stt"
#
#     @property
#     def supported_languages(self) -> list[str]:
#         return ["zh-CN", "en-US"]
#
#     @property
#     def supported_formats(self) -> list[str]:
#         return ["wav", "mp3"]
#
#     @property
#     def supported_models(self) -> list[str]:
#         return ["model-1"]
#
#     async def transcribe(
#         self,
#         audio_data: bytes,
#         metadata: AudioMetadata,
#         **kwargs: Any,
#     ) -> STTResult:
#         '''Transcribe audio to text.'''
#         # Call your API
#         text = "Transcribed text"
#         return STTResult(text=text)


# To register your provider, add it to _register_builtin_providers in __init__.py:
#
# try:
#     from .my_provider import MyNewTTSProvider
#     registry.register(
#         MyNewTTSProvider,
#         is_default=False,  # Set True if this should be the default
#         requires_api_key=True,  # Set False if no API key needed
#         description="My provider description",
#     )
# except ImportError as e:
#     _LOGGER.debug("My provider not available: %s", e)
