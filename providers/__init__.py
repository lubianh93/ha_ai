"""Provider abstraction for AI Hub integration.

This module provides a unified abstraction layer for all service providers
(LLM, TTS, STT), making it easy to add support for new providers and
switch between them.

Features:
- Unified interface for all provider types
- Provider registration and discovery
- Easy configuration management
- Recommended mode with zero-config defaults
- Advanced mode with full customization

Usage:
    # Get the global registry
    from .providers import get_registry, ProviderType

    registry = get_registry()

    # List available providers
    llm_providers = registry.list_providers(ProviderType.LLM)
    tts_providers = registry.list_providers(ProviderType.TTS)

    # Create a provider with default config (recommended mode)
    tts = create_default_provider(ProviderType.TTS, {"api_key": None})

    # Create a specific provider (advanced mode)
    stt = create_provider("siliconflow_stt", {"api_key": "xxx"})
"""

from __future__ import annotations

import logging

# LLM provider classes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

# Re-export base classes
from .base import (
    BaseProvider,
    BaseProviderConfig,
    ProviderInfo,
    ProviderType,
    UnifiedProviderRegistry,
)

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# LLM Provider Classes
# ============================================================================


@dataclass
class LLMMessage:
    """Represents a message in a conversation.

    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content (text or list of content parts)
        tool_calls: Optional list of tool calls (for assistant messages)
        tool_call_id: Optional tool call ID (for tool messages)
    """

    role: str
    content: str | list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class LLMConfig(BaseProviderConfig):
    """Configuration for an LLM provider.

    Attributes:
        api_key: API key for authentication
        model: Model identifier
        base_url: Base URL for the API
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        extra: Additional provider-specific configuration
    """

    model: str = ""
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 250
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: Generated text content
        tool_calls: Optional list of tool calls
        usage: Token usage information
        model: Model that was used
        finish_reason: Reason for completion
        raw_response: Raw response from the provider
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    model: str | None = None
    finish_reason: str | None = None
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


class LLMProvider(BaseProvider[LLMConfig], ABC):
    """Abstract base class for LLM providers.

    All LLM providers should inherit from this class and implement
    the required methods.
    """

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.LLM

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of supported models."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given messages.

        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated content
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion for the given messages.

        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters

        Yields:
            Generated content chunks
        """
        pass

    def supports_vision(self) -> bool:
        """Check if the provider supports vision/image inputs."""
        return False

    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling."""
        return True


# ============================================================================
# Global Registry Management
# ============================================================================

# Global provider registry instance
_registry: UnifiedProviderRegistry | None = None


def get_registry() -> UnifiedProviderRegistry:
    """Get or create the global provider registry.

    Returns:
        UnifiedProviderRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = UnifiedProviderRegistry()
        _register_builtin_providers(_registry)
    return _registry


def _register_builtin_providers(registry: UnifiedProviderRegistry) -> None:
    """Register all built-in providers."""
    # Import and register LLM providers

    try:
        from .openai_compatible import OpenAICompatibleProvider

        registry.register(
            OpenAICompatibleProvider,
            is_default=True,
            requires_api_key=True,
            description="OpenAI-compatible API (OpenAI, Azure, local LLMs)",
        )
    except ImportError as e:
        _LOGGER.debug("OpenAI compatible provider not available: %s", e)

    # Import and register TTS providers
    try:
        from .edge_tts import EdgeTTSProvider

        registry.register(
            EdgeTTSProvider,
            is_default=True,
            requires_api_key=False,
            description="Microsoft Edge TTS - Free, high quality",
        )
    except ImportError as e:
        _LOGGER.debug("Edge TTS provider not available: %s", e)

    # Import and register STT providers
    try:
        from .siliconflow_stt import SiliconFlowSTTProvider

        registry.register(
            SiliconFlowSTTProvider,
            is_default=True,
            requires_api_key=True,
            description="SiliconFlow ASR - Free tier available",
        )
    except ImportError as e:
        _LOGGER.debug("SiliconFlow STT provider not available: %s", e)


# ============================================================================
# Convenience Functions
# ============================================================================


def get_provider_registry() -> UnifiedProviderRegistry:
    """Get the global provider registry (alias for get_registry).

    Returns:
        UnifiedProviderRegistry instance
    """
    return get_registry()


def register_provider(
    name: str,
    provider_class: type[LLMProvider],
) -> None:
    """Register an LLM provider in the global registry.

    This is a backwards-compatible function for registering LLM providers.

    Args:
        name: Provider name
        provider_class: Provider class
    """
    get_registry().register(
        provider_class,
        is_default=False,
        requires_api_key=True,
    )


def create_provider(
    name: str,
    config_dict: dict[str, Any],
) -> BaseProvider | None:
    """Create a provider from the global registry.

    Args:
        name: Provider name
        config_dict: Configuration dictionary

    Returns:
        Provider instance or None if not found
    """
    registry = get_registry()
    info = registry.get(name)

    if info is None:
        _LOGGER.warning("Unknown provider: %s", name)
        return None

    # Create appropriate config based on provider type
    if info.provider_type == ProviderType.LLM:
        config = LLMConfig(**config_dict)
    elif info.provider_type == ProviderType.TTS:
        from .tts_base import TTSConfig

        config = TTSConfig(**config_dict)
    elif info.provider_type == ProviderType.STT:
        from .stt_base import STTConfig

        config = STTConfig(**config_dict)
    else:
        config = BaseProviderConfig(**config_dict)

    return info.provider_class(config)


def create_default_provider(
    provider_type: ProviderType,
    config_dict: dict[str, Any],
) -> BaseProvider | None:
    """Create the default provider for a type.

    This is used in recommended mode where users don't need to configure
    which provider to use.

    Args:
        provider_type: Type of provider (LLM, TTS, STT)
        config_dict: Configuration dictionary

    Returns:
        Provider instance or None
    """
    registry = get_registry()
    info = registry.get_default(provider_type)

    if info is None:
        _LOGGER.warning("No default provider for type: %s", provider_type.value)
        return None

    # Get default config from provider class and merge with provided config
    default_config = info.provider_class.get_default_config()
    merged_config = {**default_config, **config_dict}

    return create_provider(info.provider_class._name, merged_config)


def list_providers(provider_type: ProviderType | None = None) -> list[str]:
    """List all registered provider names.

    Args:
        provider_type: Optional type filter

    Returns:
        List of provider names
    """
    return get_registry().list_providers(provider_type)


def get_provider_info(name: str) -> ProviderInfo | None:
    """Get information about a provider.

    Args:
        name: Provider name

    Returns:
        ProviderInfo or None
    """
    return get_registry().get(name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    "BaseProvider",
    "BaseProviderConfig",
    "ProviderInfo",
    "ProviderType",
    "UnifiedProviderRegistry",
    # LLM classes
    "LLMProvider",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    # Registry functions
    "get_registry",
    "get_provider_registry",
    "register_provider",
    "create_provider",
    "create_default_provider",
    "list_providers",
    "get_provider_info",
]
