"""Base provider classes for AI Hub integration.

This module provides the abstract base classes for all service providers,
including unified configuration and registry management.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

_LOGGER = logging.getLogger(__name__)


class ProviderType(Enum):
    """Types of service providers."""

    LLM = "llm"
    TTS = "tts"
    STT = "stt"


@dataclass
class BaseProviderConfig:
    """Base configuration for all providers.

    Attributes:
        api_key: API key for authentication (optional for some providers)
        timeout: Request timeout in seconds
        extra: Additional provider-specific configuration
    """

    api_key: str | None = None
    timeout: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=BaseProviderConfig)


class BaseProvider(ABC, Generic[ConfigT]):
    """Abstract base class for all service providers.

    All providers (LLM, TTS, STT) should inherit from this class.
    """

    def __init__(self, config: ConfigT) -> None:
        """Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    @property
    def display_name(self) -> str:
        """Return a human-readable display name."""
        return self.name.replace("_", " ").title()

    async def health_check(self) -> bool:
        """Check if the provider is healthy.

        Returns:
            True if the provider is reachable
        """
        return True

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration values.

        Used for recommended mode setup.

        Returns:
            Dictionary of default config values
        """
        return {}

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for advanced mode.

        Returns:
            Dictionary describing config options
        """
        return {}


class ProviderInfo:
    """Information about a registered provider."""

    def __init__(
        self,
        provider_class: type[BaseProvider],
        provider_type: ProviderType,
        is_default: bool = False,
        requires_api_key: bool = True,
        description: str = "",
    ) -> None:
        """Initialize provider info.

        Args:
            provider_class: The provider class
            provider_type: Type of provider (LLM, TTS, STT)
            is_default: Whether this is the default provider for its type
            requires_api_key: Whether an API key is required
            description: Human-readable description
        """
        self.provider_class = provider_class
        self.provider_type = provider_type
        self.is_default = is_default
        self.requires_api_key = requires_api_key
        self.description = description


class UnifiedProviderRegistry:
    """Unified registry for all provider types.

    This registry manages LLM, TTS, and STT providers in a single place,
    making it easy to add new providers and discover available services.

    Example:
        registry = UnifiedProviderRegistry()

        # Register providers
        registry.register(SiliconFlowProvider, is_default=True)
        registry.register(EdgeTTSProvider, is_default=True)

        # Get providers by type
        llm_providers = registry.get_by_type(ProviderType.LLM)

        # Create provider instance
        provider = registry.create("siliconflow", config)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: dict[str, ProviderInfo] = {}
        self._defaults: dict[ProviderType, str] = {}

    def register(
        self,
        provider_class: type[BaseProvider],
        is_default: bool = False,
        requires_api_key: bool = True,
        description: str = "",
    ) -> None:
        """Register a provider class.

        Args:
            provider_class: Provider class to register
            is_default: Whether this is the default for its type
            requires_api_key: Whether an API key is required
            description: Human-readable description
        """
        # Create a temporary instance to get name and type
        # We use a minimal config just to access properties
        temp_config = BaseProviderConfig()
        try:
            temp_instance = provider_class(temp_config)  # type: ignore
            name = temp_instance.name
            provider_type = temp_instance.provider_type
        except Exception:
            # If instantiation fails, try to get from class attributes
            name = getattr(provider_class, "_name", provider_class.__name__.lower())
            provider_type = getattr(
                provider_class, "_provider_type", ProviderType.LLM
            )

        info = ProviderInfo(
            provider_class=provider_class,
            provider_type=provider_type,
            is_default=is_default,
            requires_api_key=requires_api_key,
            description=description,
        )

        self._providers[name] = info

        if is_default:
            self._defaults[provider_type] = name

        _LOGGER.debug(
            "Registered %s provider: %s (default=%s)",
            provider_type.value,
            name,
            is_default,
        )

    def unregister(self, name: str) -> None:
        """Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        if name in self._providers:
            info = self._providers[name]
            del self._providers[name]

            # Remove from defaults if it was the default
            if self._defaults.get(info.provider_type) == name:
                del self._defaults[info.provider_type]

            _LOGGER.debug("Unregistered provider: %s", name)

    def get(self, name: str) -> ProviderInfo | None:
        """Get provider info by name.

        Args:
            name: Provider name

        Returns:
            ProviderInfo or None if not found
        """
        return self._providers.get(name)

    def get_by_type(self, provider_type: ProviderType) -> list[ProviderInfo]:
        """Get all providers of a specific type.

        Args:
            provider_type: Type of providers to get

        Returns:
            List of ProviderInfo for that type
        """
        return [
            info
            for info in self._providers.values()
            if info.provider_type == provider_type
        ]

    def get_default(self, provider_type: ProviderType) -> ProviderInfo | None:
        """Get the default provider for a type.

        Args:
            provider_type: Type of provider

        Returns:
            Default ProviderInfo or None
        """
        default_name = self._defaults.get(provider_type)
        if default_name:
            return self._providers.get(default_name)
        return None

    def create(
        self,
        name: str,
        config: BaseProviderConfig,
    ) -> BaseProvider | None:
        """Create a provider instance.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            Provider instance or None if not found
        """
        info = self.get(name)
        if info is None:
            _LOGGER.warning("Unknown provider: %s", name)
            return None

        return info.provider_class(config)

    def create_default(
        self,
        provider_type: ProviderType,
        config: BaseProviderConfig,
    ) -> BaseProvider | None:
        """Create the default provider for a type.

        Args:
            provider_type: Type of provider
            config: Provider configuration

        Returns:
            Provider instance or None
        """
        info = self.get_default(provider_type)
        if info is None:
            _LOGGER.warning("No default provider for type: %s", provider_type.value)
            return None

        return info.provider_class(config)

    def list_providers(self, provider_type: ProviderType | None = None) -> list[str]:
        """List all registered provider names.

        Args:
            provider_type: Optional type filter

        Returns:
            List of provider names
        """
        if provider_type is None:
            return list(self._providers.keys())

        return [
            name
            for name, info in self._providers.items()
            if info.provider_type == provider_type
        ]

    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered
        """
        return name in self._providers
