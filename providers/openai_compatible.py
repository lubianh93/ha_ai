"""OpenAI-compatible LLM provider for AI Hub integration.

This module provides an OpenAI-compatible implementation of the LLM provider
interface, which can be used with any OpenAI-compatible API endpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from . import LLMMessage, LLMProvider, LLMResponse, register_provider

_LOGGER = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible LLM provider implementation.

    This provider can be used with any API that follows the OpenAI
    chat completions format, including:
    - OpenAI
    - Azure OpenAI
    - Local LLMs (LM Studio, Ollama, etc.)
    - Other compatible services

    Example:
        config = LLMConfig(
            api_key="your-api-key",
            model="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1/chat/completions",
        )
        provider = OpenAICompatibleProvider(config)

        response = await provider.complete([
            LLMMessage(role="user", content="Hello!")
        ])
    """

    # Class-level attributes for registration
    _name = "openai_compatible"

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "openai_compatible"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported models.

        Since this is a generic provider, return empty list.
        The specific models depend on the endpoint being used.
        """
        return []

    def supports_vision(self) -> bool:
        """Check if vision is supported.

        Depends on the model being used.
        """
        vision_keywords = ["vision", "4v", "gpt-4o", "4-turbo"]
        return any(kw in self.config.model.lower() for kw in vision_keywords)

    def supports_tools(self) -> bool:
        """Check if tools are supported."""
        return True

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _get_api_url(self) -> str:
        """Get the API URL."""
        return self.config.base_url or "https://api.openai.com/v1/chat/completions"

    def _build_request(
        self,
        messages: list[LLMMessage],
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the API request body."""
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": stream,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            request["tools"] = tools

        request.update(self.config.extra)
        request.update(kwargs)

        return request

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the generated content
        """
        request = self._build_request(messages, stream=False, tools=tools, **kwargs)
        headers = self._get_headers()
        url = self._get_api_url()

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=request, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("OpenAI compatible API error: %s", error_text)
                    raise Exception(f"API error: {error_text}")

                data = await response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            usage=data.get("usage"),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools
            **kwargs: Additional parameters

        Yields:
            Generated content chunks
        """
        request = self._build_request(messages, stream=True, tools=tools, **kwargs)
        headers = self._get_headers()
        url = self._get_api_url()

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=request, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("OpenAI compatible streaming error: %s", error_text)
                    raise Exception(f"API error: {error_text}")

                buffer = ""
                async for chunk in response.content:
                    if not chunk:
                        continue

                    chunk_text = chunk.decode("utf-8", errors="ignore")
                    buffer += chunk_text

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line or line == "data: [DONE]":
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if not data_str.strip():
                                continue

                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                _LOGGER.debug("SSE parse failed: %s", data_str)
                                continue

    async def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            # Try a simple GET to the base domain
            url = self._get_api_url()
            # Extract base URL
            from urllib.parse import urlparse

            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(base) as response:
                    return response.status < 500
        except Exception as e:
            _LOGGER.debug("OpenAI compatible health check failed: %s", e)
            return False


# Register the provider
register_provider("openai_compatible", OpenAICompatibleProvider)
