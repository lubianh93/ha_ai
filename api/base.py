"""Lightweight API client base classes used by tests and services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class APIResponse:
    """Normalized API response container."""

    success: bool
    data: Any = None
    status_code: int | None = None

    @property
    def is_error(self) -> bool:
        return not self.success

    def get_error_message(self) -> str | None:
        if self.success:
            return None
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, dict):
            if isinstance(self.data.get("error"), str):
                return self.data["error"]
            if isinstance(self.data.get("message"), str):
                return self.data["message"]
            nested_error = self.data.get("error")
            if isinstance(nested_error, dict) and isinstance(nested_error.get("message"), str):
                return nested_error["message"]
        return None


class APIError(Exception):
    """Base API exception."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(APIError):
    """Raised for authentication failures."""


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class TimeoutError(APIError):
    """Raised for API timeout failures."""


class APIClient(ABC):
    """Abstract API client with shared session handling."""

    def __init__(self, api_key: str, session: aiohttp.ClientSession | None = None) -> None:
        self._api_key = api_key
        self._session = session
        self._own_session = session is None

    @property
    def api_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _get_base_url(self) -> str:
        """Return API base URL."""

    def _get_default_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._own_session = True
        return self._session

    async def close(self) -> None:
        if self._session is not None and self._own_session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "APIClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _extract_error_message(self, payload: Any) -> str | None:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("error"), str):
                return payload["error"]
            if isinstance(payload.get("message"), str):
                return payload["message"]
            nested_error = payload.get("error")
            if isinstance(nested_error, dict):
                nested_message = nested_error.get("message")
                if isinstance(nested_message, str):
                    return nested_message
        return None
