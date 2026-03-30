"""Unified retry mechanism for AI Hub integration.

This module provides a robust retry mechanism with exponential backoff
for handling transient failures in API calls and network operations.

Features:
- Configurable retry attempts and delays
- Exponential backoff with jitter
- Specific exception handling
- Logging of retry attempts
- Async-first design

Example usage:
    from .utils import async_retry_with_backoff, RetryConfig

    config = RetryConfig(max_attempts=3, base_delay=1.0)

    @async_retry_with_backoff(config)
    async def call_api():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import aiohttp

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception | None = None,
    ) -> None:
        """Initialize retry error."""
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        jitter_factor: Factor for jitter calculation (default: 0.1)
        retryable_exceptions: Tuple of exceptions that should trigger a retry
        retryable_status_codes: HTTP status codes that should trigger a retry
        on_retry: Optional callback called before each retry
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
    )
    retryable_status_codes: tuple[int, ...] = field(
        default_factory=lambda: (408, 429, 500, 502, 503, 504)
    )
    on_retry: Callable[[int, Exception], None] | None = None


# Default configurations for different scenarios
DEFAULT_API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=60.0,
)

QUICK_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.3,
    max_delay=5.0,
)


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for the given attempt number.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base**attempt)

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.0, delay)


def is_retryable_exception(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exception: The exception to check
        config: Retry configuration

    Returns:
        True if the exception should trigger a retry
    """
    # Check for HTTP response errors with retryable status codes
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status in config.retryable_status_codes

    # Check against retryable exception types
    return isinstance(exception, config.retryable_exceptions)


async def async_retry(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        config: Retry configuration (uses DEFAULT_API_RETRY_CONFIG if None)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        RetryError: When all retry attempts have been exhausted
    """
    if config is None:
        config = DEFAULT_API_RETRY_CONFIG

    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc

            # Check if we should retry
            if not is_retryable_exception(exc, config):
                _LOGGER.debug(
                    "Non-retryable exception encountered: %s",
                    type(exc).__name__,
                )
                raise

            # Check if we have more attempts
            if attempt + 1 >= config.max_attempts:
                _LOGGER.warning(
                    "All %d retry attempts exhausted for %s",
                    config.max_attempts,
                    func.__name__,
                )
                break

            # Calculate delay and wait
            delay = calculate_delay(attempt, config)
            _LOGGER.debug(
                "Retry %d/%d for %s after %.2fs delay (error: %s)",
                attempt + 1,
                config.max_attempts,
                func.__name__,
                delay,
                exc,
            )

            # Call on_retry callback if provided
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, exc)
                except Exception as callback_err:
                    _LOGGER.warning("on_retry callback error: %s", callback_err)

            await asyncio.sleep(delay)

    raise RetryError(
        f"Failed after {config.max_attempts} attempts",
        attempts=config.max_attempts,
        last_exception=last_exception,
    )


def async_retry_with_backoff(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration (uses DEFAULT_API_RETRY_CONFIG if None)

    Returns:
        Decorator function

    Example:
        @async_retry_with_backoff(RetryConfig(max_attempts=5))
        async def fetch_data():
            async with session.get(url) as response:
                return await response.json()
    """
    if config is None:
        config = DEFAULT_API_RETRY_CONFIG

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await async_retry(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


class RetryContext:
    """Context manager for manual retry control.

    Useful when you need more control over the retry logic.

    Example:
        async with RetryContext(config) as ctx:
            while ctx.should_retry:
                try:
                    result = await some_operation()
                    ctx.success()
                    break
                except Exception as e:
                    await ctx.handle_error(e)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize retry context."""
        self.config = config or DEFAULT_API_RETRY_CONFIG
        self.attempt = 0
        self._succeeded = False
        self._last_exception: Exception | None = None

    @property
    def should_retry(self) -> bool:
        """Check if we should attempt another retry."""
        return not self._succeeded and self.attempt < self.config.max_attempts

    def success(self) -> None:
        """Mark the operation as successful."""
        self._succeeded = True

    async def handle_error(self, exception: Exception) -> None:
        """Handle an error and prepare for retry if appropriate.

        Args:
            exception: The exception that occurred

        Raises:
            Exception: If the exception is not retryable
            RetryError: If all attempts are exhausted
        """
        self._last_exception = exception

        if not is_retryable_exception(exception, self.config):
            raise exception

        self.attempt += 1

        if self.attempt >= self.config.max_attempts:
            raise RetryError(
                f"Failed after {self.config.max_attempts} attempts",
                attempts=self.config.max_attempts,
                last_exception=exception,
            )

        delay = calculate_delay(self.attempt - 1, self.config)
        _LOGGER.debug(
            "RetryContext: attempt %d/%d, waiting %.2fs",
            self.attempt,
            self.config.max_attempts,
            delay,
        )

        if self.config.on_retry:
            try:
                self.config.on_retry(self.attempt, exception)
            except Exception as callback_err:
                _LOGGER.warning("on_retry callback error: %s", callback_err)

        await asyncio.sleep(delay)

    async def __aenter__(self) -> "RetryContext":
        """Enter the context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the context."""
        return False


async def retry_on_status_codes(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> aiohttp.ClientResponse:
    """Make an HTTP request with automatic retry on specific status codes.

    Args:
        session: aiohttp ClientSession
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        config: Retry configuration
        **kwargs: Additional arguments for the request

    Returns:
        aiohttp.ClientResponse

    Raises:
        RetryError: When all retry attempts have been exhausted
    """
    if config is None:
        config = DEFAULT_API_RETRY_CONFIG

    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            response = await session.request(method, url, **kwargs)

            # Check if we got a retryable status code
            if response.status in config.retryable_status_codes:
                if attempt + 1 < config.max_attempts:
                    delay = calculate_delay(attempt, config)
                    _LOGGER.debug(
                        "Retrying due to status %d: attempt %d/%d, waiting %.2fs",
                        response.status,
                        attempt + 1,
                        config.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            return response

        except Exception as exc:
            last_exception = exc

            if not is_retryable_exception(exc, config):
                raise

            if attempt + 1 >= config.max_attempts:
                break

            delay = calculate_delay(attempt, config)
            _LOGGER.debug(
                "Retry %d/%d for %s %s after %.2fs delay (error: %s)",
                attempt + 1,
                config.max_attempts,
                method,
                url,
                delay,
                exc,
            )

            await asyncio.sleep(delay)

    raise RetryError(
        f"Failed after {config.max_attempts} attempts",
        attempts=config.max_attempts,
        last_exception=last_exception,
    )
