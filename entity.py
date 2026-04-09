"""Base entity for AI Hub integration."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

import aiohttp
from homeassistant.components import conversation, media_source
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_entry_flow, llm
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity
from homeassistant.util import ulid
from voluptuous_openapi import convert

from .const import (
    AI_HUB_CHAT_URL,
    CONF_CHAT_MODEL,
    CONF_CHAT_URL,
    CONF_CUSTOM_API_KEY,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_LONG_MEMORY_ENABLED,
    CONF_LONG_MEMORY_MAX_CHARS,
    CONF_LONG_MEMORY_PINNED,
    CONF_LONG_MEMORY_UPDATE_TURNS,
    CONF_LLM_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    DOMAIN,
    ERROR_GETTING_RESPONSE,
    RECOMMENDED_IMAGE_ANALYSIS_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_LONG_MEMORY_ENABLED,
    RECOMMENDED_LONG_MEMORY_MAX_CHARS,
    RECOMMENDED_LONG_MEMORY_UPDATE_TURNS,
)
from .markdown_filter import filter_markdown_streaming
from .memory import get_memory_store
from .providers import LLMMessage, LLMStreamDelta, create_provider

_LOGGER = logging.getLogger(__name__)


def _ensure_string(value: Any) -> str:
    """Ensure a value is a valid string for API calls.

    Args:
        value: The value to convert to string

    Returns:
        A string representation of the value, or empty string if None/empty
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        # Convert complex types to JSON string
        return json.dumps(value, ensure_ascii=False)
    # For other types, convert to string
    return str(value)


class _AIHubEntityMixin:
    """Mixin class providing common initialization logic for AI Hub entities.

    This mixin provides shared initialization behavior for all AI Hub entity types:
    - LLM conversation entities
    - TTS entities
    - STT entities

    Attributes:
        entry: The config entry
        subentry: The subentry for this entity instance
        default_model: The default model to use
        _api_key: The API key for this entity
    """

    _attr_has_entity_name = False
    _attr_should_poll = False

    def _initialize_aihub_entity(
        self,
        entry: config_entry_flow.ConfigEntry,
        subentry: config_entry_flow.ConfigSubentry,
        default_model: str,
    ) -> None:
        """Initialize common AI Hub entity attributes.

        This method should be called from the entity's __init__ method.

        Args:
            entry: The config entry
            subentry: The subentry for this entity instance
            default_model: The default model to use
        """
        self.entry = entry
        self.subentry = subentry
        self.default_model = default_model
        self._attr_unique_id = subentry.subentry_id
        self._attr_name = subentry.title

        # Get API key: use custom key if provided, otherwise use main key
        custom_key_raw = subentry.data.get(CONF_CUSTOM_API_KEY, "")
        custom_key = str(custom_key_raw).strip() if custom_key_raw else ""
        main_key = entry.runtime_data if entry.runtime_data else ""
        # Ensure API key is always a string
        if custom_key:
            self._api_key = custom_key
        elif isinstance(main_key, str) and main_key.strip():
            self._api_key = main_key
        else:
            self._api_key = ""
            _LOGGER.warning("No valid API key found for entity %s", subentry.title)

    def _get_device_model(self, default_model: str) -> str:
        """Get the device model for device info.

        Can be overridden by subclasses to provide custom model validation.

        Args:
            default_model: The default model to use if no model is configured

        Returns:
            The model name to use for device info
        """
        return self.subentry.data.get(CONF_CHAT_MODEL, default_model)

    def _create_device_info(self, domain: str) -> dr.DeviceInfo:
        """Create device info for this entity.

        Args:
            domain: The domain for this integration

        Returns:
            DeviceInfo object
        """
        return dr.DeviceInfo(
            identifiers={(domain, self.subentry.subentry_id)},
            name=self.subentry.title,
            manufacturer="老王杂谈说",
            model=self._get_device_model(self.default_model),
            entry_type=dr.DeviceEntryType.SERVICE,
        )


class AIHubBaseLLMEntity(Entity, _AIHubEntityMixin):
    """Base entity for AI Hub LLM."""

    def __init__(
        self,
        entry: config_entry_flow.ConfigEntry,
        subentry: config_entry_flow.ConfigSubentry,
        default_model: str,
    ) -> None:
        """Initialize the entity."""
        # Use mixin initialization
        self._initialize_aihub_entity(entry, subentry, default_model)
        # Create device info using mixin method
        self._attr_device_info = self._create_device_info(DOMAIN)

    def _get_device_model(self, default_model: str) -> str:
        """Get the device model with validation for LLM entities.

        Args:
            default_model: The default model to use if no model is configured

        Returns:
            The validated model name to use for device info
        """
        device_model = self.subentry.data.get(CONF_CHAT_MODEL) or default_model
        if not isinstance(device_model, str) or not device_model.strip():
            device_model = default_model
        return device_model

    def _get_model_config(self, chat_log: conversation.ChatLog | None = None) -> dict[str, Any]:
        """Get model configuration from options."""
        options = self.subentry.data
        configured_model = options.get(CONF_CHAT_MODEL) or self.default_model
        # Ensure configured_model is a valid string
        if not isinstance(configured_model, str) or not configured_model.strip():
            configured_model = self.default_model

        # Check if we need to switch to vision model
        final_model = configured_model
        if chat_log:
            # Detect if any content has attachments
            has_attachments = any(
                hasattr(content, 'attachments') and content.attachments
                for content in chat_log.content
            )

            # Check if attachments contain images/videos
            has_media_attachments = False
            if has_attachments:
                for content in chat_log.content:
                    if hasattr(content, 'attachments') and content.attachments:
                        for attachment in content.attachments:
                            mime_type = getattr(attachment, 'mime_type', '')
                            if mime_type.startswith(('image/', 'video/')):
                                has_media_attachments = True
                                break
                    if has_media_attachments:
                        break

            # Auto-switch to vision model if needed (prefer free model!)
            if has_media_attachments:
                vision_models = ["glm-4.1v-thinking", "glm-4v-flash"]
                if configured_model not in vision_models:
                    final_model = RECOMMENDED_IMAGE_ANALYSIS_MODEL  # GLM-4.1V-Thinking
                    _LOGGER.debug(
                        "Auto-switching to vision model %s for media attachments (original: %s)",
                        final_model,
                        configured_model)

        # Only use parameters that the working service uses (top_p causes API error!)
        # Ensure model is always a valid string
        model_value = final_model or self.default_model
        if not isinstance(model_value, str) or not model_value.strip():
            model_value = self.default_model

        return {
            "model": model_value,
            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        }
    def _get_llm_provider_type(self) -> str:
        """Get configured LLM provider type."""
        provider_type = self.subentry.data.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        if not isinstance(provider_type, str) or not provider_type.strip():
            return DEFAULT_LLM_PROVIDER
        return provider_type.strip()
    def _convert_messages_to_llm_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[LLMMessage]:
        """Convert internal message dicts to provider LLMMessage objects."""
        llm_messages: list[LLMMessage] = []

        for msg in messages:
            llm_messages.append(
                LLMMessage(
                    role=str(msg.get("role", "")),
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                )
            )

        return llm_messages
    def _get_pinned_memory_text(self) -> str:
        """Render pinned memory from conversation config."""
        raw = self.subentry.data.get(CONF_LONG_MEMORY_PINNED, "")
        if not isinstance(raw, str) or not raw.strip():
            return ""

        items = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•").strip()
            if line:
                items.append(line)

        if not items:
            return ""

        return "Pinned memory from configuration:\n" + "\n".join(f"- {item}" for item in items)
        
    async def _async_get_long_memory_message(self) -> dict[str, Any] | None:
        """Build a system message for long-term memory."""
        options = self.subentry.data
        enabled = options.get(CONF_LONG_MEMORY_ENABLED, RECOMMENDED_LONG_MEMORY_ENABLED)
        if not enabled:
            return None

        max_chars = options.get(CONF_LONG_MEMORY_MAX_CHARS, RECOMMENDED_LONG_MEMORY_MAX_CHARS)
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = RECOMMENDED_LONG_MEMORY_MAX_CHARS

        store = get_memory_store(self.hass)
        store_text = await store.async_get_memory_block(max_chars=max_chars)
        pinned_text = self._get_pinned_memory_text()

        parts = []
        if pinned_text:
            parts.append(pinned_text)
        if store_text:
            parts.append(store_text)

        if not parts:
            return None

        return {
            "role": "system",
            "content": "\n\n".join(parts).strip(),
        }
    
    def _build_recent_memory_snippet(self, chat_log: conversation.ChatLog, limit: int = 12) -> str:
        """Build a compact text snippet from recent chat turns for memory summarization."""
        items: list[str] = []

        for content in chat_log.content:
            role = getattr(content, "role", "")
            if role not in ("user", "assistant"):
                continue

            text = _ensure_string(getattr(content, "content", "")).strip()
            if not text:
                continue

            prefix = "User" if role == "user" else "Assistant"
            items.append(f"{prefix}: {text}")

        if not items:
            return ""
        return "\n".join(items[-limit:])
    
    async def _async_generate_memory_summary(self, source_text: str, max_chars: int) -> str:
        """Generate a concise long-term memory summary from recent dialogue."""
        if not source_text.strip():
            return ""

        api_url = self.subentry.data.get(CONF_CHAT_URL) or AI_HUB_CHAT_URL
        if not isinstance(api_url, str) or not api_url.strip():
            api_url = AI_HUB_CHAT_URL

        model_config = self._get_model_config()
        model_name = model_config.get("model") or self.default_model
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = self.default_model

        prompt = (
            "Extract durable long-term memory from the dialogue below.\n"
            "Only keep information that is likely useful in future conversations, such as:\n"
            "- stable user preferences\n"
            "- recurring home automation habits\n"
            "- device naming conventions or aliases\n"
            "- persistent projects or ongoing context\n"
            "\n"
            "Do NOT keep:\n"
            "- one-time control commands\n"
            "- temporary device states\n"
            "- execution results\n"
            "- troubleshooting details that are only relevant right now\n"
            "- casual small talk\n"
            "\n"
            f"If there is no durable memory worth storing, return exactly: EMPTY\n"
            f"Otherwise return plain text only, under {max_chars} characters."
        )

        request_params = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_text},
            ],
            "max_tokens": 300,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        session = async_get_clientsession(self.hass)
        async with session.post(
            api_url,
            json=request_params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HomeAssistantError(f"Memory summary request failed: {error_text}")

            data = await response.json()

        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            _LOGGER.warning("Unexpected memory summary response format: %s", data)
            return ""

        return _ensure_string(content).strip()[:max_chars]
    
    async def _async_merge_memory_summary(self, old_summary: str, new_summary: str, max_chars: int) -> str:
        """Merge existing long-term summary with a new summary chunk."""
        old_summary = old_summary.strip()
        new_summary = new_summary.strip()

        if not old_summary:
            return new_summary[:max_chars]
        if not new_summary:
            return old_summary[:max_chars]

        api_url = self.subentry.data.get(CONF_CHAT_URL) or AI_HUB_CHAT_URL
        if not isinstance(api_url, str) or not api_url.strip():
            api_url = AI_HUB_CHAT_URL

        model_config = self._get_model_config()
        model_name = model_config.get("model") or self.default_model
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = self.default_model

        prompt = (
            "Merge the existing long-term memory summary with the new memory summary.\n"
            "Keep only stable, reusable information for future conversations.\n"
            "Remove redundancy, temporary device states, one-time commands, and execution results.\n"
            f"Return plain text only, under {max_chars} characters.\n"
            "If nothing valuable remains, return exactly: EMPTY"
        )

        source_text = (
            f"Existing long-term summary:\n{old_summary}\n\n"
            f"New summary:\n{new_summary}"
        )

        request_params = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_text},
            ],
            "max_tokens": 300,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        session = async_get_clientsession(self.hass)
        async with session.post(
            api_url,
            json=request_params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HomeAssistantError(f"Memory merge request failed: {error_text}")

            data = await response.json()

        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            _LOGGER.warning("Unexpected memory merge response format: %s", data)
            return new_summary[:max_chars]

        return _ensure_string(content).strip()[:max_chars]
    
    async def _async_update_long_memory(self, chat_log: conversation.ChatLog) -> None:
        """Update persistent long-term memory after a successful response."""
        options = self.subentry.data
        enabled = options.get(CONF_LONG_MEMORY_ENABLED, RECOMMENDED_LONG_MEMORY_ENABLED)
        if not enabled:
            return

        update_turns = options.get(
            CONF_LONG_MEMORY_UPDATE_TURNS,
            RECOMMENDED_LONG_MEMORY_UPDATE_TURNS,
        )
        if not isinstance(update_turns, int) or update_turns <= 0:
            update_turns = RECOMMENDED_LONG_MEMORY_UPDATE_TURNS

        max_chars = options.get(
            CONF_LONG_MEMORY_MAX_CHARS,
            RECOMMENDED_LONG_MEMORY_MAX_CHARS,
        )
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = RECOMMENDED_LONG_MEMORY_MAX_CHARS

        store = get_memory_store(self.hass)
        memory = await store.async_increment_turn_count()

        if memory.turn_count % update_turns != 0:
            return

        source_text = self._build_recent_memory_snippet(chat_log)
        if not source_text:
            return

        new_summary = await self._async_generate_memory_summary(source_text, max_chars)
        if not new_summary:
            return

        new_summary = new_summary.strip()
        if new_summary.upper() == "EMPTY":
            return

        await store.async_set_conversation_summary(new_summary)

        merged_summary = await self._async_merge_memory_summary(
            memory.global_summary,
            new_summary,
            max_chars,
        )
        if merged_summary.strip().upper() == "EMPTY":
            await store.async_set_global_summary("")
            return
        await store.async_set_global_summary(merged_summary)          
    
    async def _transform_provider_stream(
        self,
        stream: AsyncGenerator[LLMStreamDelta, None],
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Transform provider stream events into Home Assistant chat log deltas."""
        yielded_assistant_role = False

        async for delta in stream:
            if not yielded_assistant_role:
                yield {"role": "assistant"}
                yielded_assistant_role = True

            if delta.content:
                yield {"content": filter_markdown_streaming(delta.content)}

            if delta.tool_calls:
                tool_inputs = []
                for tool_call in delta.tool_calls:
                    tool_id = tool_call.get("id")
                    if not tool_id or not isinstance(tool_id, str) or not tool_id.strip():
                        tool_id = ulid.ulid_now()

                    arguments = tool_call.get("arguments", "")
                    try:
                        parsed_args = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
                    except json.JSONDecodeError:
                        parsed_args = {}

                    tool_inputs.append(
                        llm.ToolInput(
                            id=tool_id,
                            tool_name=tool_call.get("name", ""),
                            tool_args=parsed_args,
                        )
                    )

                if tool_inputs:
                    yield {"tool_calls": tool_inputs}
                                 
    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: dict[str, Any] | None = None,
    ) -> None:
        """Generate an answer for the chat log via configured LLM provider."""
        options = self.subentry.data
        model_config = self._get_model_config(chat_log)

        # Build messages from chat log (attachment processing will be done during conversion)
        messages = await self._async_convert_chat_log_to_messages(chat_log)

        # Add JSON format instruction to system message if structure is requested
        if structure and messages:
            for i, message in enumerate(messages):
                if message.get("role") == "system":
                    original_content = message.get("content", "")
                    if "JSON" not in original_content:
                        message["content"] = (
                            original_content
                            + "\n\nWhen providing structured data like automation names/"
                            "descriptions, respond ONLY with valid JSON. Use the exact "
                            "JSON structure requested in the prompt. Do not include any "
                            "markdown formatting, explanations, or additional text."
                        )
                    break

        # Add tools if available
        tools = []
        if chat_log.llm_api:
            tools.extend(
                [
                    self._format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]
            )

        # Validate model name
        model_name = model_config.get("model", "")
        if not model_name or not isinstance(model_name, str):
            model_name = self.default_model
            _LOGGER.warning("Model name was invalid, using default: %s", model_name)

        # Validate API key before making request
        if not self._api_key:
            _LOGGER.error("Cannot make provider request: API key is empty or not configured")
            raise HomeAssistantError("API key is not configured")

        if not isinstance(self._api_key, str):
            self._api_key = str(self._api_key)

        # Get provider type from config
        provider_type = self._get_llm_provider_type()

        # Convert internal message dicts to provider messages
        llm_messages = self._convert_messages_to_llm_messages(messages)

        # Build provider config
        provider_config = {
            "api_key": self._api_key,
            "base_url": options.get(CONF_CHAT_URL) or AI_HUB_CHAT_URL,
            "model": model_name,
        }

        _LOGGER.debug(
            "LLM Provider Request: provider=%s, model=%s, messages_count=%d, tools_count=%d",
            provider_type,
            model_name,
            len(llm_messages),
            len(tools),
        )

        try:
            provider = create_provider(provider_type, provider_config)
        except Exception as err:
            _LOGGER.error("Failed to create LLM provider %s: %s", provider_type, err)
            raise HomeAssistantError(ERROR_GETTING_RESPONSE) from err

        try:
            provider_stream = provider.complete_stream(
                llm_messages,
                tools=tools if tools else None,
                temperature=model_config.get("temperature"),
                max_tokens=model_config.get("max_tokens"),
            )

            [
                content
                async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    self._transform_provider_stream(provider_stream),
                )
            ]

            # Update long-term memory only after a successful full response
            try:
                await self._async_update_long_memory(chat_log)
            except Exception as err:
                _LOGGER.warning("Failed to update long-term memory: %s", err)

        except aiohttp.ClientError as err:
            _LOGGER.error("Network error calling provider %s: %s", provider_type, err)
            raise HomeAssistantError(f"{ERROR_GETTING_RESPONSE}: Network error") from err
        except Exception as err:
            _LOGGER.error("Error calling provider %s: %s", provider_type, err)
            raise HomeAssistantError(ERROR_GETTING_RESPONSE) from err
    
    async def _async_convert_chat_log_to_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert chat log to AI Hub message format."""
        options = self.subentry.data
        max_history = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)

        if not chat_log.content:
            return []

        last_content = chat_log.content[-1]
        messages: list[dict[str, Any]] = []
        system_messages: list[dict[str, Any]] = []

        # Collect original system messages first
        for content in chat_log.content:
            if content.role == "system":
                system_messages.append({
                    "role": "system",
                    "content": _ensure_string(content.content),
                })

        # Inject long-term memory as an extra system message
        long_memory_message = await self._async_get_long_memory_message()
        if long_memory_message:
            system_messages.append(long_memory_message)

        # Attachment mode: keep system + memory + current message
        if last_content.role == "user" and last_content.attachments:
            _LOGGER.debug("Using attachment mode with system and long-term memory")
            return system_messages + [await self._convert_user_message(last_content)]

        # Track tool_call IDs for matching with tool results
        # Maps: original_id -> generated_id (if we had to generate one)
        # Also tracks: index -> id for tool_calls without original ids
        tool_call_id_map: dict[str, str] = {}
        last_tool_call_ids: list[str] = []

        # Standard conversation handling for messages without attachments
        # First message is system message (index 0)
        # History is content[1:-1] (excluding first system and last user input)
        # Last message is current user input (index -1)


        # Process history messages (excluding system and last user input)
        history_content = chat_log.content[1:-1] if len(chat_log.content) > 1 else []

        # Build history messages with ID tracking
        history_messages = []
        for content in history_content:
            if content.role == "user":
                history_messages.append(await self._convert_user_message(content))
            elif content.role == "assistant":
                msg, generated_ids = self._convert_assistant_message_with_id_tracking(content, tool_call_id_map)
                history_messages.append(msg)
                last_tool_call_ids = generated_ids
            elif content.role == "tool_result":
                msg = self._convert_tool_message_with_id_matching(content, tool_call_id_map, last_tool_call_ids)
                history_messages.append(msg)

        # Limit history: keep only the most recent conversation turns
        # Count user messages to determine conversation turns
        if max_history > 0:
            user_message_count = sum(1 for msg in history_messages if msg.get("role") == "user")
            if user_message_count > max_history:
                # Find the index to start keeping messages
                # We want to keep the last max_history user turns and their associated messages
                user_count = 0
                start_index = len(history_messages)
                for i in range(len(history_messages) - 1, -1, -1):
                    if history_messages[i].get("role") == "user":
                        user_count += 1
                        if user_count >= max_history:
                            start_index = i
                            break
                history_messages = history_messages[start_index:]

        # Add system + long-term memory messages first
        messages.extend(system_messages)
        
        # Add history to messages
        messages.extend(history_messages)

        # Add current user input (with ID tracking for consistency)
        if last_content.role == "user":
            messages.append(await self._convert_user_message(last_content))
        elif last_content.role == "assistant":
            msg, _ = self._convert_assistant_message_with_id_tracking(last_content, tool_call_id_map)
            messages.append(msg)
        elif last_content.role == "tool_result":
            msg = self._convert_tool_message_with_id_matching(last_content, tool_call_id_map, last_tool_call_ids)
            messages.append(msg)

        return messages

    async def _convert_user_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert user message to AI Hub format."""
        message: dict[str, Any] = {"role": "user"}

        if not content.attachments:
            message["content"] = _ensure_string(content.content)
            return message

        # Process attachments
        successful_images = await self._process_attachments(content.attachments)

        if successful_images:
            # Build content with images first, then text (like working services.py)
            parts = successful_images + [{"type": "text", "text": _ensure_string(content.content)}]
            message["content"] = parts
            _LOGGER.debug(
                "Final message content has %d parts (%d images + text)",
                len(parts), len(successful_images)
            )
        else:
            _LOGGER.warning("No images were processed successfully, falling back to text only")
            message["content"] = _ensure_string(content.content)

        return message

    async def _process_attachments(
        self, attachments: list[Any]
    ) -> list[dict[str, Any]]:
        """Process attachments and return list of successful image parts."""
        successful_images = []
        _LOGGER.debug("Processing %d attachments for user message", len(attachments))

        for i, attachment in enumerate(attachments):
            _LOGGER.debug("Processing attachment %d: %s", i, attachment)

            if not (attachment.mime_type and attachment.mime_type.startswith("image/")):
                _LOGGER.debug(
                    "Skipping non-image attachment: %s (mime: %s)",
                    attachment, getattr(attachment, 'mime_type', 'unknown')
                )
                continue

            image_data = await self._get_image_data_from_attachment(attachment)
            if image_data:
                successful_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
                _LOGGER.debug("Successfully added image to message parts")
            else:
                _LOGGER.warning("Could not get image data from attachment: %s", attachment)

        return successful_images

    async def _get_image_data_from_attachment(self, attachment: Any) -> str | None:
        """Extract base64 image data from an attachment."""
        try:
            mime_type = attachment.mime_type

            # Strategy 1: Direct file path (most reliable)
            if hasattr(attachment, 'path') and attachment.path:
                return await self._read_image_from_path(attachment.path)

            # Strategy 2: Media content ID
            if hasattr(attachment, 'media_content_id') and attachment.media_content_id:
                _LOGGER.debug("Attachment has media_content_id: %s", attachment.media_content_id)
                image_bytes = await self._async_get_media_content(
                    attachment.media_content_id, mime_type
                )
                if image_bytes:
                    image_data = base64.b64encode(image_bytes).decode()
                    _LOGGER.debug("Successfully resolved media content, base64 length: %d", len(image_data))
                    return image_data
                _LOGGER.warning("Failed to resolve media content for: %s", attachment.media_content_id)
                return None

            # Strategy 3: Direct content
            if hasattr(attachment, 'content') and attachment.content:
                if isinstance(attachment.content, bytes):
                    image_data = base64.b64encode(attachment.content).decode()
                    _LOGGER.debug("Converted bytes to base64, length: %d", len(image_data))
                    return image_data
                elif isinstance(attachment.content, str):
                    _LOGGER.debug("Using existing base64 content, length: %d", len(attachment.content))
                    return attachment.content

            _LOGGER.warning("Attachment format not supported: %s", attachment)
            return None

        except Exception as err:
            _LOGGER.error("Failed to process image attachment %s: %s", attachment, err, exc_info=True)
            return None

    async def _read_image_from_path(self, path: str) -> str | None:
        """Read image from file path and return base64 encoded data."""
        import asyncio

        try:
            _LOGGER.debug("Reading file directly from path: %s", path)
            image_bytes = await asyncio.to_thread(self._read_file_bytes, str(path))
            image_data = base64.b64encode(image_bytes).decode()
            _LOGGER.debug("Successfully read file directly, base64 length: %d", len(image_data))
            return image_data
        except Exception as err:
            _LOGGER.error("Failed to read file %s: %s", path, err, exc_info=True)
            return None

    def _convert_assistant_message_with_id_tracking(
        self, content: conversation.Content, id_map: dict[str, str]
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert assistant message with ID tracking for tool calls.

        Returns:
            Tuple of (message dict, list of tool_call IDs used in this message)
        """
        generated_ids: list[str] = []

        # Use base conversion
        message = self._convert_assistant_message(content)

        # Extract and track the IDs that were used
        if content.tool_calls and "tool_calls" in message:
            for i, tc in enumerate(message["tool_calls"]):
                tool_id = tc["id"]
                generated_ids.append(tool_id)
                # Map original ID (if any) to the used ID
                original_id = content.tool_calls[i].id if content.tool_calls[i].id else None
                if original_id:
                    id_map[original_id] = tool_id
                else:
                    id_map[f"_index_{i}"] = tool_id

        return message, generated_ids

    def _convert_tool_message_with_id_matching(
        self, content: conversation.Content, id_map: dict[str, str], last_tool_call_ids: list[str]
    ) -> dict[str, Any]:
        """Convert tool result message with ID matching.

        Tries to match the tool_call_id with IDs from the most recent assistant message.
        """
        original_id = content.tool_call_id

        # First use base conversion
        message = self._convert_tool_message(content)

        # Then try to fix the ID if needed
        if original_id and isinstance(original_id, str) and original_id.strip():
            # Check if this ID needs remapping
            if original_id in id_map:
                message["tool_call_id"] = id_map[original_id]
        elif last_tool_call_ids:
            # No valid original ID - use the first available ID from last assistant's tool_calls
            message["tool_call_id"] = last_tool_call_ids[0]
            # Remove it so the next tool result uses the next ID
            if len(last_tool_call_ids) > 1:
                last_tool_call_ids.pop(0)

        return message

    def _convert_assistant_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert assistant message to AI Hub format."""
        message: dict[str, Any] = {"role": "assistant"}

        if content.tool_calls:
            tool_calls_list = []
            for tool_call in content.tool_calls:
                # Ensure tool_call id is always a valid non-empty string
                tool_id = tool_call.id if tool_call.id else None
                if not tool_id or not isinstance(tool_id, str) or not tool_id.strip():
                    tool_id = ulid.ulid_now()

                tool_calls_list.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": str(tool_call.tool_name) if tool_call.tool_name else "",
                        "arguments": (
                            json.dumps(tool_call.tool_args, ensure_ascii=False)
                            if tool_call.tool_args
                            else "{}"
                        ),
                    },
                })
            message["tool_calls"] = tool_calls_list
            message["content"] = _ensure_string(content.content)
        else:
            message["content"] = _ensure_string(content.content)

        return message

    def _convert_tool_message(
        self, content: conversation.Content
    ) -> dict[str, Any]:
        """Convert tool result to AI Hub format."""
        # Ensure tool_call_id is always a valid string
        # API requires a non-empty tool_call_id
        tool_call_id = content.tool_call_id
        if not tool_call_id or not isinstance(tool_call_id, str) or not tool_call_id.strip():
            # Generate a valid ID if missing - this maintains API compatibility
            tool_call_id = ulid.ulid_now()
            _LOGGER.debug("Generated tool_call_id for tool result: %s", tool_call_id)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": (
                json.dumps(content.tool_result, ensure_ascii=False, default=str)
                if content.tool_result is not None
                else "{}"
            ),
        }

    def _format_tool(
        self, tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
    ) -> dict[str, Any]:
        """Format tool for AI Hub API."""
        # Ensure tool name and description are valid strings
        tool_name = str(tool.name) if tool.name else ""
        tool_description = str(tool.description) if tool.description else ""

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": self._convert_schema(tool.parameters, custom_serializer),
            },
        }

    def _convert_schema(
        self, schema: dict[str, Any], custom_serializer: Callable[[Any], Any] | None
    ) -> dict[str, Any]:
        """Convert schema to AI Hub format."""
        # AI Hub uses standard JSON Schema
        # Use voluptuous_openapi to convert the schema properly
        try:
            return convert(
                schema,
                custom_serializer=custom_serializer if custom_serializer else llm.selector_serializer,
            )
        except Exception as err:
            _LOGGER.warning("Failed to convert schema with custom_serializer: %s", err)
            # Fall back to basic conversion without custom_serializer
            try:
                return convert(schema, custom_serializer=llm.selector_serializer)
            except Exception:
                # If all else fails, return as-is
                return schema

    async def _transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Transform AI Hub SSE stream into HA format."""
        buffer = ""
        tool_call_buffer: dict[int, dict[str, Any]] = {}
        has_started = False

        async for chunk in response.content:
            if not chunk:
                continue

            chunk_text = chunk.decode("utf-8", errors="ignore")
            buffer += chunk_text

            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # Skip empty lines and end markers
                if not line or line == "data: [DONE]":
                    continue

                # Process SSE data lines
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if not data_str.strip():
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        _LOGGER.debug("SSE data parse failed: %s", data_str)
                        continue

                    if not data.get("choices"):
                        continue

                    delta = data["choices"][0].get("delta", {})

                    # Start assistant message if not started
                    if not has_started:
                        yield {"role": "assistant"}
                        has_started = True

                    # Handle content delta
                    if "content" in delta and delta["content"]:
                        # Filter markdown from content using streaming filter to preserve spaces
                        filtered_content = filter_markdown_streaming(delta["content"])
                        yield {"content": filtered_content}

                    # Handle tool calls
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)

                            # Initialize tool call buffer if needed
                            if index not in tool_call_buffer:
                                # Ensure we always have a valid id
                                tool_id = tc_delta.get("id")
                                if not tool_id or not isinstance(tool_id, str) or not tool_id.strip():
                                    tool_id = ulid.ulid_now()
                                tool_call_buffer[index] = {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    },
                                }

                            # Update tool call data - only update id if it's a valid non-empty string
                            if (
                                "id" in tc_delta
                                and tc_delta["id"]
                                and isinstance(tc_delta["id"], str)
                                and tc_delta["id"].strip()
                            ):
                                tool_call_buffer[index]["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                func = tc_delta["function"]
                                if "name" in func:
                                    tool_call_buffer[index]["function"]["name"] = func["name"]
                                if "arguments" in func:
                                    tool_call_buffer[index]["function"]["arguments"] += func["arguments"]

        # Yield final tool calls if any
        if tool_call_buffer:
            tool_calls = []
            for tc in tool_call_buffer.values():
                try:
                    # Ensure id is valid before creating ToolInput
                    tool_id = tc["id"]
                    if not tool_id or not isinstance(tool_id, str) or not tool_id.strip():
                        tool_id = ulid.ulid_now()

                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                    tool_calls.append(
                        llm.ToolInput(
                            id=tool_id,
                            tool_name=tc["function"]["name"],
                            tool_args=args,
                        )
                    )
                except json.JSONDecodeError as err:
                    _LOGGER.warning("Failed to parse tool call arguments: %s", err)

            if tool_calls:
                yield {"tool_calls": tool_calls}

    async def _async_download_image_from_url(self, url: str) -> bytes | None:
        """Download image from URL."""
        try:
            session = async_get_clientsession(self.hass)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    _LOGGER.warning("Failed to download image from URL: %s, status: %s", url, response.status)
                    return None
        except Exception as err:
            _LOGGER.warning("Error downloading image from URL %s: %s", url, err)
            return None

    async def _async_get_media_content(self, media_content_id: str, mime_type: str) -> bytes | None:
        """Get media content from Home Assistant media source."""
        _LOGGER.debug("Getting media content for ID: %s, mime_type: %s", media_content_id, mime_type)

        try:
            # Route to appropriate handler based on URL format
            if media_content_id.startswith("media-source://"):
                return await self._resolve_media_source(media_content_id)
            elif media_content_id.startswith(("/api/image/serve/", "api/image/serve/")):
                return await self._fetch_served_image(media_content_id)
            elif media_content_id.startswith(("http://", "https://")):
                _LOGGER.debug("Processing direct URL: %s", media_content_id)
                return await self._async_download_image_from_url(media_content_id)
            else:
                _LOGGER.warning("Unsupported media content ID format: %s", media_content_id)
                return None

        except Exception as err:
            _LOGGER.error("Unexpected error getting media content %s: %s", media_content_id, err, exc_info=True)
            return None

    async def _resolve_media_source(self, media_content_id: str) -> bytes | None:
        """Resolve and download content from a media-source:// URL."""
        _LOGGER.debug("Processing media-source URL: %s", media_content_id)

        if not media_source.is_media_source_id(media_content_id):
            _LOGGER.warning("Invalid media source ID: %s", media_content_id)
            return None

        # Resolve media source
        try:
            media_item = await media_source.async_resolve_media(
                self.hass, media_content_id, self.entity_id
            )
            _LOGGER.debug("Resolved media item: %s", media_item)
        except Exception as err:
            _LOGGER.error("Error resolving media source %s: %s", media_content_id, err, exc_info=True)
            return None

        if not (media_item and hasattr(media_item, 'url') and media_item.url):
            _LOGGER.warning("Could not resolve media source or no URL: %s", media_content_id)
            return None

        # Build full URL if it's a relative path
        media_url = self._build_full_url(media_item.url)
        _LOGGER.debug("Media item URL: %s", media_url)

        return await self._download_from_url(media_url)

    def _build_full_url(self, url: str) -> str:
        """Build full URL from a potentially relative path."""
        if not url.startswith('/'):
            return url

        try:
            if hasattr(self.hass.config, 'external_url') and self.hass.config.external_url:
                base_url = self.hass.config.external_url.rstrip('/')
            elif hasattr(self.hass.config, 'internal_url') and self.hass.config.internal_url:
                base_url = self.hass.config.internal_url.rstrip('/')
            else:
                base_url = "http://localhost:8123"
        except Exception as err:
            _LOGGER.warning("Could not get Home Assistant URL, using localhost: %s", err)
            base_url = "http://localhost:8123"

        return f"{base_url}{url}"

    async def _fetch_served_image(self, url: str) -> bytes | None:
        """Fetch image from /api/image/serve/ endpoint."""
        if not url.startswith("/"):
            url = f"/{url}"

        _LOGGER.debug("Processing serve URL: %s", url)
        return await self._download_from_url(url)

    async def _download_from_url(self, url: str) -> bytes | None:
        """Download content from a URL using Home Assistant's session."""
        try:
            session = async_get_clientsession(self.hass)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                _LOGGER.debug("Response status: %s", response.status)
                if response.status == 200:
                    content = await response.read()
                    _LOGGER.debug("Successfully downloaded content, size: %d bytes", len(content))
                    return content
                else:
                    _LOGGER.warning("Failed to download from %s, status: %s", url, response.status)
                    return None
        except Exception as err:
            _LOGGER.error("Error downloading from %s: %s", url, err, exc_info=True)
            return None

    def _read_file_bytes(self, file_path: str) -> bytes:
        """Read file bytes synchronously (to be used with asyncio.to_thread)."""
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as err:
            raise Exception(f"Failed to read file {file_path}: {err}")


class AIHubEntityBase(Entity, _AIHubEntityMixin):
    """Base entity for AI Hub integration.

    This class is used by TTS and STT entities which don't need the full
    LLM functionality but require the same initialization logic.
    """

    def __init__(
        self,
        entry: config_entry_flow.ConfigEntry,
        subentry: config_entry_flow.ConfigSubentry,
        default_model: str,
    ) -> None:
        """Initialize the entity."""
        # Use mixin initialization
        self._initialize_aihub_entity(entry, subentry, default_model)
        # Create device info using mixin method
        self._attr_device_info = self._create_device_info(DOMAIN)
