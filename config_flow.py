"""Config flow for AI Hub integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    AI_HUB_CHAT_MODELS,
    AI_HUB_CHAT_URL,
    AI_HUB_IMAGE_GEN_URL,
    AI_HUB_IMAGE_MODELS,
    CONF_CHAT_MODEL,
    CONF_CHAT_URL,
    CONF_CUSTOM_API_KEY,
    CONF_LLM_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDER_OPTIONS,

    CONF_IMAGE_MODEL,
    CONF_IMAGE_URL,

    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_STT_MODEL,

    CONF_LONG_MEMORY_ENABLED,
    CONF_LONG_MEMORY_UPDATE_TURNS,
    CONF_LONG_MEMORY_MAX_CHARS,
    CONF_LONG_MEMORY_PINNED,
    CONF_LONG_MEMORY_GLOBAL,
    CONF_LONG_MEMORY_CONVERSATION,
    RECOMMENDED_LONG_MEMORY_ENABLED,
    RECOMMENDED_LONG_MEMORY_UPDATE_TURNS,
    RECOMMENDED_LONG_MEMORY_MAX_CHARS,
    RECOMMENDED_LONG_MEMORY_PINNED,
    
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TTS_LANG,
    CONF_TTS_VOICE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TITLE,

    DEFAULT_TTS_NAME,
    DOMAIN,
    EDGE_TTS_VOICES,
    RECOMMENDED_AI_TASK_MAX_TOKENS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_AI_TASK_TEMPERATURE,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_MAX_HISTORY_MESSAGES,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_K,

    RECOMMENDED_TTS_OPTIONS,
    SILICONFLOW_STT_MODELS,
    TTS_DEFAULT_LANG,
    TTS_DEFAULT_VOICE,
)

from .memory import get_memory_store

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required(CONF_API_KEY): str,
})


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    # Only validate API key if it's provided
    if CONF_API_KEY in data and data[CONF_API_KEY].strip():
        headers = {
            "Authorization": f"Bearer {data[CONF_API_KEY]}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                AI_HUB_CHAT_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 401:
                    raise ValueError("invalid_auth")
                if response.status != 200:
                    await response.text()  # Read response but don't use it
                    raise ValueError("cannot_connect")


class AIHubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Hub."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
                description_placeholders={
                    "api_key_url": "https://cloud.siliconflow.cn/account/ak"
                },
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except ValueError as err:
            reason = str(err)
            if reason in {"invalid_auth", "cannot_connect"}:
                errors["base"] = reason
            else:
                _LOGGER.exception("Unexpected validation error: %s", err)
                errors["base"] = "unknown"
        except aiohttp.ClientError:
            _LOGGER.exception("Cannot connect")
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            # Create entry with subentries
            subentries = [
                {
                    "subentry_type": "conversation",
                    "data": RECOMMENDED_CONVERSATION_OPTIONS,
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": "ai_task_data",
                    "data": RECOMMENDED_AI_TASK_OPTIONS,
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": "tts",
                    "data": RECOMMENDED_TTS_OPTIONS,
                    "title": DEFAULT_TTS_NAME,
                    "unique_id": None,
                },
                {
                    "subentry_type": "stt",
                    "data": RECOMMENDED_STT_OPTIONS,
                    "title": DEFAULT_STT_NAME,
                    "unique_id": None,
                },                
            ]

            return self.async_create_entry(
                title=DEFAULT_TITLE,
                data=user_input,
                subentries=subentries,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders={
                "api_key_url": "https://cloud.siliconflow.cn/account/ak"
            },
        )

    @classmethod
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": AIHubSubentryFlowHandler,
            "ai_task_data": AIHubSubentryFlowHandler,
            "tts": AIHubSubentryFlowHandler,
            "stt": AIHubSubentryFlowHandler,
        }


class AIHubSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for conversation and AI task."""

    options: dict[str, Any]
    last_rendered_recommended: bool = False

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle options for subentry."""
        errors: dict[str, str] = {}

        if user_input is None:
            # First render: get current options
            if self._is_new:
                if self._subentry_type == "ai_task_data":
                    self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
                elif self._subentry_type == "tts":
                    self.options = RECOMMENDED_TTS_OPTIONS.copy()
                elif self._subentry_type == "stt":
                    self.options = RECOMMENDED_STT_OPTIONS.copy()
                else:
                    self.options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
            else:
                # If reconfiguration, copy existing options to show current values
                self.options = self._get_reconfigure_subentry().data.copy()

            if self._subentry_type == "conversation":
                store = get_memory_store(self.hass)
                memory = await store.async_get()
                self.options[CONF_LONG_MEMORY_GLOBAL] = memory.global_summary or ""
                self.options[CONF_LONG_MEMORY_CONVERSATION] = memory.conversation_summary or ""

            self.last_rendered_recommended = self.options.get(CONF_RECOMMENDED, True)

        else:
            # Check if recommended mode has changed
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                # Recommended mode unchanged, save the configuration
                # Use user input directly (no complex model name processing needed)
                processed_input = user_input.copy()

                # Conversation memory fields are stored in storage, not subentry data
                if self._subentry_type == "conversation":
                    store = get_memory_store(self.hass)

                    global_memory = processed_input.get(CONF_LONG_MEMORY_GLOBAL, "")
                    conversation_memory = processed_input.get(CONF_LONG_MEMORY_CONVERSATION, "")

                    await store.async_set_global_summary(
                        global_memory if isinstance(global_memory, str) else ""
                    )
                    await store.async_set_conversation_summary(
                        conversation_memory if isinstance(conversation_memory, str) else ""
                    )

                    processed_input.pop(CONF_LONG_MEMORY_GLOBAL, None)
                    processed_input.pop(CONF_LONG_MEMORY_CONVERSATION, None)

                    # Always enable LLM_HASS_API for conversation
                    processed_input[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST

                # Update or create subentry
                if self._is_new:
                    return self.async_create_entry(
                        title=processed_input.pop(CONF_NAME),
                        data=processed_input,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=processed_input,
                )

            # Recommended mode changed, re-render form with new options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
            self.options.update(user_input)  # Update current options with user input

        # Build schema based on current options

        if self._subentry_type == "conversation":
            store = get_memory_store(self.hass)
            memory = await store.async_get()
            self.options[CONF_LONG_MEMORY_GLOBAL] = memory.global_summary or ""
            self.options[CONF_LONG_MEMORY_CONVERSATION] = memory.conversation_summary or ""
            
        schema = await ai_hub_config_option_schema(
            self._is_new, self._subentry_type, self.options
        )
                
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async_step_reconfigure = async_step_init
    async_step_user = async_step_init


async def ai_hub_config_option_schema(
    is_new: bool,
    subentry_type: str,
    options: Mapping[str, Any],
) -> dict:
    """Return a schema for AI Hub completion options."""

    schema = {}

    # Add name field for new entries
    if is_new:
        if CONF_NAME in options:
            default_name = options[CONF_NAME]
        elif subentry_type == "ai_task_data":
            default_name = DEFAULT_AI_TASK_NAME
        elif subentry_type == "tts":
            default_name = DEFAULT_TTS_NAME
        elif subentry_type == "stt":
            default_name = DEFAULT_STT_NAME
        else:
            default_name = DEFAULT_CONVERSATION_NAME
        schema[vol.Required(CONF_NAME, default=default_name)] = str

    # Add recommended mode toggle
    schema[
        vol.Required(CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, True))
    ] = bool

    # If recommended mode is enabled, only show basic fields
    if options.get(CONF_RECOMMENDED):
        # In recommended mode, only show prompt for conversation
        if subentry_type == "conversation":
            schema.update({
                vol.Optional(
                    CONF_PROMPT,
                    default=options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    description={"suggested_value": options.get(CONF_PROMPT)},
                ): TemplateSelector(),
            })
            # Ensure LLM Hass API is always enabled in recommended mode
            options[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST
        elif subentry_type == "tts":
            # In recommended mode, no configuration options shown - use defaults
            pass
        elif subentry_type == "stt":
            # In recommended mode, no configuration options needed
            pass        
        return schema

    # Show advanced options only when not in recommended mode
    if subentry_type == "conversation":
        # Always enable LLM Hass API for conversation, don't show to user
        options[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST

        schema.update({
            vol.Optional(
                CONF_PROMPT,
                default=options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                description={"suggested_value": options.get(CONF_PROMPT)},
            ): TemplateSelector(),
            vol.Optional(
                CONF_CHAT_MODEL,
                default=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=AI_HUB_CHAT_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_CHAT_URL,
                default=options.get(CONF_CHAT_URL, AI_HUB_CHAT_URL),
                description={"suggested_value": options.get(CONF_CHAT_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_LLM_PROVIDER,
                default=options.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
                description={"suggested_value": options.get(CONF_LLM_PROVIDER)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=LLM_PROVIDER_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_CUSTOM_API_KEY,
                default=options.get(CONF_CUSTOM_API_KEY, ""),
                description={"suggested_value": options.get(CONF_CUSTOM_API_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_TOP_K,
                default=options.get(CONF_TOP_K, RECOMMENDED_TOP_K),
                description={"suggested_value": options.get(CONF_TOP_K)},
            ): int,
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            ): int,
            vol.Optional(
                CONF_MAX_HISTORY_MESSAGES,
                default=options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES),
                description={"suggested_value": options.get(CONF_MAX_HISTORY_MESSAGES)},
            ): int,
            vol.Optional(
                CONF_LONG_MEMORY_ENABLED,
                default=options.get(CONF_LONG_MEMORY_ENABLED, RECOMMENDED_LONG_MEMORY_ENABLED),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_ENABLED)},
            ): bool,

            vol.Optional(
                CONF_LONG_MEMORY_UPDATE_TURNS,
                default=options.get(CONF_LONG_MEMORY_UPDATE_TURNS, RECOMMENDED_LONG_MEMORY_UPDATE_TURNS),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_UPDATE_TURNS)},
            ): int,

            vol.Optional(
                CONF_LONG_MEMORY_MAX_CHARS,
                default=options.get(CONF_LONG_MEMORY_MAX_CHARS, RECOMMENDED_LONG_MEMORY_MAX_CHARS),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_MAX_CHARS)},
            ): int,

            vol.Optional(
                CONF_LONG_MEMORY_PINNED,
                default=options.get(CONF_LONG_MEMORY_PINNED, RECOMMENDED_LONG_MEMORY_PINNED),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_PINNED)},
            ): TemplateSelector(),
            
            vol.Optional(
                CONF_LONG_MEMORY_GLOBAL,
                default=options.get(CONF_LONG_MEMORY_GLOBAL, ""),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_GLOBAL)},
            ): TemplateSelector(),

            vol.Optional(
                CONF_LONG_MEMORY_CONVERSATION,
                default=options.get(CONF_LONG_MEMORY_CONVERSATION, ""),
                description={"suggested_value": options.get(CONF_LONG_MEMORY_CONVERSATION)},
            ): TemplateSelector(),
        })

    elif subentry_type == "ai_task_data":
        # AI Task follows Conversation Agent's chat model and URL
        schema.update({
            vol.Optional(
                CONF_IMAGE_MODEL,
                default=options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL),
                description={"suggested_value": options.get(CONF_IMAGE_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=AI_HUB_IMAGE_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_IMAGE_URL,
                default=options.get(CONF_IMAGE_URL, AI_HUB_IMAGE_GEN_URL),
                description={"suggested_value": options.get(CONF_IMAGE_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_CUSTOM_API_KEY,
                default=options.get(CONF_CUSTOM_API_KEY, ""),
                description={"suggested_value": options.get(CONF_CUSTOM_API_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, RECOMMENDED_AI_TASK_TEMPERATURE),
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=2, step=0.01, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, RECOMMENDED_AI_TASK_MAX_TOKENS),
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            ): int,
        })

    elif subentry_type == "tts":
        # Simple TTS configuration to avoid potential issues
        # Use basic string selectors for now to isolate the problem

        # Create language options from unique languages in EDGE_TTS_VOICES
        unique_languages = sorted(list(set(EDGE_TTS_VOICES.values())))

        schema.update({
            vol.Optional(
                CONF_TTS_LANG,
                default=options.get(CONF_TTS_LANG, TTS_DEFAULT_LANG),
                description={"suggested_value": options.get(CONF_TTS_LANG)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=unique_languages,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_VOICE,
                default=options.get(CONF_TTS_VOICE, TTS_DEFAULT_VOICE),
                description={"suggested_value": options.get(CONF_TTS_VOICE)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
        })

    elif subentry_type == "stt":
        schema.update({
            vol.Optional(
                CONF_STT_MODEL,
                default=options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
                description={"suggested_value": options.get(CONF_STT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=SILICONFLOW_STT_MODELS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        })
    

    return schema



class AIHubOptionsFlowHandler(OptionsFlow):
    """Handle options flow for AI Hub."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options for the custom component."""
        return self.async_abort(reason="configure_via_subentries")
