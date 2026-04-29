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
    CONF_API_KEYS,
    CONF_CHAT_MODEL,
    CONF_CHAT_URL,
    CONF_CUSTOM_API_KEY,
    CONF_PROVIDER_PRESET,
    CONF_LLM_PROVIDER,
    CONF_MODEL_CATALOG,
    CONF_PROVIDER_KEY,
    CHAT_MODEL_EXAMPLES,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_STT_PROVIDER,
    DEFAULT_TTS_PROVIDER,
    LLM_PROVIDER_OPTIONS,

    CONF_IMAGE_MODEL,
    CONF_IMAGE_URL,

    CONF_LLM_HASS_API,
    CONF_MAX_HISTORY_MESSAGES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_STT_MODEL,
    CONF_STT_PROVIDER,
    CONF_STT_URL,

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
    CONF_TTS_MODEL,
    CONF_TTS_PROVIDER,
    CONF_TTS_URL,
    CONF_TTS_VOICE,
    DEFAULT_CHAT_URL,
    DEFAULT_IMAGE_URL,
    DEFAULT_STT_URL,
    DEFAULT_TTS_URL,
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
    IMAGE_MODEL_EXAMPLES,
    PROVIDER_PRESETS,
    PROVIDER_PRESET_OPTIONS,
    STT_MODEL_EXAMPLES,
    STT_PROVIDER_OPTIONS,
    TTS_DEFAULT_LANG,
    TTS_DEFAULT_VOICE,
    TTS_PROVIDER_OPTIONS,
)

from .memory import get_memory_store
from .model_catalog import make_catalog, selector_options, validate_catalog

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Optional(CONF_NAME, default=DEFAULT_TITLE): str,
    vol.Optional(CONF_API_KEY, default=""): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
    vol.Optional(
        CONF_API_KEYS,
        default='{\n  "openai": "",\n  "siliconflow": "",\n  "aliyun": ""\n}',
        description={"suggested_value": "JSON object, for example: {\"openai\":\"sk-...\"}"},
    ): str,
})


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    # Initial setup is provider-neutral. Provider URLs, models, and API keys are
    # validated when each subentry is used.
    return None


def _preset_values_for_subentry(subentry_type: str, preset: str) -> dict[str, Any]:
    """Return provider preset values relevant to one subentry type."""
    raw_preset = PROVIDER_PRESETS.get(preset, {})
    values = {
        CONF_PROVIDER_PRESET: preset if preset in PROVIDER_PRESETS else "custom",
    }
    if preset == "custom":
        return values

    common_keys = (CONF_PROVIDER_KEY,)
    by_subentry = {
        "conversation": (CONF_CHAT_URL, CONF_CHAT_MODEL),
        "ai_task_data": (CONF_IMAGE_URL, CONF_IMAGE_MODEL),
        "tts": (CONF_TTS_PROVIDER, CONF_TTS_URL, CONF_TTS_MODEL, CONF_TTS_VOICE),
        "stt": (CONF_STT_PROVIDER, CONF_STT_URL, CONF_STT_MODEL),
    }
    for key in (*common_keys, *by_subentry.get(subentry_type, ())):
        if key in raw_preset:
            values[key] = raw_preset[key]
    return values


def _apply_provider_preset(
    subentry_type: str,
    options: Mapping[str, Any],
    preset: str,
) -> dict[str, Any]:
    """Apply a provider preset to current form options."""
    updated = dict(options)
    updated.update(_preset_values_for_subentry(subentry_type, preset))
    return updated


class AIHubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Hub."""

    VERSION = 2
    MINOR_VERSION = 3

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return options flow for top-level provider key storage."""
        return AIHubOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
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
            entry_title = user_input.pop(CONF_NAME, DEFAULT_TITLE)
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
                title=entry_title,
                data=user_input,
                subentries=subentries,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
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
    last_rendered_preset: str = "custom"

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

            self.last_rendered_preset = str(
                self.options.get(CONF_PROVIDER_PRESET, "custom") or "custom"
            )

        else:
            requested_preset = str(
                user_input.get(CONF_PROVIDER_PRESET, "custom") or "custom"
            )
            if requested_preset != self.last_rendered_preset:
                self.options = _apply_provider_preset(
                    self._subentry_type,
                    user_input,
                    requested_preset,
                )
                self.last_rendered_preset = requested_preset
            else:
                processed_input = user_input.copy()
                processed_input[CONF_RECOMMENDED] = False

                try:
                    if CONF_MODEL_CATALOG in processed_input:
                        validate_catalog(processed_input.get(CONF_MODEL_CATALOG))
                except (ValueError, TypeError):
                    errors[CONF_MODEL_CATALOG] = "invalid_model_catalog"
                    self.options = processed_input
                else:
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
                        processed_input[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST

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

    if subentry_type == "conversation":
        # Always enable LLM Hass API for conversation, don't show to user
        options[CONF_LLM_HASS_API] = llm.LLM_API_ASSIST
        chat_catalog = options.get(
            CONF_MODEL_CATALOG,
            make_catalog(CHAT_MODEL_EXAMPLES, {CHAT_MODEL_EXAMPLES[0]: "example, edit this list freely"}),
        )

        schema.update({
            vol.Optional(
                CONF_PROVIDER_PRESET,
                default=options.get(CONF_PROVIDER_PRESET, "custom"),
                description={"suggested_value": options.get(CONF_PROVIDER_PRESET)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=PROVIDER_PRESET_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
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
                    options=selector_options(chat_catalog, CHAT_MODEL_EXAMPLES),
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_MODEL_CATALOG,
                default=chat_catalog,
                description={"suggested_value": chat_catalog},
            ): str,
            vol.Optional(
                CONF_CHAT_URL,
                default=options.get(CONF_CHAT_URL, DEFAULT_CHAT_URL),
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
                CONF_PROVIDER_KEY,
                default=options.get(CONF_PROVIDER_KEY, ""),
                description={"suggested_value": options.get(CONF_PROVIDER_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
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
        image_catalog = options.get(
            CONF_MODEL_CATALOG,
            make_catalog(IMAGE_MODEL_EXAMPLES, {"gpt-image-1": "example; edit this list freely"}),
        )
        schema.update({
            vol.Optional(
                CONF_PROVIDER_PRESET,
                default=options.get(CONF_PROVIDER_PRESET, "custom"),
                description={"suggested_value": options.get(CONF_PROVIDER_PRESET)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=PROVIDER_PRESET_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_IMAGE_MODEL,
                default=options.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL),
                description={"suggested_value": options.get(CONF_IMAGE_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=selector_options(image_catalog, IMAGE_MODEL_EXAMPLES),
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_MODEL_CATALOG,
                default=image_catalog,
                description={"suggested_value": image_catalog},
            ): str,
            vol.Optional(
                CONF_IMAGE_URL,
                default=options.get(CONF_IMAGE_URL, DEFAULT_IMAGE_URL),
                description={"suggested_value": options.get(CONF_IMAGE_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_PROVIDER_KEY,
                default=options.get(CONF_PROVIDER_KEY, ""),
                description={"suggested_value": options.get(CONF_PROVIDER_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
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
                CONF_PROVIDER_PRESET,
                default=options.get(CONF_PROVIDER_PRESET, "custom"),
                description={"suggested_value": options.get(CONF_PROVIDER_PRESET)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=PROVIDER_PRESET_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_PROVIDER,
                default=options.get(CONF_TTS_PROVIDER, DEFAULT_TTS_PROVIDER),
                description={"suggested_value": options.get(CONF_TTS_PROVIDER)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=TTS_PROVIDER_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_TTS_MODEL,
                default=options.get(CONF_TTS_MODEL, ""),
                description={"suggested_value": options.get(CONF_TTS_MODEL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_TTS_URL,
                default=options.get(CONF_TTS_URL, DEFAULT_TTS_URL),
                description={"suggested_value": options.get(CONF_TTS_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_PROVIDER_KEY,
                default=options.get(CONF_PROVIDER_KEY, ""),
                description={"suggested_value": options.get(CONF_PROVIDER_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_CUSTOM_API_KEY,
                default=options.get(CONF_CUSTOM_API_KEY, ""),
                description={"suggested_value": options.get(CONF_CUSTOM_API_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
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
        stt_catalog = options.get(
            CONF_MODEL_CATALOG,
            make_catalog(STT_MODEL_EXAMPLES, {"whisper-1": "example; edit freely"}),
        )
        schema.update({
            vol.Optional(
                CONF_PROVIDER_PRESET,
                default=options.get(CONF_PROVIDER_PRESET, "custom"),
                description={"suggested_value": options.get(CONF_PROVIDER_PRESET)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=PROVIDER_PRESET_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_STT_PROVIDER,
                default=options.get(CONF_STT_PROVIDER, DEFAULT_STT_PROVIDER),
                description={"suggested_value": options.get(CONF_STT_PROVIDER)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=STT_PROVIDER_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_STT_MODEL,
                default=options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
                description={"suggested_value": options.get(CONF_STT_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=selector_options(stt_catalog, STT_MODEL_EXAMPLES),
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_MODEL_CATALOG,
                default=stt_catalog,
                description={"suggested_value": stt_catalog},
            ): str,
            vol.Optional(
                CONF_STT_URL,
                default=options.get(CONF_STT_URL, DEFAULT_STT_URL),
                description={"suggested_value": options.get(CONF_STT_URL)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            vol.Optional(
                CONF_PROVIDER_KEY,
                default=options.get(CONF_PROVIDER_KEY, ""),
                description={"suggested_value": options.get(CONF_PROVIDER_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            vol.Optional(
                CONF_CUSTOM_API_KEY,
                default=options.get(CONF_CUSTOM_API_KEY, ""),
                description={"suggested_value": options.get(CONF_CUSTOM_API_KEY)},
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        })
    

    return schema



class AIHubOptionsFlowHandler(OptionsFlow):
    """Handle options flow for AI Hub."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options for the custom component."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current_api_keys = (
            self.config_entry.options.get(CONF_API_KEYS)
            or self.config_entry.data.get(CONF_API_KEYS)
            or '{\n  "openai": "",\n  "siliconflow": "",\n  "aliyun": ""\n}'
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_API_KEYS,
                    default=current_api_keys,
                    description={"suggested_value": current_api_keys},
                ): str,
            }),
        )
