"""Conversation agent support for AI Hub integration.

This module implements the ConversationEntity for AI-powered
dialogue interactions, supporting:
- Streaming responses
- Tool calling (Home Assistant control)
- Image understanding (vision models)
- Context-aware conversations
- Three-tier intent processing (local → HA built-in → LLM)"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_LLM_HASS_API, CONF_PROMPT, DOMAIN
from .entity import AIHubBaseLLMEntity
from .intents import get_config_cache

_LOGGER = logging.getLogger(__name__)

MATCH_ALL: Literal["*"] = "*"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    _LOGGER.debug("Setting up conversation entities, subentries: %s", config_entry.subentries)

    if not config_entry.subentries:
        _LOGGER.warning("No subentries found in config entry")
        return

    for subentry in config_entry.subentries.values():
        _LOGGER.debug("Processing subentry: %s, type: %s", subentry.subentry_id, subentry.subentry_type)
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [AIHubConversationAgent(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )
        _LOGGER.debug("Created conversation agent for subentry: %s", subentry.subentry_id)


class AIHubConversationAgent(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    AIHubBaseLLMEntity,
):
    """AI Hub conversation agent."""

    _attr_supports_streaming = True

    def __init__(
        self, entry: ConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the agent."""
        from .const import RECOMMENDED_CHAT_MODEL

        super().__init__(entry, subentry, RECOMMENDED_CHAT_MODEL)

        # 初始化配置缓存
        self._config_cache = get_config_cache()

        # Enable control feature if LLM Hass API is configured
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a sentence and return a response.

        处理流程:
        1. 本地意图处理（全局设备控制等 HA 原生不支持的功能）
        2. 如果本地没匹配，交给 Home Assistant 处理（内置意图 + LLM）
        """
        options = self.subentry.data

        # ========== 步骤1: 本地意图处理 ==========
        # 只处理 HA 原生不支持的功能（如全局设备控制）

        # 1a. 检查是否是自动化创建请求
        try:
            automation_result = await self._handle_automation_request(user_input)
            if automation_result:
                return automation_result
        except Exception as e:
            _LOGGER.debug("Automation handling failed: %s", e)

        # 1b. 检查是否需要本地意图处理 (全局设备控制)
        try:
            from .intents import get_global_intent_handler
            intent_handler = get_global_intent_handler(self.hass)

            if intent_handler and intent_handler.should_handle(user_input.text):
                _LOGGER.debug("Local intent processing: %s", user_input.text)
                intent_result = await intent_handler.handle(user_input.text, user_input.language)

                if intent_result:
                    _LOGGER.debug("Local intent completed")
                    return conversation.ConversationResult(
                        response=intent_result["response"],
                        conversation_id=user_input.conversation_id
                    )
        except Exception as e:
            _LOGGER.debug("Local intent handling failed: %s", e)

        # ========== 步骤2: 尝试 HA 内置意图处理 ==========
        # timer、shopping list、设备控制等 HA 原生支持的意图
        try:
            from homeassistant.components import conversation as ha_conversation

            _LOGGER.debug("调用 HA 内置 intents 处理: %s", user_input.text)
            ha_response = await ha_conversation.async_handle_intents(
                self.hass,
                user_input,
                chat_log,
            )

            _LOGGER.debug("HA intents 返回结果: %s", ha_response)

            if ha_response is not None:
                response_type = ha_response.response_type
                _LOGGER.debug("HA intents response_type: %s", response_type)

                has_error = hasattr(ha_response, "error") and ha_response.error
                is_error_type = response_type == intent.IntentResponseType.ERROR
                is_no_match = (
                    response_type == intent.IntentResponseType.NO_INTENT_MATCHED
                    if hasattr(intent.IntentResponseType, "NO_INTENT_MATCHED")
                    else False
                )
                response_has_content = bool(ha_response.speech and ha_response.speech.get("plain"))

                if not has_error and not is_error_type and not is_no_match and response_has_content:
                    _LOGGER.info("HA 内置意图处理成功: %s, type: %s", user_input.text, response_type)
                    return conversation.ConversationResult(
                        response=ha_response,
                        conversation_id=chat_log.conversation_id,
                    )

                _LOGGER.debug(
                    "HA 内置意图未匹配或返回错误(has_error=%s, is_error_type=%s, is_no_match=%s)，交给 LLM 处理",
                    has_error,
                    is_error_type,
                    is_no_match,
                )

        except Exception as e:
            _LOGGER.warning("HA 内置意图处理异常: %s", e, exc_info=True)

        # ========== 步骤3: LLM 处理 ==========

        try:
            # High-performance LLM API config retrieval
            llm_apis = options.get(CONF_LLM_HASS_API, [])

            if not llm_apis:
                _LOGGER.warning("LLM API config not found, checking LLM_API_ASSIST")
                # Try using the default LLM API ASSIST
                try:
                    default_llm_api = llm.LLM_API_ASSIST
                    llm_apis = [default_llm_api]
                    _LOGGER.debug("Using default LLM API")
                except Exception as e:
                    _LOGGER.error(f"LLM_API_ASSIST retrieval failed: {e}")
                    empty_response = intent.IntentResponse(language=user_input.language)
                    error_msg = self._config_cache.get_error_message("llm_config_error")
                    empty_response.async_set_speech(error_msg)
                    return conversation.ConversationResult(
                        response=empty_response,
                        conversation_id=user_input.conversation_id
                    )

            # Provide LLM data (tools, home info, etc.)
            user_prompt = options.get(CONF_PROMPT, "")
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_apis,
                user_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            _LOGGER.error(f"LLM conversation error: {err}")
            return err.as_conversation_result()

        # Process the chat log with AI Hub
        # Loop to handle tool calls: model may call tools, then we need to call again with results
        loop_count = 0
        while True:
            loop_count += 1
            _LOGGER.debug("LLM processing loop %d", loop_count)

            # Check chat_log status
            if hasattr(chat_log, 'tool_calls') and chat_log.tool_calls:
                _LOGGER.debug("LLM initiated %d tool calls", len(chat_log.tool_calls))
            else:
                _LOGGER.debug("LLM did not initiate tool calls")

            await self._async_handle_chat_log(chat_log)

            # Check if there are tool call results
            if hasattr(chat_log, 'unresponded_tool_results') and chat_log.unresponded_tool_results:
                _LOGGER.debug("Found unprocessed tool call results")
            else:
                _LOGGER.debug("No more tool calls, processing completed")

            # If there are no unresponded tool results, continue the loop
            if not chat_log.unresponded_tool_results:
                break

        # Apply markdown filtering to the final assistant message before returning
        from .markdown_filter import filter_markdown_content
        if chat_log.content and len(chat_log.content) > 0:
            last_content = chat_log.content[-1]
            if last_content.role == "assistant" and last_content.content:
                original_content = str(last_content.content)
                filtered_content = filter_markdown_content(original_content)
                if filtered_content != original_content:
                    _LOGGER.debug(
                        "Filtered markdown from chat_log before returning: '%s' -> '%s'",
                        original_content[:50] if len(original_content) > 50 else original_content,
                        filtered_content[:50] if len(filtered_content) > 50 else filtered_content
                    )
                    chat_log.content[-1] = replace(last_content, content=filtered_content)

        # Return result from chat log
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    def _is_device_operation(self, tool_name: str) -> bool:
        """Check if this is a device control operation"""
        try:
            from .intents import is_device_operation
            return is_device_operation(tool_name)
        except Exception as e:
            _LOGGER.debug(f"Device operation check failed: {e}")
            return False

    async def _handle_automation_request(
        self,
        user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult | None:
        """Handle automation creation requests."""
        user_text = user_input.text.lower()

        # Use config cache to get automation keywords
        try:
            automation_keywords = self._config_cache.get_automation_config('automation_keywords', [])

            if not automation_keywords:
                _LOGGER.debug("No automation_keywords found in config")
                return None

        except Exception as e:
            _LOGGER.debug("Failed to load automation keywords: %s", e)
            return None

        if not any(keyword in user_text for keyword in automation_keywords):
            return None

        try:
            from .ai_automation import get_automation_manager
            manager = get_automation_manager(self.hass)

            # Extract automation description
            description = self._extract_automation_description(user_input.text)
            if not description:
                return None

            # Create automation
            result = await manager.create_automation_from_description(description)

            # Create response
            intent_response = intent.IntentResponse(language=user_input.language)

            if result.get("success", False):
                config_data = result.get("config", {})
                automation_name = config_data.get("alias", "新自动化")

                # Use configured success message
                responses = self._config_cache.get_responses_config()
                success_template = responses.get('automation', {}).get('creation_success')
                if success_template:
                    try:
                        message = success_template.format(automation_name=automation_name)
                    except KeyError:
                        message = success_template
                else:
                    message = f"我已经为您创建了自动化: {automation_name}"

                # Can add more configuration details
                if config_data.get("trigger"):
                    unknown_platform = self._config_cache.get_error_message("automation_trigger_unknown")
                    triggers = [t.get("platform", unknown_platform) for t in config_data["trigger"]]
                    message += f"\\n触发条件: {', '.join(triggers)}"

                intent_response.async_set_speech(message)
            else:
                # Use configured error message
                responses = self._config_cache.get_responses_config()
                error_template = responses.get('automation', {}).get('creation_error')
                if error_template:
                    try:
                        error_msg = error_template.format(error=result.get('error', '未知错误'))
                    except KeyError:
                        error_msg = error_template
                else:
                    error_msg = f"创建自动化失败: {result.get('error', '未知错误')}"

                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    error_msg
                )

            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id
            )

        except Exception as e:
            _LOGGER.error("Error handling automation request: %s", e)
            return None

    def _extract_automation_description(self, user_text: str) -> str | None:
        """Extract automation description from user input."""
        # Use config cache to get automation prefixes
        try:
            prefixes = self._config_cache.get_automation_config('automation_prefixes', [])

        except Exception as e:
            _LOGGER.debug("Failed to load automation prefixes: %s", e)
            prefixes = []

        description = user_text
        for prefix in prefixes:
            if prefix in description:
                description = description.replace(prefix, "").strip()

        # If description is too short or empty, return None
        if len(description) < 5:
            return None

        return description

    def _is_local_special_function(self, intent_type: str, intent_info: dict[str, Any]) -> bool:
        """Check if this is a true local special function"""
        # Check if this is an "all devices" operation (not natively supported by HA)
        if self._is_all_device_operation(intent_info):
            return True

        # Check for other special functions that need local processing
        # Can dynamically determine based on configuration or patterns
        return self._has_local_intent_config(intent_type, intent_info)

    def _is_all_device_operation(self, intent_info: dict[str, Any]) -> bool:
        """Check if this is an 'all devices' operation"""
        text = intent_info.get("text", "").lower()

        # Use config cache to get global keywords, avoid hardcoding
        try:
            global_keywords = self._config_cache.get_global_keywords()
        except Exception as e:
            _LOGGER.debug(f"Failed to read global_keywords, using defaults: {e}")
            global_keywords = ["所有", "全部", "一切"]

        return any(keyword in text for keyword in global_keywords)

    def _has_local_intent_config(self, intent_type: str, intent_info: dict[str, Any]) -> bool:
        """Check if intent is defined in local config as needing special handling"""
        text = intent_info.get("text", "").lower()

        # Use config cache to get local feature keywords, avoid hardcoding
        try:
            local_features = self._config_cache.get_local_features()
            _LOGGER.debug(f"Loaded local feature keywords from config cache: {len(local_features)} items")
        except Exception as e:
            _LOGGER.debug(f"Failed to read local config, using defaults: {e}")
            local_features = ["所有设备", "全部设备", "所有灯", "全部灯"]

        return any(feature in text for feature in local_features)

    def _should_skip_ha_standard_processing(self, text: str) -> bool:
        """Check whether to skip Home Assistant standard processing and go directly to local intent

        Only very explicit local intents skip HA standard processing, such as:
        - "Turn on all devices"
        - "Turn off all lights"
        """
        try:
            config = self._config_cache.get_config()
            if not config:
                return False

            # Get config from local_intents.GlobalDeviceControl
            local_intents = config.get('local_intents', {})
            global_config = local_intents.get('GlobalDeviceControl', {})
            if not global_config:
                return False

            text_lower = text.lower().strip()

            # Check global keywords
            global_keywords = global_config.get('global_keywords', [])
            has_global = any(keyword in text_lower for keyword in global_keywords)

            # Check explicit on/off keywords
            on_keywords = global_config.get('on_keywords', [])
            off_keywords = global_config.get('off_keywords', [])
            has_action = any(keyword in text_lower for keyword in on_keywords + off_keywords)

            # Check if contains parameter control keywords (brightness, volume, etc.)
            param_keywords = global_config.get('param_keywords', [])
            brightness_keywords = global_config.get('brightness_keywords', [])
            volume_keywords = global_config.get('volume_keywords', [])
            color_keywords = global_config.get('color_keywords', [])
            temperature_keywords = global_config.get('temperature_keywords', [])

            has_param_control = any(keyword in text_lower for keyword in
                                    param_keywords + brightness_keywords + volume_keywords +
                                    color_keywords + temperature_keywords)

            # Skip HA processing: only explicit global commands go to local processing
            # Requires both global keywords and explicit action or parameter control
            should_skip = has_global and (has_action or has_param_control)

            if should_skip:
                _LOGGER.debug(
                    "Skipping HA processing: '%s' (global=%s, action=%s, param=%s)",
                    text, has_global, has_action, has_param_control
                )

            return should_skip

        except Exception as e:
            _LOGGER.debug(f"Error checking whether to skip HA processing: {e}")
            return False
