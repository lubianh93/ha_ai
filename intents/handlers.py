"""意图处理器 - 本地意图和中文意图处理."""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from .loader import get_global_config
from .command_classifier import classify_global_control_command

_LOGGER = logging.getLogger(__name__)


def get_local_intents_config() -> dict[str, Any] | None:
    """获取本地意图配置."""
    config = get_global_config()
    if not config:
        return None
    return config.get('local_intents', {})


class ChineseIntentHandler(intent.IntentHandler):
    """中文意图处理器 - 仅作为后备选项."""

    def __init__(self, hass: HomeAssistant, intent_name: str, sentence: str):
        self.hass = hass
        self.intent_type = intent_name
        self.sentence = sentence

    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        """处理意图 - 委托给Home Assistant的原生意图处理."""
        try:
            response = intent.IntentResponse(language="zh-CN")
            response.async_set_speech("好的，正在处理您的请求")
            return response

        except Exception as e:
            _LOGGER.error(f"Intent handling failed {self.intent_type}: {e}")
            response = intent.IntentResponse()
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"意图处理失败: {str(e)}"
            )
            return response


class LocalIntentHandler:
    """本地意图处理器 - 处理全局设备控制等本地意图."""

    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        self._config = None  # 延迟加载
        self._local_config = None
        self._local_sentence_patterns: list[re.Pattern[str]] | None = None
        
    @property
    def local_config(self):
        """延迟加载本地意图配置."""
        if self._local_config is None:
            self._local_config = get_local_intents_config()
        return self._local_config

    @property
    def config(self):
        """延迟加载全局配置."""
        if self._config is None:
            self._config = get_global_config()
        return self._config

    def _get_default_area_name(self) -> str:
        """获取默认区域名称."""
        global_config = self.local_config.get('GlobalDeviceControl', {}) if self.local_config else {}
        return global_config.get('default_area_name', '全屋')

    def _format_error_suffix(self, error_count: int) -> str:
        """格式化错误后缀消息."""
        if error_count > 0:
            return f"，{error_count}个设备失败"
        return ""

    def should_handle(self, text: str) -> bool:
        """判断是否应该使用本地意图处理。

        本地处理只接管明确的、命令式的全局控制。
        如果配置了 local_sentence_templates，还必须命中句式白名单。
        """
        if not self.local_config:
            return False

        global_config = self.local_config.get("GlobalDeviceControl", {})
        if not global_config:
            return False

        decision = classify_global_control_command(text, global_config)
        if not decision.should_execute_locally:
            _LOGGER.debug(
                "Local intent rejected by classifier: text=%r, kind=%s, reason=%s",
                text,
                decision.kind,
                decision.reason,
            )
            return False

        # 如果配置了句式白名单，则必须命中白名单。
        # 如果暂时没配置，则保留 classifier 的结果，避免一升级就把本地控制全关掉。
        if self._has_local_sentence_templates():
            matched = self._matches_local_sentence_template(text.lower().strip())
            _LOGGER.debug(
                "Local intent sentence-template gate: text=%r, matched=%s",
                text,
                matched,
            )
            return matched

        _LOGGER.debug(
            "Local intent accepted by classifier without sentence templates: text=%r, reason=%s",
            text,
            decision.reason,
        )
        return True
    def _has_local_sentence_templates(self) -> bool:
        """Return whether local sentence templates are configured."""
        config = self.config or {}
        templates = config.get("local_sentence_templates", [])
        return isinstance(templates, list) and bool(templates)


    def _matches_local_sentence_template(self, text_lower: str) -> bool:
        """Return whether text matches any configured local sentence template."""
        patterns = self._get_local_sentence_patterns()
        if not patterns:
            return True

        normalized_text = re.sub(r"\s+", " ", text_lower).strip()
        return any(pattern.fullmatch(normalized_text) for pattern in patterns)


    def _get_local_sentence_patterns(self) -> list[re.Pattern[str]]:
        """Compile local sentence templates into regex patterns once."""
        if self._local_sentence_patterns is not None:
            return self._local_sentence_patterns

        config = self.config or {}
        templates = config.get("local_sentence_templates", [])

        if not isinstance(templates, list):
            self._local_sentence_patterns = []
            return self._local_sentence_patterns

        patterns: list[re.Pattern[str]] = []

        for template in templates:
            if not isinstance(template, str) or not template.strip():
                continue

            try:
                regex = self._template_to_regex(template, config)
                patterns.append(re.compile(regex))
            except ValueError as err:
                _LOGGER.debug("Invalid local sentence template %r: %s", template, err)
                continue
            except re.error as err:
                _LOGGER.debug("Invalid local sentence regex from template %r: %s", template, err)
                continue

        self._local_sentence_patterns = patterns
        return self._local_sentence_patterns


    def _template_to_regex(
        self,
        template: str,
        config: dict[str, Any],
        seen_tokens: set[str] | None = None,
    ) -> str:
        """Convert a sentence template into an anchored regex."""
        seen_tokens = set() if seen_tokens is None else set(seen_tokens)

        def _convert(fragment: str) -> str:
            parts: list[str] = []
            i = 0

            while i < len(fragment):
                char = fragment[i]

                if char == "[":
                    end = self._find_matching(fragment, i, "[", "]")
                    inner = _convert(fragment[i + 1:end])
                    parts.append(f"(?:{inner})?")
                    i = end + 1
                    continue

                if char == "<":
                    end = self._find_matching(fragment, i, "<", ">")
                    token = fragment[i + 1:end].strip()
                    parts.append(self._expand_template_token(token, config, seen_tokens=seen_tokens))
                    i = end + 1
                    continue

                if char == "{":
                    end = self._find_matching(fragment, i, "{", "}")
                    token = fragment[i + 1:end].split(":", 1)[0].strip()
                    parts.append(
                        self._expand_template_token(
                            token,
                            config,
                            fallback=".+?",
                            seen_tokens=seen_tokens,
                        )
                    )
                    i = end + 1
                    continue

                if char.isspace():
                    parts.append(r"\s*")
                elif char in "()|":
                    parts.append(char)
                else:
                    parts.append(re.escape(char))

                i += 1

            return "".join(parts)

        body = _convert(template.strip())
        return rf"^\s*{body}[\s。！？?!.，,:：]*$"


    def _expand_template_token(
        self,
        token: str,
        config: dict[str, Any],
        fallback: str = ".+?",
        seen_tokens: set[str] | None = None,
    ) -> str:
        """Expand a rule/list token into regex."""
        seen_tokens = set() if seen_tokens is None else set(seen_tokens)

        local_intents = config.get("local_intents", {}) if isinstance(config, dict) else {}
        global_config = local_intents.get("GlobalDeviceControl", {}) if isinstance(local_intents, dict) else {}

        # Built-in aliases for local global-control templates.
        builtin_values: dict[str, list[str]] = {
            "turn_on": global_config.get("on_keywords", []),
            "turn_off": global_config.get("off_keywords", []),
            "global_scope": global_config.get("global_keywords", []),
            "parameter": (
                global_config.get("param_keywords", [])
                + global_config.get("brightness_keywords", [])
                + global_config.get("volume_keywords", [])
                + global_config.get("color_keywords", [])
                + global_config.get("temperature_keywords", [])
            ),
        }

        if token == "device_type":
            builtin_values[token] = self._get_device_type_template_values(global_config, config)

        if token in builtin_values:
            return self._values_to_regex(builtin_values[token], fallback)

        expansion_rules = config.get("expansion_rules", {}) if isinstance(config, dict) else {}
        if token in expansion_rules:
            if token in seen_tokens:
                return fallback

            next_seen = set(seen_tokens)
            next_seen.add(token)

            rule = expansion_rules[token]
            if isinstance(rule, str):
                return f"(?:{self._template_to_regex_fragment(rule, config, next_seen)})"

            if isinstance(rule, list):
                return self._values_to_regex([str(item) for item in rule], fallback)

        lists_config = config.get("lists", {}) if isinstance(config, dict) else {}
        list_config = lists_config.get(token)

        values: list[str] = []
        if isinstance(list_config, dict):
            raw_values = list_config.get("values", [])
            if isinstance(raw_values, list):
                values = [str(item) for item in raw_values]
        elif isinstance(list_config, list):
            values = [str(item) for item in list_config]

        return self._values_to_regex(values, fallback)


    def _template_to_regex_fragment(
        self,
        fragment: str,
        config: dict[str, Any],
        seen_tokens: set[str] | None = None,
    ) -> str:
        """Convert a reusable expansion-rule fragment to regex body."""
        regex = self._template_to_regex(fragment, config, seen_tokens)

        prefix = r"^\s*"
        suffix = r"[\s。！？?!.，,:：]*$"

        if regex.startswith(prefix):
            regex = regex[len(prefix):]
        if regex.endswith(suffix):
            regex = regex[: -len(suffix)]

        return regex


    def _find_matching(self, text: str, start: int, opening: str, closing: str) -> int:
        """Find the matching closing character for a template token."""
        depth = 0
        for index in range(start, len(text)):
            if text[index] == opening:
                depth += 1
            elif text[index] == closing:
                depth -= 1
                if depth == 0:
                    return index

        raise ValueError(f"Unmatched {opening!r} in template: {text}")


    def _values_to_regex(self, values: list[str], fallback: str = ".+?") -> str:
        """Convert a list of literal values to regex."""
        cleaned = sorted({value.strip() for value in values if value and value.strip()}, key=len, reverse=True)
        if not cleaned:
            return fallback
        return "(?:" + "|".join(re.escape(value) for value in cleaned) + ")"


    def _get_device_type_template_values(
        self,
        global_config: dict[str, Any],
        config: dict[str, Any],
    ) -> list[str]:
        """Return device-type words allowed in local sentence templates."""
        values: list[str] = []

        device_type_keywords = global_config.get("device_type_keywords", {})
        if isinstance(device_type_keywords, dict):
            values.extend(str(keyword) for keyword in device_type_keywords.keys())

        lists_config = config.get("lists", {}) if isinstance(config, dict) else {}
        for list_name in (
            "light_names",
            "switch_names",
            "climate_names",
            "fan_names",
            "cover_names",
            "media_names",
            "media_player_names",
            "lock_names",
            "vacuum_names",
            "valve_names",
            "camera_names",
        ):
            list_config = lists_config.get(list_name)
            if isinstance(list_config, dict):
                raw_values = list_config.get("values", [])
                if isinstance(raw_values, list):
                    values.extend(str(item) for item in raw_values)

        return values
    async def handle(self, text: str, language: str = "zh-CN"):
        """处理本地意图."""
        if not self.should_handle(text):
            return None

        global_config = self.local_config.get('GlobalDeviceControl', {})
        text_lower = text.lower().strip()

        # 1. 检查是否为参数控制命令
        param_result = await self._handle_parameter_control(text, text_lower, global_config)
        if param_result:
            return param_result

        # 2. 解析操作类型
        on_keywords = global_config.get('on_keywords', [])
        off_keywords = global_config.get('off_keywords', [])

        is_on = any(keyword in text_lower for keyword in on_keywords)
        is_off = any(keyword in text_lower for keyword in off_keywords)

        if not (is_on or is_off):
            return None

        # 3. 解析设备类型和区域
        area_names, device_types = self._parse_device_and_area(text_lower, global_config)

        if not device_types:
            return None

        # 4. 执行设备控制
        return await self._execute_control(
            area_names, device_types, is_on, global_config, language
        )

    def _parse_device_and_area(self, text_lower: str, global_config: dict) -> tuple:
        """解析设备类型和区域."""
        area_names = []
        device_types = []

        # 使用缓存的配置
        config = self.config

        # 获取区域配置
        try:
            if config and 'lists' in config:
                areas_config = config['lists'].get('area_names', {}).get('values', [])
                for area in areas_config:
                    if area in text_lower:
                        area_names.append(area)
        except Exception:
            pass

        # 获取设备类型
        device_type_keywords = global_config.get('device_type_keywords', {})

        if isinstance(device_type_keywords, str) and device_type_keywords.startswith("{{lists}}"):
            lists_config = config.get('lists', {}) if config else {}

            domain_mapping = {
                'light': ('light_names',),
                'switch': ('switch_names',),
                'climate': ('climate_names',),
                'fan': ('fan_names',),
                'cover': ('cover_names',),
                'media_player': ('media_names', 'media_player_names'),
                'lock': ('lock_names',),
                'vacuum': ('vacuum_names',),
                'valve': ('valve_names',),
                'camera': ('camera_names',),
            }

            for domain, list_names in domain_mapping.items():
                for list_name in list_names:
                    keywords_list = lists_config.get(list_name, {}).get('values', [])
                    if keywords_list and any(keyword in text_lower for keyword in keywords_list):
                        device_types.append(domain)
                        break
        else:
            for keyword, domain in device_type_keywords.items():
                if keyword in text_lower:
                    device_types.append(domain)

        return area_names, list(set(device_types))

    async def _execute_control(
        self, area_names: list, device_types: list,
        is_on: bool, global_config: dict, language: str
    ):
        """执行设备控制."""
        is_global_control = not area_names

        try:
            all_devices = []

            if is_global_control:
                for domain in device_types:
                    try:
                        devices = self.hass.states.async_entity_ids(domain)
                        all_devices.extend(devices)
                    except Exception as e:
                        _LOGGER.debug(f"Failed to get {domain} devices: {e}")
            else:
                all_devices = await self._get_area_devices(area_names, device_types)

            if not all_devices:
                return None

            # 执行批量控制
            domain_services = global_config.get('domain_services', {})
            service_key = "turn_on" if is_on else "turn_off"

            all_success = 0
            all_errors = 0
            all_failed_devices = []

            # 按域分组设备
            devices_by_domain = {}
            for device_id in all_devices:
                domain = device_id.split('.')[0]
                if domain not in devices_by_domain:
                    devices_by_domain[domain] = []
                devices_by_domain[domain].append(device_id)

            # 执行操作
            for domain, devices in devices_by_domain.items():
                service_name = domain_services.get(domain, {}).get(service_key, service_key)
                success, errors, failed = await self._execute_device_operations(
                    devices, domain, service_name
                )
                all_success += success
                all_errors += errors
                all_failed_devices.extend(failed)

            # 生成响应
            responses = global_config.get('responses', {})
            if is_on:
                template = responses.get('success_on', '已打开{count}个设备{fail_msg}')
            else:
                template = responses.get('success_off', '已关闭{count}个设备{fail_msg}')

            fail_msg = self._format_failure_message(all_errors, all_failed_devices)
            area_text = area_names[0] if area_names else ""
            message = template.format(count=all_success, area=area_text, fail_msg=fail_msg)

            return self._create_response(language, message)

        except Exception as e:
            error_template = global_config.get('responses', {}).get('error', '设备控制失败：{error}')
            return self._create_response(language, error_template.format(error=str(e)), is_error=True)

    async def _get_area_devices(self, area_names: list, device_types: list) -> list:
        """获取指定区域的设备."""
        all_devices = []

        try:
            from homeassistant.helpers import entity_registry as er
            registry = er.async_get(self.hass)

            for domain in device_types:
                domain_devices = self.hass.states.async_entity_ids(domain)
                for device_id in domain_devices:
                    try:
                        entity_entry = registry.async_get(device_id)
                        if entity_entry and entity_entry.area_id:
                            area_entry = registry.async_get_area(entity_entry.area_id)
                            if area_entry and self._match_area_name(area_entry.name, area_names):
                                all_devices.append(device_id)
                    except Exception:
                        continue
        except Exception as e:
            _LOGGER.debug(f"Failed to get area devices: {e}")
            # 回退到全局
            for domain in device_types:
                all_devices.extend(self.hass.states.async_entity_ids(domain))

        return all_devices

    def _match_area_name(self, area_name: str, target_areas: list) -> bool:
        """区域名称匹配."""
        if area_name in target_areas:
            return True
        area_lower = area_name.lower()
        for target in target_areas:
            if target.lower() in area_lower or area_lower in target.lower():
                return True
        return False

    async def _execute_device_operations(
        self, devices: list, domain: str, service_name: str, service_data: dict | None = None
    ) -> tuple:
        """执行批量设备操作."""
        if service_data is None:
            service_data = {}

        success_count = 0
        error_count = 0
        failed_devices = []

        for device_id in devices:
            try:
                data = {'entity_id': device_id, **service_data}
                await self.hass.services.async_call(domain, service_name, data)
                success_count += 1
            except Exception as e:
                _LOGGER.debug(f"Failed to control device {device_id}: {e}")
                error_count += 1
                failed_devices.append(self._get_device_friendly_name(device_id))

        return success_count, error_count, failed_devices

    def _get_device_friendly_name(self, device_id: str) -> str:
        """获取设备友好名称."""
        try:
            state = self.hass.states.get(device_id)
            if state and state.attributes.get('friendly_name'):
                return state.attributes['friendly_name']
        except Exception:
            pass

        if '.' in device_id:
            return device_id.split('.', 1)[1].replace('_', ' ')
        return device_id

    def _format_failure_message(self, error_count: int, failed_devices: list) -> str:
        """格式化失败消息."""
        if error_count == 0:
            return ""

        unique_failed = list(set(failed_devices))
        if len(unique_failed) <= 3:
            return f"，但以下{len(unique_failed)}个设备失败：{'、'.join(unique_failed)}"
        else:
            return f"，但{len(unique_failed)}个设备失败，包括：{'、'.join(unique_failed[:3])}等"

    def _create_response(self, language: str, message: str, is_error: bool = False):
        """创建响应结果."""
        response = intent.IntentResponse(language=language)
        if is_error:
            response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        else:
            response.async_set_speech(message)

        return {
            "response": response,
            "success": not is_error,
            "message": message
        }

    # ========== 参数控制方法 ==========

    async def _handle_parameter_control(self, text: str, text_lower: str, global_config: dict):
        """处理参数控制命令."""
        if not self._has_parameter_command(text, text_lower, global_config):
            return None

        area_names = self._parse_areas(text_lower)
        is_global = not area_names

        # Try each parameter type in order
        result = await self._try_brightness_control(text, text_lower, global_config, area_names, is_global)
        if result:
            return result

        result = await self._try_brightness_complaint(text_lower, global_config, area_names, is_global)
        if result:
            return result

        result = await self._try_temperature_control(text_lower, global_config, area_names, is_global)
        if result:
            return result

        result = await self._try_volume_control(text, text_lower, global_config, area_names, is_global)
        if result:
            return result

        return None

    def _has_parameter_command(self, text: str, text_lower: str, global_config: dict) -> bool:
        """Check if the text contains a parameter control command."""
        param_keywords = global_config.get('param_keywords', [])
        if any(keyword in text_lower for keyword in param_keywords):
            return True

        # Check for direct parameter patterns
        if self._has_brightness_param(text, text_lower, global_config):
            return True
        if self._has_volume_param(text, text_lower, global_config):
            return True
        if self._has_temperature_param(text_lower, global_config):
            return True
        if self._has_brightness_complaint(text_lower, global_config):
            return True

        return False

    def _has_brightness_param(self, text: str, text_lower: str, global_config: dict) -> bool:
        """Check if text contains brightness parameter."""
        keywords = global_config.get('brightness_keywords', [])
        return any(kw in text_lower for kw in keywords) and re.search(r'(\d{1,3})\s*%?', text)

    def _has_volume_param(self, text: str, text_lower: str, global_config: dict) -> bool:
        """Check if text contains volume parameter."""
        keywords = global_config.get('volume_keywords', [])
        return any(kw in text_lower for kw in keywords) and re.search(r'(\d{1,3})\s*%?', text)

    def _has_temperature_param(self, text_lower: str, global_config: dict) -> bool:
        """Check if text contains temperature parameter."""
        keywords = global_config.get('temperature_keywords', [])
        return any(kw in text_lower for kw in keywords) and re.search(r'(\d{1,2})\s*度', text_lower)

    def _has_brightness_complaint(self, text_lower: str, global_config: dict) -> bool:
        """Check if text contains brightness complaint keywords."""
        complaint = global_config.get('brightness_complaint', {})
        if not complaint:
            return False
        hot_kw = complaint.get('hot_keywords', [])
        cold_kw = complaint.get('cold_keywords', [])
        return any(kw in text_lower for kw in hot_kw + cold_kw)

    async def _try_brightness_control(
        self, text: str, text_lower: str, global_config: dict,
        area_names: list, is_global: bool
    ):
        """Try to handle brightness control command."""
        keywords = global_config.get('brightness_keywords', [])
        if not any(kw in text_lower for kw in keywords):
            return None

        match = re.search(r'(\d{1,3})\s*%?', text)
        if not match:
            return None

        brightness = int(match.group(1))
        if 0 <= brightness <= 100:
            return await self._control_light_brightness(area_names, is_global, brightness)
        return None

    async def _try_brightness_complaint(
        self, text_lower: str, global_config: dict,
        area_names: list, is_global: bool
    ):
        """Try to handle brightness complaint (too hot/cold)."""
        complaint = global_config.get('brightness_complaint', {})
        if not complaint:
            return None

        hot_kw = complaint.get('hot_keywords', [])
        cold_kw = complaint.get('cold_keywords', [])
        default_brightness = complaint.get('default_brightness', {})

        if any(kw in text_lower for kw in hot_kw):
            brightness = default_brightness.get('hot_recommendation', 30)
            return await self._control_light_brightness(area_names, is_global, brightness)

        if any(kw in text_lower for kw in cold_kw):
            brightness = default_brightness.get('cold_recommendation', 70)
            return await self._control_light_brightness(area_names, is_global, brightness)

        return None

    async def _try_temperature_control(
        self, text_lower: str, global_config: dict,
        area_names: list, is_global: bool
    ):
        """Try to handle temperature control command."""
        keywords = global_config.get('temperature_keywords', [])
        if not any(kw in text_lower for kw in keywords):
            return None

        match = re.search(r'(\d{1,2})\s*度', text_lower)
        if not match:
            return None

        temp = int(match.group(1))
        if 16 <= temp <= 30:
            return await self._control_climate_temperature(area_names, is_global, temp)
        return None

    async def _try_volume_control(
        self, text: str, text_lower: str, global_config: dict,
        area_names: list, is_global: bool
    ):
        """Try to handle volume control command."""
        keywords = global_config.get('volume_keywords', [])
        if not any(kw in text_lower for kw in keywords):
            return None

        match = re.search(r'(\d{1,3})\s*%?', text)
        if not match:
            return None

        volume = int(match.group(1))
        if 0 <= volume <= 100:
            return await self._control_media_volume(area_names, is_global, volume)
        return None

    def _parse_areas(self, text_lower: str) -> list:
        """解析区域名称."""
        area_names = []
        # 使用缓存的配置
        config = self.config
        try:
            if config and 'lists' in config:
                areas = config['lists'].get('area_names', {}).get('values', [])
                for area in areas:
                    if area in text_lower:
                        area_names.append(area)
        except Exception:
            pass
        return area_names

    async def _control_light_brightness(self, area_names: list, is_global: bool, brightness: int):
        """控制灯光亮度."""
        devices = await self._get_devices_by_domain(['light'], area_names, is_global)
        if not devices:
            return None

        success, errors, failed = await self._execute_device_operations(
            devices, 'light', 'turn_on', {'brightness_pct': brightness}
        )

        area_text = area_names[0] if area_names else self._get_default_area_name()
        message = f"已将{area_text}灯光亮度调至{brightness}%"
        if errors > 0:
            message += self._format_error_suffix(errors)

        return self._create_response("zh-CN", message)

    async def _control_climate_temperature(self, area_names: list, is_global: bool, temperature: int):
        """控制空调温度."""
        devices = await self._get_devices_by_domain(['climate'], area_names, is_global)
        if not devices:
            return None

        success, errors, failed = await self._execute_device_operations(
            devices, 'climate', 'set_temperature', {'temperature': temperature}
        )

        area_text = area_names[0] if area_names else self._get_default_area_name()
        message = f"已将{area_text}温度设置为{temperature}度"
        if errors > 0:
            message += self._format_error_suffix(errors)

        return self._create_response("zh-CN", message)

    async def _control_media_volume(self, area_names: list, is_global: bool, volume: int):
        """控制媒体音量."""
        devices = await self._get_devices_by_domain(['media_player'], area_names, is_global)
        if not devices:
            return None

        volume_level = volume / 100.0
        success, errors, failed = await self._execute_device_operations(
            devices, 'media_player', 'volume_set', {'volume_level': volume_level}
        )

        area_text = area_names[0] if area_names else self._get_default_area_name()
        message = f"已将{area_text}音量调至{volume}%"
        if errors > 0:
            message += self._format_error_suffix(errors)

        return self._create_response("zh-CN", message)

    async def _get_devices_by_domain(self, domains: list, area_names: list, is_global: bool) -> list:
        """根据域获取设备."""
        if is_global:
            all_devices = []
            for domain in domains:
                try:
                    all_devices.extend(self.hass.states.async_entity_ids(domain))
                except Exception:
                    pass
            return all_devices
        else:
            return await self._get_area_devices(area_names, domains)


# 全局意图处理器实例
_global_intent_handler: LocalIntentHandler | None = None


def get_global_intent_handler(hass: HomeAssistant) -> LocalIntentHandler | None:
    """获取全局意图处理器实例."""
    global _global_intent_handler
    if _global_intent_handler is None:
        _global_intent_handler = LocalIntentHandler(hass)
    return _global_intent_handler
