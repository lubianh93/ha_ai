"""Simplified AI Automation support for natural language automation creation."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class AIAutomationManager:
    """Simplified AI automation manager that directly uses AI-generated YAML."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the AI automation manager."""
        self.hass = hass

    async def create_automation_from_description(
        self,
        description: str,
        name: str | None = None,
        area_id: str | None = None
    ) -> dict[str, Any]:
        """Create an automation from natural language description."""
        try:
            # Generate YAML config using conversation engine
            yaml_config = await self._generate_automation_yaml(description, name, area_id)

            if not yaml_config:
                return {
                    "success": False,
                    "error": "Failed to generate automation YAML"
                }

            # Write to automations.yaml file
            success = await self._write_yaml_to_config(yaml_config)

            if success:
                # Reload automations
                try:
                    await self.hass.services.async_call("automation", "reload", {}, blocking=True)
                    return {
                        "success": True,
                        "message": "Automation created successfully! Config written to automations.yaml and reloaded."
                    }
                except Exception as reload_error:
                    return {
                        "success": True,
                        "message": (
                            f"Automation config saved but reload failed: {reload_error}. "
                            "Please reload automations manually or restart Home Assistant."
                        )
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to write automation configuration"
                }

        except Exception as e:
            _LOGGER.error("Error creating automation: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_automation_yaml(
        self,
        description: str,
        name: str | None = None,
        area_id: str | None = None
    ) -> str | None:
        """Generate automation YAML using the AI Hub conversation engine."""
        try:
            # Build YAML generation prompt (Chinese for Chinese LLM)
            yaml_prompt = f"""请根据以下描述生成一个标准的Home Assistant自动化YAML配置：

描述: {description}"""

            if name:
                yaml_prompt += f"\n自动化名称: {name}"

            if area_id:
                yaml_prompt += f"\n区域ID: {area_id}"

            yaml_prompt += """

要求:
1. 生成完整的YAML格式自动化配置
2. 只返回YAML代码，不要任何其他说明
3. 确保YAML格式正确，可以直接写入automations.yaml文件
4. 使用标准的Home Assistant自动化字段: alias, trigger, condition, action
5. 不要包含id字段（系统会自动生成）

示例格式:
```yaml
alias: "每天晚上8点半提醒接儿子"
trigger:
  - platform: time
    at: "20:30:00"
condition: []
action:
  - service: notify.persistent_notification
    data:
      title: "提醒"
      message: "时间到了，该去接儿子了"
mode: single
```

请直接返回YAML代码，不要包含任何其他文字："""

            # Call SiliconFlow conversation engine
            try:
                # Call SiliconFlow API to generate YAML
                yaml_result = await self._call_siliconflow_api_for_yaml(yaml_prompt)

                if yaml_result:
                    # Clean response and extract pure YAML code
                    return self._extract_yaml_from_response(yaml_result)

            except ImportError:
                _LOGGER.warning("Could not import conversation agent, using fallback logic")

            # Fallback logic if conversation engine is unavailable
            return self._generate_fallback_yaml(description, name, area_id)

        except Exception as e:
            _LOGGER.error("Error generating automation YAML: %s", e)
            return None

    async def _call_siliconflow_api_for_yaml(self, prompt: str) -> str | None:
        """Call AI Hub API directly for YAML generation."""
        try:
            import aiohttp

            # Get API key from HASS configuration
            api_key = self._get_api_key()
            if not api_key:
                _LOGGER.error("No AI Hub API key available")
                return None

            # Build request parameters
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            request_params = {
                "model": "Qwen/Qwen3-8B",  # Use default model
                "messages": messages,
                "stream": False,  # Non-streaming response
                "max_tokens": 2000,
                "temperature": 0.1  # Low temperature for stable YAML output
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Call SiliconFlow API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.siliconflow.cn/v1/chat/completions",
                    json=request_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error("API request failed: %s", error_text)
                        return None

                    response_data = await response.json()

                    # Extract response content
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        content = response_data["choices"][0]["message"]["content"]
                        _LOGGER.info("AI Hub response received for YAML generation")
                        return content
                    else:
                        _LOGGER.error("Invalid response format from AI Hub")
                        return None

        except Exception as e:
            _LOGGER.error("Error calling AI Hub API: %s", e)
            return None

    def _get_api_key(self) -> str | None:
        """Get AI Hub API key from Home Assistant configuration."""
        try:
            from homeassistant.config_entries import ConfigEntries

            from .const import DOMAIN

            config_entries: ConfigEntries = self.hass.config_entries
            ai_hub_entries = config_entries.async_entries_for_domain(DOMAIN)

            if ai_hub_entries:
                entry = ai_hub_entries[0]
                # API key is stored in runtime_data
                api_key = entry.runtime_data
                if api_key:
                    return api_key

            _LOGGER.warning("No AI Hub configuration found")
            return None

        except Exception as e:
            _LOGGER.error("Error getting API key: %s", e)
            return None

    def _extract_yaml_from_response(self, response: str) -> str | None:
        """Extract pure YAML code from AI response."""
        try:
            import re

            # 查找 ```yaml 代码块
            yaml_match = re.search(r'```yaml\s*\n(.*?)\n```', response, re.DOTALL)
            if yaml_match:
                return yaml_match.group(1).strip()

            # 查找 ``` 代码块（没有语言标识）
            code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # 如果没有代码块，尝试提取看起来像YAML的内容
            lines = response.split('\n')
            yaml_lines = []
            in_yaml = False

            for line in lines:
                line = line.strip()
                if line.startswith('alias:') or line.startswith('trigger:') or line.startswith('action:'):
                    in_yaml = True

                if in_yaml and line:
                    yaml_lines.append(line)

            if yaml_lines:
                return '\n'.join(yaml_lines)

            return None

        except Exception as e:
            _LOGGER.error("Error extracting YAML from response: %s", e)
            return None

    def _generate_fallback_yaml(
        self,
        description: str,
        name: str | None = None,
        area_id: str | None = None
    ) -> str | None:
        """Fallback YAML generation when AI is not available."""
        if "提醒" in description or "通知" in description or "记得" in description or "记住" in description:
            time_str = self._extract_time_from_description(description)
            alias_name = name or self._extract_name_from_description(description)

            return f'''alias: "{alias_name}"
trigger:
  - platform: time
    at: "{time_str}"
condition: []
action:
  - service: notify.persistent_notification
    data:
      title: "⏰ 提醒"
      message: "{description}"
mode: single'''

        elif "当" in description and "就" in description:
            return f'''alias: "{name or self._extract_name_from_description(description)}"
trigger:
  - platform: state
    entity_id: binary_sensor.motion_sensor
    to: "on"
condition: []
action:
  - service: light.turn_on
    target:
      entity_id: light.living_room
mode: single'''

        else:
            return f'''alias: "{name or self._extract_name_from_description(description)}"
trigger:
  - platform: state
    entity_id: binary_sensor.default_sensor
    to: "on"
condition: []
action:
  - service: light.turn_on
    target:
      entity_id: light.default_light
mode: single'''

    def _extract_time_from_description(self, description: str) -> str:
        """Extract time from natural language description."""
        import re

        description_lower = description.lower()

        # Match "X点Y分" format (Chinese time)
        match = re.search(r'(\d+)\s*点\s*(\d+)\s*分', description_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}:{minute:02d}:00"

        # Match "X点半" format (Chinese half hour)
        match = re.search(r'(\d+)\s*点半', description_lower)
        if match:
            hour = int(match.group(1))
            return f"{hour:02d}:30:00"

        # Match "X点" format (Chinese hour)
        match = re.search(r'(\d+)\s*点', description_lower)
        if match:
            hour = int(match.group(1))
            return f"{hour:02d}:00:00"

        # Default time
        return "20:30:00"

    def _extract_name_from_description(self, description: str) -> str:
        """Extract a meaningful name from description."""
        # Simple name extraction logic
        if len(description) > 50:
            return description[:47] + "..."
        return description

    async def _write_yaml_to_config(self, yaml_config: str) -> bool:
        """Write YAML configuration to automations.yaml file with validation."""
        try:
            import os
            import shutil
            import uuid

            import yaml

            config_dir = self.hass.config.config_dir
            automations_file = os.path.join(config_dir, "automations.yaml")

            def _write_file():
                """Execute file operations in executor."""
                # Backup existing file
                if os.path.exists(automations_file):
                    backup_file = f"{automations_file}.backup_{int(dt_util.now().timestamp())}"
                    shutil.copy2(automations_file, backup_file)
                    _LOGGER.info("Backed up existing automations.yaml")

                # Read existing config
                existing_automations = []
                if os.path.exists(automations_file):
                    with open(automations_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            existing_automations = yaml.safe_load(content) or []

                # Parse new YAML config
                try:
                    new_automation = yaml.safe_load(yaml_config)
                    if not isinstance(new_automation, list):
                        new_automation = [new_automation]
                except yaml.YAMLError as e:
                    _LOGGER.error("Invalid YAML generated: %s", e)
                    raise ValueError(f"Invalid YAML format: {e}")

                # Validate and enhance each automation
                for automation in new_automation:
                    # Validate required fields
                    if not isinstance(automation, dict):
                        raise ValueError("Automation must be a dictionary")

                    if "trigger" not in automation:
                        raise ValueError("Automation must have a 'trigger' field")

                    if "action" not in automation:
                        raise ValueError("Automation must have an 'action' field")

                    # Generate unique ID if not present (required for UI editing)
                    if "id" not in automation:
                        automation["id"] = str(uuid.uuid4())
                        _LOGGER.debug("Generated automation ID: %s", automation["id"])

                    # Ensure alias exists
                    if "alias" not in automation:
                        automation["alias"] = f"AI Generated Automation {automation['id'][:8]}"

                    # Ensure mode exists
                    if "mode" not in automation:
                        automation["mode"] = "single"

                # Add new automations
                existing_automations.extend(new_automation)

                # Write config file
                with open(automations_file, 'w', encoding='utf-8') as f:
                    f.write("# Home Assistant Automations\n")
                    f.write("# Generated/Modified by AI Hub Integration\n\n")
                    yaml.dump(existing_automations, f,
                              default_flow_style=False,
                              allow_unicode=True,
                              sort_keys=False,
                              indent=2)

                return True

            # Run file operations in executor
            await self.hass.async_add_executor_job(_write_file)
            _LOGGER.info("Automation written to config file")
            return True

        except OSError as err:
            _LOGGER.error("Could not write automation to config file: %s", err)
            return False


def get_automation_manager(hass: HomeAssistant) -> AIAutomationManager:
    """Get the AI automation manager instance from hass.data."""
    from . import get_or_create_ai_hub_data

    ai_hub_data = get_or_create_ai_hub_data(hass)
    if ai_hub_data.automation_manager is None:
        ai_hub_data.automation_manager = AIAutomationManager(hass)
    return ai_hub_data.automation_manager


async def async_setup_ai_automation(hass: HomeAssistant) -> None:
    """Set up AI automation services."""
    manager = get_automation_manager(hass)

    async def create_automation_service(call: ServiceCall) -> dict[str, Any]:
        """Create automation from natural language."""
        description = call.data.get("description", "")
        name = call.data.get("name")
        area_id = call.data.get("area_id")

        if not description:
            return {"success": False, "error": "Description is required"}

        return await manager.create_automation_from_description(description, name, area_id)

    hass.services.async_register(
        DOMAIN, "create_automation", create_automation_service,
        schema=vol.Schema({
            vol.Required("description"): cv.string,
            vol.Optional("name"): cv.string,
            vol.Optional("area_id"): cv.string,
        })
    )

    _LOGGER.info("AI automation services registered")
