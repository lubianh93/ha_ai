"""配置缓存管理器 - 避免重复加载配置文件."""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


class ConfigCache:
    """配置缓存管理器，使用 loader 模块的全局缓存."""

    def get_config(self, force_reload: bool = False) -> dict[str, Any] | None:
        """获取配置，使用 loader 的缓存."""
        from .loader import get_global_config, reload_config

        if force_reload:
            return reload_config()
        return get_global_config()

    def _get_defaults(self) -> dict[str, Any]:
        """获取默认配置."""
        config = self.get_config()
        if config and 'intents' in config and 'ai_hub' in config['intents']:
            return config['intents']['ai_hub'].get('defaults', {})
        return {}

    def get_global_keywords(self) -> list[str]:
        """获取全局关键词."""
        # 首先尝试从GlobalDeviceControl获取
        config = self.get_config()
        if config and 'intents' in config:
            global_config = config['intents'].get('ai_hub', {}).get('GlobalDeviceControl', {})
            if global_config and 'global_keywords' in global_config:
                return global_config['global_keywords']

        # 如果没有，从默认配置获取
        defaults = self._get_defaults()
        return defaults.get('global_keywords', ["所有", "全部", "一切"])

    def get_local_features(self) -> list[str]:
        """获取本地特征关键词."""
        # 首先尝试从expansion_rules中提取
        config = self.get_config()
        if config and 'intents' in config:
            expansion_rules = config['intents'].get('ai_hub', {}).get('expansion_rules', {})
            local_features = []
            for key, value in expansion_rules.items():
                if isinstance(value, str) and '|' in value:
                    local_features.extend(value.split('|'))
            if local_features:
                return local_features

        # 如果没有，从默认配置获取
        defaults = self._get_defaults()
        return defaults.get('local_features', ["所有设备", "全部设备", "所有灯", "全部灯"])

    def get_automation_config(self, key: str, default_value=None) -> Any:
        """获取自动化配置."""
        config = self.get_config()
        if config and 'intents' in config:
            ai_hub_config = config['intents']['ai_hub']
            if key in ai_hub_config:
                return ai_hub_config[key]
            # 从默认配置获取
            defaults = ai_hub_config.get('defaults', {})
            if key in defaults:
                return defaults[key]

        # 如果都没有，返回传入的默认值
        return default_value

    def get_responses_config(self) -> dict[str, Any]:
        """获取响应配置."""
        config = self.get_config()
        if config and 'intents' in config:
            ai_hub_config = config['intents']['ai_hub']
            # 首先尝试从responses获取
            if 'responses' in ai_hub_config:
                return ai_hub_config['responses']
            # 从默认配置获取
            defaults = ai_hub_config.get('defaults', {})
            if 'responses' in defaults:
                return defaults['responses']

        return {}

    def get_verification_config(self) -> dict[str, Any]:
        """获取验证配置."""
        config = self.get_config()
        if config and 'intents' in config:
            ai_hub_config = config['intents']['ai_hub']
            # 首先尝试从verification获取
            if 'verification' in ai_hub_config:
                return ai_hub_config['verification']
            # 从默认配置获取
            defaults = ai_hub_config.get('defaults', {})
            if 'verification' in defaults:
                return defaults['verification']

        # 最后的硬编码备用值
        return {
            'total_timeout': 3,
            'max_retries': 3,
            'wait_times': [0.5, 0.8, 1.1]
        }

    def get_device_state_simulation(self) -> dict[str, Any]:
        """获取设备状态模拟配置."""
        defaults = self._get_defaults()
        return defaults.get('device_state_simulation', {
            "lights": {"living_room_main": "off", "living_room_ambient": "on"},
            "switches": {},
            "climate": {},
            "covers": {},
            "media_players": {},
            "locks": {},
            "vacuums": {}
        })

    def get_error_message(self, message_key: str) -> str:
        """获取错误消息."""
        defaults = self._get_defaults()
        error_messages = defaults.get('error_messages', {})
        return error_messages.get(message_key, f"未知错误: {message_key}")

    def get_timeout_config(self, timeout_key: str, default_value: int = 3) -> int:
        """获取超时配置."""
        defaults = self._get_defaults()
        timeouts = defaults.get('timeouts', {})
        return timeouts.get(timeout_key, default_value)


# 全局配置缓存实例
_config_cache = ConfigCache()


def get_config_cache() -> ConfigCache:
    """获取配置缓存实例."""
    return _config_cache
