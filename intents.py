"""意图模块 - 兼容层.

此文件保留用于向后兼容，所有功能已迁移到 intents/ 子模块。
"""

from __future__ import annotations

# 从新模块导入所有公共接口
from .intents import (
    # 意图处理器
    ChineseIntentHandler,
    # 配置缓存
    ConfigCache,
    LocalIntentHandler,
    # 配置加载
    async_setup_intents,
    get_config_cache,
    get_device_operations_config,
    get_device_verification_config,
    get_global_intent_handler,
    get_intents_config,
    get_local_intents_config,
    is_device_operation,
)

# 导出所有公共接口
__all__ = [
    "ConfigCache",
    "get_config_cache",
    "async_setup_intents",
    "get_intents_config",
    "get_device_operations_config",
    "get_device_verification_config",
    "is_device_operation",
    "ChineseIntentHandler",
    "LocalIntentHandler",
    "get_global_intent_handler",
    "get_local_intents_config",
]
