# 意图配置说明

本目录包含 AI Hub 的意图配置文件，采用模块化设计便于维护。

## 文件结构

```
config/
├── base.yaml          # 基础配置、响应消息、设备操作定义
├── lists.yaml         # 设备类型和区域名称列表
├── expansion.yaml     # 扩展规则（动词、参数等）
├── intents.yaml       # 意图定义（句式模板）
└── local_control.yaml # 本地控制配置（全局设备控制）
```

## 配置说明

### base.yaml
- `responses`: 响应消息模板
- `device_operations`: 设备操作配置
- `automation_keywords`: 自动化关键词

### lists.yaml
- `light_names`: 灯光设备关键词
- `climate_names`: 气候设备关键词
- `area_names`: 区域名称
- 其他设备类型...

### expansion.yaml
- `let`: 礼貌用语（请、帮我等）
- `turn`: 开启动词
- `close`: 关闭动词
- `action_verbs`: 动作动词
- 数值参数范围...

### intents.yaml
- HA 原生意图扩展（timer、climate 等）
- 自定义意图定义

### local_control.yaml
- `GlobalDeviceControl`: 全局设备控制配置
  - `global_keywords`: 全局关键词（所有、全部等）
  - `on_keywords`: 开启关键词
  - `off_keywords`: 关闭关键词
  - `domain_services`: 设备域到服务的映射
  - `responses`: 响应消息模板

## 添加新意图

1. 在 `intents.yaml` 中添加意图定义
2. 如需新的设备类型，在 `lists.yaml` 中添加
3. 如需新的动词，在 `expansion.yaml` 中添加

## 注意事项

- 修改配置后需要重启 Home Assistant
- 配置会在启动时自动验证
- 查看日志了解配置加载状态
