"""Command classifier for safe local intent execution.

This module is conservative by design. It only allows local execution when the
text is structurally an explicit global control command, rather than merely
containing action words and global keywords.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CommandKind(str, Enum):
    """Classified command kind."""

    EXECUTE_GLOBAL_CONTROL = "execute_global_control"
    PARAMETER_GLOBAL_CONTROL = "parameter_global_control"
    DISCUSSION_OR_FEEDBACK = "discussion_or_feedback"
    NON_GLOBAL_CONTROL = "non_global_control"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CommandDecision:
    """Decision result for local command execution."""

    kind: CommandKind
    should_execute_locally: bool
    reason: str


_COMMAND_PREFIXES = (
    "请",
    "帮我",
    "给我",
    "麻烦你",
    "麻烦",
    "帮忙",
)

_OBJECT_MARKERS = ("把", "将")


def _compact(text: str) -> str:
    """Normalize text for lightweight Chinese command classification."""
    return (
        text.lower()
        .strip()
        .replace(" ", "")
        .replace("，", "")
        .replace(",", "")
        .replace("。", "")
        .replace(".", "")
    )


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def _starts_with_any(text: str, keywords: list[str]) -> bool:
    return any(keyword and text.startswith(keyword) for keyword in keywords)


def _first_index(text: str, keywords: list[str]) -> int:
    indexes = [text.find(keyword) for keyword in keywords if keyword and keyword in text]
    return min(indexes) if indexes else -1


def _strip_command_prefix(text: str) -> str:
    """Remove one or more polite/request prefixes."""
    current = text
    changed = True
    while changed:
        changed = False
        for prefix in _COMMAND_PREFIXES:
            if current.startswith(prefix):
                current = current[len(prefix):]
                changed = True
                break
    return current


def _looks_like_meta_or_feedback(text: str) -> bool:
    """Detect discussion, correction, complaint, or reflection.

    This is not a negation-word list. It rejects sentences by discourse role:
    question, explanation, feedback, or reference to previous assistant action.
    """
    if not text:
        return True

    if "?" in text or "？" in text:
        return True

    # First-person/second-person or reflective openings are usually feedback or
    # discussion, not a direct device-control command.
    feedback_prefixes = (
        "我",
        "你",
        "刚才",
        "刚刚",
        "上次",
        "前面",
        "之前",
        "为什么",
        "怎么",
        "如何",
        "如果",
        "假如",
        "是不是",
        "能不能",
        "可不可以",
        "会不会",
    )
    if text.startswith(feedback_prefixes):
        return True

    discourse_markers = (
        "因为",
        "但是",
        "可是",
        "结果",
        "然后",
        "所以",
        "其实",
        "意思",
        "不是",
        "并不是",
    )
    if any(marker in text for marker in discourse_markers):
        return True

    return False


def _is_action_command_shape(
    text: str,
    global_keywords: list[str],
    action_keywords: list[str],
) -> bool:
    """Return whether the text has a safe action-command shape."""
    if _starts_with_any(text, action_keywords):
        return True

    rest = _strip_command_prefix(text)

    if _starts_with_any(rest, action_keywords):
        return True

    # 把/将 + 全局对象 + 动作
    # 例如：把所有灯关掉、请把全部灯打开、帮我将所有插座关闭
    for marker in _OBJECT_MARKERS:
        if rest.startswith(marker):
            obj_phrase = rest[len(marker):]
            global_pos = _first_index(obj_phrase, global_keywords)
            action_pos = _first_index(obj_phrase, action_keywords)
            return global_pos != -1 and action_pos != -1 and action_pos > global_pos

    # 全局对象 + 动作
    # 例如：所有灯关掉、全部插座关闭
    global_pos = _first_index(rest, global_keywords)
    action_pos = _first_index(rest, action_keywords)
    if global_pos == 0 and action_pos > global_pos:
        return True

    return False


def _is_parameter_command_shape(
    text: str,
    global_keywords: list[str],
    parameter_keywords: list[str],
) -> bool:
    """Return whether the text has a safe parameter-command shape."""
    rest = _strip_command_prefix(text)

    parameter_prefixes = ("调", "设置", "设为", "调到", "调整")
    if _starts_with_any(rest, list(parameter_prefixes)):
        return _contains_any(rest, global_keywords) and _contains_any(rest, parameter_keywords)

    # 把/将 + 全局对象 + 参数动作
    # 例如：把所有灯亮度调到50%、将全部空调温度设为26度
    for marker in _OBJECT_MARKERS:
        if rest.startswith(marker):
            obj_phrase = rest[len(marker):]
            global_pos = _first_index(obj_phrase, global_keywords)
            param_pos = _first_index(obj_phrase, parameter_keywords)
            return global_pos != -1 and param_pos != -1 and param_pos > global_pos

    return False


def classify_global_control_command(
    text: str,
    global_config: dict[str, Any],
) -> CommandDecision:
    """Classify whether text is safe for direct local global control."""
    text_clean = text.strip()
    text_norm = _compact(text)

    if not text_norm:
        return CommandDecision(CommandKind.UNKNOWN, False, "empty_text")

    global_keywords = global_config.get("global_keywords", [])
    on_keywords = global_config.get("on_keywords", [])
    off_keywords = global_config.get("off_keywords", [])

    param_keywords = global_config.get("param_keywords", [])
    brightness_keywords = global_config.get("brightness_keywords", [])
    volume_keywords = global_config.get("volume_keywords", [])
    color_keywords = global_config.get("color_keywords", [])
    temperature_keywords = global_config.get("temperature_keywords", [])

    action_keywords = on_keywords + off_keywords
    parameter_keywords = (
        param_keywords
        + brightness_keywords
        + volume_keywords
        + color_keywords
        + temperature_keywords
    )

    has_global = _contains_any(text_norm, global_keywords)
    has_action = _contains_any(text_norm, action_keywords)
    has_parameter = _contains_any(text_norm, parameter_keywords)

    if not has_global:
        return CommandDecision(CommandKind.NON_GLOBAL_CONTROL, False, "no_global_scope")

    if not (has_action or has_parameter):
        return CommandDecision(CommandKind.UNKNOWN, False, "no_action_or_parameter")

    if _looks_like_meta_or_feedback(text_norm):
        return CommandDecision(CommandKind.DISCUSSION_OR_FEEDBACK, False, "meta_or_feedback")

    # Global device control is high risk. Long sentences are more likely to be
    # feedback, explanation, or discussion rather than an immediate command.
    if len(text_clean) > 20:
        return CommandDecision(
            CommandKind.DISCUSSION_OR_FEEDBACK,
            False,
            "too_long_for_direct_global_command",
        )

    if has_parameter:
        if _is_parameter_command_shape(text_norm, global_keywords, parameter_keywords):
            return CommandDecision(
                CommandKind.PARAMETER_GLOBAL_CONTROL,
                True,
                "explicit_global_parameter_command",
            )

        return CommandDecision(
            CommandKind.DISCUSSION_OR_FEEDBACK,
            False,
            "parameter_not_in_command_position",
        )

    if has_action:
        if _is_action_command_shape(text_norm, global_keywords, action_keywords):
            return CommandDecision(
                CommandKind.EXECUTE_GLOBAL_CONTROL,
                True,
                "explicit_global_action_command",
            )

        return CommandDecision(
            CommandKind.DISCUSSION_OR_FEEDBACK,
            False,
            "action_not_in_command_position",
        )

    return CommandDecision(CommandKind.UNKNOWN, False, "unclassified")