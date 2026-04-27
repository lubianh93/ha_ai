"""Command classifier for safe local intent execution.

This module is intentionally conservative. It only allows local execution when
the text is structurally an explicit control command, rather than merely
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


def _after_prefix_starts_with_action(
    text: str,
    action_keywords: list[str],
) -> bool:
    """Allow common polite/request prefixes before action words."""
    prefixes = [
        "请",
        "帮我",
        "给我",
        "麻烦",
        "麻烦你",
        "把",
        "将",
        "帮忙",
    ]

    if _starts_with_any(text, action_keywords):
        return True

    for prefix in prefixes:
        if text.startswith(prefix):
            rest = text[len(prefix):]
            if _starts_with_any(rest, action_keywords):
                return True

    return False


def _looks_like_meta_or_feedback(text: str) -> bool:
    """Detect sentences that discuss, correct, ask, or evaluate instead of command.

    This is not a negation-word list. It rejects sentences by discourse role:
    question, explanation, feedback, or reference to previous assistant action.
    """
    if not text:
        return True

    # Interrogative or reflective sentence.
    if "?" in text or "？" in text:
        return True

    # First-person or second-person evaluation/feedback is usually not a direct
    # device command, even if it contains action words.
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
        "如果",
        "假如",
        "是不是",
        "能不能",
        "可不可以",
    )
    if text.startswith(feedback_prefixes):
        return True

    # Long compound sentence is unsafe for direct global execution.
    discourse_markers = ("因为", "但是", "可是", "结果", "然后", "所以", "其实", "意思")
    if any(marker in text for marker in discourse_markers):
        return True

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

    # Safety rule: global direct execution must be short and command-shaped.
    # This rejects sentences that merely mention "open all devices".
    if len(text_clean) > 16:
        return CommandDecision(CommandKind.DISCUSSION_OR_FEEDBACK, False, "too_long_for_direct_global_command")

    if has_action and not _after_prefix_starts_with_action(text_norm, action_keywords):
        return CommandDecision(CommandKind.DISCUSSION_OR_FEEDBACK, False, "action_not_in_command_position")

    if has_parameter:
        # Parameter commands often start with 调/设置/设为/把/将 etc.
        parameter_command_prefixes = ["调", "设置", "设为", "把", "将", "请", "帮我", "给我"]
        if not _starts_with_any(text_norm, parameter_command_prefixes):
            return CommandDecision(CommandKind.DISCUSSION_OR_FEEDBACK, False, "parameter_not_in_command_position")
        return CommandDecision(CommandKind.PARAMETER_GLOBAL_CONTROL, True, "explicit_global_parameter_command")

    return CommandDecision(CommandKind.EXECUTE_GLOBAL_CONTROL, True, "explicit_global_action_command")