"""Markdown filter for cleaning AI-generated content.

This module provides functions to remove markdown formatting from AI responses
while preserving the actual content.
"""

import re


def _remove_emojis(text: str) -> str:
    """Remove all emojis from text."""
    if not text:
        return ""

    result = []
    for char in text:
        code = ord(char)
        # Skip characters in emoji ranges (remove emojis), keep non-emoji characters
        if (
            (0x1F300 <= code <= 0x1F5FF) or  # Supplemental Symbols and Pictographs
            (0x1F600 <= code <= 0x1F64F) or  # Emoticons
            (0x1F680 <= code <= 0x1F6FF) or  # Transport & Map Symbols
            (0x1F900 <= code <= 0x1F9FF) or  # Miscellaneous Symbols
            (0x1FA70 <= code <= 0x1FAFF) or  # Symbols and Pictographs Extended-A
            (0x2600 <= code <= 0x26FF) or    # Miscellaneous Symbols
            (0x2700 <= code <= 0x27BF)       # Dingbats
        ):
            # This is an emoji, skip it (don't add to result)
            continue
        else:
            # This is not an emoji, keep it
            result.append(char)

    return ''.join(result)


# Patterns that should preserve content (capture groups)
_CAPTURE_PATTERNS = [
    re.compile(r'\*\*([^*\n]*)\*\*'),      # Bold: keep content
    re.compile(r'\*([^*\n]*)\*'),          # Italic: keep content
    re.compile(r'__([^_\n]*)__'),          # Bold: keep content
    re.compile(r'_([^_\n]*)_'),            # Italic: keep content
    re.compile(r'~~([^~\n]*)~~'),          # Strikethrough: keep content
    re.compile(r'`([^`\n]*)`'),            # Inline code: keep content
]

# Patterns that should be completely removed
_REMOVE_PATTERNS = [
    re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),     # Headers
    re.compile(r'^\s*[-*+]\s+', re.MULTILINE),       # Lists
    re.compile(r'^\s*>\s+', re.MULTILINE),           # Blockquotes
    re.compile(r'```[a-zA-Z0-9_-]*', re.MULTILINE),  # Code blocks start
    re.compile(r'```\s*$', re.MULTILINE),            # Code blocks end
    re.compile(r'^\|[^\n]*\|$', re.MULTILINE),       # Tables
    re.compile(r'^\|[\s-]*\|[\s-]*\|$', re.MULTILINE),  # Table separators
    re.compile(r'^-{3,}$|^_{3,}$|^\*{3,}$', re.MULTILINE),  # Horizontal rules
    re.compile(r'\[\^[^\]]*\]'),                     # Footnotes
    re.compile(r'^\[\^[^\]]*\]:.*$', re.MULTILINE),  # Footnote definitions
    re.compile(r'<[^>]*>'),                          # HTML tags
    re.compile(r'^\s*$\n^\s*$', re.MULTILINE),       # Empty lines
    re.compile(r'^`[a-zA-Z0-9_-]*$', re.MULTILINE)   # Language identifiers
]


def _apply_markdown_filters(content: str) -> str:
    """Apply all markdown filter patterns to content.

    Args:
        content: The content to filter

    Returns:
        Filtered content with markdown removed but content preserved
    """
    # Apply patterns that preserve content (capture groups)
    for pattern in _CAPTURE_PATTERNS:
        content = pattern.sub(r'\1', content)

    # Apply patterns that remove entirely
    for pattern in _REMOVE_PATTERNS:
        content = pattern.sub('', content)

    # Normalize line breaks
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content


def filter_markdown_content(content: str) -> str:
    """Filter markdown formatting from content, strip whitespace.

    This is the standard filter for complete responses.

    Args:
        content: The content to filter

    Returns:
        Filtered content with markdown removed and whitespace stripped
    """
    if not content:
        return ""

    # Remove emojis first
    content = _remove_emojis(content)

    # Apply markdown filters
    content = _apply_markdown_filters(content)

    # Remove leading/trailing whitespace per line
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)

    return content.strip()


def filter_markdown_streaming(content: str) -> str:
    """Filter markdown formatting for streaming responses.

    This version preserves all spaces for chunk-by-chunk streaming.

    Args:
        content: The content to filter

    Returns:
        Filtered content with markdown removed, spaces preserved
    """
    if not content:
        return ""

    # Remove emojis first
    content = _remove_emojis(content)

    # Apply markdown filters
    content = _apply_markdown_filters(content)

    # Don't strip to preserve chunk spacing

    return content
