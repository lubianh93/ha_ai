from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_memory"


@dataclass
class MemoryData:
    """Persistent long-term memory data."""

    global_summary: str = ""
    conversation_summary: str = ""
    pinned_facts: list[str] | None = None
    turn_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert memory data to dict."""
        return {
            "global_summary": self.global_summary,
            "conversation_summary": self.conversation_summary,
            "pinned_facts": self.pinned_facts or [],
            "turn_count": self.turn_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryData":
        """Create memory data from dict."""
        if not data:
            return cls(pinned_facts=[])
        return cls(
            global_summary=data.get("global_summary", ""),
            conversation_summary=data.get("conversation_summary", ""),
            pinned_facts=data.get("pinned_facts", []) or [],
            turn_count=data.get("turn_count", 0),
            updated_at=data.get("updated_at", ""),
        )


class HAIMemoryStore:
    """Persistent memory store for HA AI."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._store = Store[dict[str, Any]](hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._cache: MemoryData | None = None

    async def async_load(self) -> MemoryData:
        """Load memory from storage."""
        if self._cache is not None:
            return self._cache
        raw = await self._store.async_load()
        self._cache = MemoryData.from_dict(raw)
        return self._cache

    async def async_save(self, memory: MemoryData) -> None:
        """Save memory to storage."""
        memory.updated_at = datetime.now(UTC).isoformat()
        self._cache = memory
        await self._store.async_save(memory.to_dict())

    async def async_get(self) -> MemoryData:
        """Get current memory."""
        return await self.async_load()

    async def async_clear(self) -> None:
        """Clear all memory."""
        memory = MemoryData(pinned_facts=[])
        await self.async_save(memory)

    async def async_increment_turn_count(self) -> MemoryData:
        """Increment turn count and save."""
        memory = await self.async_load()
        memory.turn_count += 1
        await self.async_save(memory)
        return memory

    async def async_set_global_summary(self, summary: str) -> MemoryData:
        """Update global summary."""
        memory = await self.async_load()
        memory.global_summary = summary.strip()
        await self.async_save(memory)
        return memory

    async def async_set_conversation_summary(self, summary: str) -> MemoryData:
        """Update conversation summary."""
        memory = await self.async_load()
        memory.conversation_summary = summary.strip()
        await self.async_save(memory)
        return memory

    async def async_set_pinned_facts(self, pinned: list[str]) -> MemoryData:
        """Replace pinned facts."""
        memory = await self.async_load()
        memory.pinned_facts = [item.strip() for item in pinned if item and item.strip()]
        await self.async_save(memory)
        return memory

    async def async_add_pinned_fact(self, fact: str) -> MemoryData:
        """Add a pinned fact."""
        memory = await self.async_load()
        facts = memory.pinned_facts or []
        fact = fact.strip()
        if fact and fact not in facts:
            facts.append(fact)
        memory.pinned_facts = facts
        await self.async_save(memory)
        return memory

    async def async_get_memory_block(self, max_chars: int = 1200) -> str:
        """Render memory as a system-message-friendly text block."""
        memory = await self.async_load()

        sections: list[str] = []

        pinned = memory.pinned_facts or []
        if pinned:
            sections.append(
                "Pinned facts:\n" + "\n".join(f"- {item}" for item in pinned)
            )

        if memory.global_summary:
            sections.append("Long-term summary:\n" + memory.global_summary)

        if memory.conversation_summary:
            sections.append("Recent conversation summary:\n" + memory.conversation_summary)

        if not sections:
            return ""

        text = "Long-term memory:\n" + "\n\n".join(sections)
        return text[:max_chars].strip()


def get_memory_store(hass: HomeAssistant) -> HAIMemoryStore:
    """Get or create memory store singleton for this HA instance."""
    key = f"{DOMAIN}_memory_store"
    store = hass.data.get(key)
    if store is None:
        store = HAIMemoryStore(hass)
        hass.data[key] = store
    return store