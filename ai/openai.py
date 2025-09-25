"""OpenAI + MCP backend (scaffold).

Current state: minimal scaffold with a no-op generate_reply. Cleaned of
unused imports and attributes pending a future implementation.
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import AI, Message


class AI_OpenAI(AI):
    def __init__(self, config: Any = None) -> None:
        super().__init__(config)

    def generate_reply(self, messages: List[Message], context: Dict | None = None) -> str:
        return "NOT IMPLEMENTED YET"
