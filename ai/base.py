"""AI base interface.

Defines the abstract contract all AI backends must implement. Concrete
implementations should inherit from `AI` and implement `generate_reply`.

Message schema used across the app:
- role: "user" | "ai" | "assistant" | "system"
- content: str
"""

from abc import ABC, abstractmethod
from typing import Any


class AI(ABC):
    """Abstract AI backend contract.

    Subclasses may accept any config object and are expected to be stateless
    with respect to conversation history (the full message list is passed in).
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the backend with an optional configuration object."""
        self.config = config

    @abstractmethod
    def generate_reply(self, messages: list, context: dict | None = None) -> str:  # pragma: no cover - abstract
        """Return an assistant reply for the given chat `messages`.

        Args:
            messages: List of message dicts with at least {"role", "content"}.
            context: Optional auxiliary context (unused by most backends).

        Returns:
            The assistant message content as a string. Empty string on no-op.
        """
        raise NotImplementedError
