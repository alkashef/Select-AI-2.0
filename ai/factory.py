"""AI factory.

Responsibility:
- Inspect configuration to select a concrete AI backend.
- Construct and return the backend instance used by the app.
"""

from .base import AI
from .gpt import AI_GPT
from .openai import AI_OpenAI
from config import get_ai_backend


def get_ai() -> AI:
    """Construct and return a concrete :class:`AI` backend based on ``AI_BACKEND``.

    Currently supported values:
    - "gpt": :class:`ai.gpt.AI_GPT`
    - "openai": :class:`ai.openai.AI_OpenAI`

    Returns
    -------
    AI
        A ready-to-use backend implementing :class:`AI`.
    """
    backend = get_ai_backend()
    try:
        # Late import to avoid import-time .env side effects in tests
        from logger import ChatLogger  # type: ignore
        ChatLogger().event("ai.backend.select", backend=backend)
    except Exception:
        pass
    if backend == "gpt":
        return AI_GPT()
    if backend == "openai":
        return AI_OpenAI()
    raise ValueError(f"Unknown AI backend: {backend}")
