"""OpenAI Chat Completions backend.

Responsibilities:
- Initialize an OpenAI client from env via config.get_openai_config().
- Map app roles ("user"/"ai"/"system") to OpenAI roles ("user"/"assistant"/"system").
- Call chat.completions.create and return the assistant's content.
- Emit lightweight events for diagnostics (init, call, call.error).
- Retry transient failures with simple exponential backoff.
"""

from typing import Any
import time
from config import get_openai_config
from logger import ChatLogger
from .base import AI, Message
import os


class AI_OpenAI(AI):
    """Concrete AI implementation using OpenAI Chat Completions models.

    Configuration is sourced from :func:`config.get_openai_config` which returns
    the API key, model name (e.g., ``gpt-4o``), and a configured OpenAI client.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.client = cfg["client"]
        try:
            ChatLogger().event(
                "ai_openai.init", model=self.model, key_suffix=self.api_key[-6:] if self.api_key else ""
            )
        except Exception:
            pass

    def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        """Generate an assistant reply using OpenAI Chat Completions.

        Notes:
        - Unknown roles are coerced to "user".
        - App's internal role "ai" is translated to OpenAI's "assistant".
        - Empty/whitespace-only contents are skipped.
        - Retries up to 3 times on exceptions with backoff (0.5s, 1s).
        """
        if not messages:
            return ""

        chat_messages: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "ai":
                role = "assistant"
            elif role not in ("user", "system", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": content})

        if not chat_messages:
            return ""

        try:
            ChatLogger().event("ai_openai.call", model=self.model, msgs=str(len(chat_messages)))
        except Exception:
            pass

        # Retry transient connection errors a few times with backoff
        # Read temperature from env (default 0)
        try:
            temperature = float(os.getenv("GPT_TEMPERATURE", "0").strip())
        except ValueError:
            temperature = 0.0

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    temperature=temperature,
                )
                break
            except Exception as e:  # Broad catch to avoid SDK version issues
                last_err = e
                try:
                    ChatLogger().event(
                        "ai_openai.call.error", error=f"{e.__class__.__name__}: {e}", attempt=str(attempt + 1)
                    )
                except Exception:
                    pass
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    raise
        msg = resp.choices[0].message
        return getattr(msg, "content", "") or ""