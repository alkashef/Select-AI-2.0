"""Lightweight append-only chat logger.

Responsibilities:
- Load logging config from env via Config.load() (LOG_ENABLED, LOG_FILE).
- Provide two write-only methods: log() for chat lines, event() for app events.
- Write timestamps in UTC ISO-8601 with seconds precision.
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
import re
from typing import Optional
from config import Config


class ChatLogger:
    """Simple, file-based logger for chat messages and app events.

    Chat line format:
        [YYYY-MM-DDTHH:MM:SSZ] role: content
    Event line format:
        [YYYY-MM-DDTHH:MM:SSZ] event:<name> key=value ...
    """

    # Class-level configuration loaded from environment
    _CFG: Config = Config.load()

    def __init__(self, file_path: Optional[Path | str] = None) -> None:
        """Initialize the logger with a file path.

        Args:
            file_path: Optional override for the log file path. If None, uses LOG_FILE from config.
        """
        # Resolve path: explicit argument wins; otherwise use configured path
        self._path = Path(file_path) if file_path is not None else self._CFG.log_file

    def log(self, role: str, content: str) -> None:
        """Append a single message to the log file.

        Args:
            role: Message role, e.g., 'user' or 'assistant'.
            content: Message content.
        """
        if not self._CFG.log_enabled:
            return

        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        # Replace newlines/CR with spaces, then collapse runs of whitespace
        safe_content = content.replace("\r", " ").replace("\n", " ")
        safe_content = re.sub(r"\s+", " ", safe_content).strip()
        line = f"[{timestamp}] {role}: {safe_content}\n"

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def event(self, name: str, **fields: str) -> None:
        """Log a structured app event as a single line.

        Example: logger.event("ai.init", backend="gpt", model="gpt-4o")
        """
        if not self._CFG.log_enabled:
            return

        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        parts = []
        for k, v in fields.items():
            val = str(v).replace("\n", " ").replace("\r", " ")
            val = re.sub(r"\s+", " ", val).strip()
            parts.append(f"{k}={val}")
        line = f"[{timestamp}] event:{name} " + " ".join(parts) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
