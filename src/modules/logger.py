from pathlib import Path

import re
import datetime as dt
from typing import Optional

from modules.config import Config


class ChatLogger:
    """Simple, file-based logger for chat messages and app events."""

    _CFG: Config = Config.load()

    def __init__(self, file_path: Optional[Path | str] = None) -> None:
        """Initialize the logger with a unique file path per run."""

        # Determine base log file path (from config or override)
        base_path = Path(file_path) if file_path is not None else self._CFG.log_file
        base_dir = base_path.parent
        base_name = base_path.stem
        ext = base_path.suffix or ".log"

        # Generate a unique filename using UTC timestamp
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        unique_name = f"{base_name}_{timestamp}{ext}"
        self._path = base_dir / unique_name

        # Ensure directory exists
        base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, role: str, content: str) -> None:
        """Append a single chat message to the log."""
        if not self._CFG.log_enabled:
            return

        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        safe_content = re.sub(r"\s+", " ", content.replace("\r", " ").replace("\n", " ")).strip()
        line = f"[{timestamp}] {role}: {safe_content}\n"

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def event(self, name: str, **fields: str) -> None:
        """Log a structured app event."""
        if not self._CFG.log_enabled:
            return

        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        parts = [
            f"{k}={re.sub(r'\\s+', ' ', str(v).replace(chr(10), ' ').replace(chr(13), ' ')).strip()}"
            for k, v in fields.items()
        ]
        line = f"[{timestamp}] event:{name} " + " ".join(parts) + "\n"

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)


# Create global logger
logger = ChatLogger()