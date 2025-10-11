"""Lightweight append-only chat logger."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from typing import Any

from .config import load_settings


@dataclass
class ChatLogger:
    path: Path
    enabled: bool

    def _write(self, line: str) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def log(self, role: str, message: str) -> None:
        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        self._write(f"[{timestamp}] {role}: {message}\n")

    def event(self, name: str, **payload: Any) -> None:
        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        extras = " ".join(f"{key}={value}" for key, value in payload.items())
        self._write(f"[{timestamp}] {name} {extras}\n")


_settings = load_settings()
logger = ChatLogger(path=_settings.logging.file, enabled=_settings.logging.enabled)
