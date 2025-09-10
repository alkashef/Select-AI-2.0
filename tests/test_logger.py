import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path

sys.path.insert(0, abspath(join(dirname(__file__), "..")))

from logger import ChatLogger
from config import Config


def test_logger_writes_and_sanitizes(tmp_path, monkeypatch):
    # Enable logging and point to temp file
    monkeypatch.setenv("LOG_ENABLED", "true")
    log_file = tmp_path / "log.txt"
    monkeypatch.setenv("LOG_FILE", str(log_file))

    # Force reload of config class-level _CFG
    ChatLogger._CFG = Config.load(base_dir=Path(__file__).parent.parent)

    logger = ChatLogger()
    logger.log("user", "hello\nworld\r!")
    logger.event("test", a="b\n\rc")

    data = log_file.read_text(encoding="utf-8")
    assert "hello world !".replace("  ", " ") in data
    assert "event:test a=b c" in data
