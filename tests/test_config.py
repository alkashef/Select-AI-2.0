import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path

sys.path.insert(0, abspath(join(dirname(__file__), "..")))

import pytest

from config import Config, get_ai_backend


def test_config_load_requires_env(tmp_path):
    # Create a temp project structure with config/.env
    proj = tmp_path / "proj"
    (proj / "config").mkdir(parents=True)
    env_path = proj / "config" / ".env"
    env_path.write_text("LOG_ENABLED=true\nLOG_FILE=log.txt\n", encoding="utf-8")

    cfg = Config.load(base_dir=proj)
    assert cfg.env_path == env_path
    assert cfg.log_enabled is True


def test_get_ai_backend_default_and_override(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / "config").mkdir(parents=True)
    (proj / "config" / ".env").write_text("AI_BACKEND=\n", encoding="utf-8")

    assert get_ai_backend(base_dir=proj) == "gpt"  # default

    monkeypatch.setenv("AI_BACKEND", "GPT")
    assert get_ai_backend(base_dir=proj) == "gpt"
