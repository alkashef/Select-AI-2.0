import sys
from os.path import abspath, dirname, join

# Ensure project root on path for direct imports when running from subfolders
sys.path.insert(0, abspath(join(dirname(__file__), "..")))

import pytest

from ai.factory import get_ai
from ai.gpt import AI_GPT


def test_get_ai_returns_gpt_when_backend_is_gpt(monkeypatch):
    monkeypatch.setenv("AI_BACKEND", "gpt")
    ai = get_ai()
    assert isinstance(ai, AI_GPT)


def test_get_ai_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("AI_BACKEND", "unknown")
    with pytest.raises(ValueError):
        _ = get_ai()
