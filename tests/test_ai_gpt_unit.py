import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(__file__), "..")))

import types

from ai.gpt import AI_GPT


class _FakeChoiceMsg:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)


class _FakeClient:
    def __init__(self, contents, fail_first=False):
        # contents: list of strings to return in successive calls
        self._contents = list(contents)
        self._fail_first = fail_first
        self._calls = 0

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                # Placeholder replaced at runtime via instance monkeypatching
                raise NotImplementedError

    def bind(self):
        # Bind the create method to the instance to allow stateful behavior
        client = self

        def _create(**kwargs):
            client._calls += 1
            if client._fail_first and client._calls == 1:
                raise RuntimeError("transient failure")
            content = client._contents.pop(0) if client._contents else ""
            return types.SimpleNamespace(choices=[_FakeChoiceMsg(content)])

        # Monkeypatch nested attribute for this instance
        self.chat.completions.create = staticmethod(_create)
        return self


def test_ai_gpt_happy_path(monkeypatch):
    # Patch get_openai_config to return our fake client
    from ai import gpt as gpt_mod

    def _fake_cfg():
        client = _FakeClient(["hello from fake"]).bind()
        return {"api_key": "x", "model": "gpt-test", "client": client}

    monkeypatch.setattr(gpt_mod, "get_openai_config", _fake_cfg)

    ai = AI_GPT()
    reply = ai.generate_reply([{"role": "user", "content": "hi"}])
    assert reply == "hello from fake"


def test_ai_gpt_retries_once_and_succeeds(monkeypatch):
    from ai import gpt as gpt_mod

    def _fake_cfg():
        client = _FakeClient(["recovered"], fail_first=True).bind()
        return {"api_key": "x", "model": "gpt-test", "client": client}

    monkeypatch.setattr(gpt_mod, "get_openai_config", _fake_cfg)

    ai = AI_GPT()
    reply = ai.generate_reply([{"role": "user", "content": "retry?"}])
    assert reply == "recovered"
