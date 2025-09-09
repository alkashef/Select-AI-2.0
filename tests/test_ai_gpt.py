import os
import sys
import pytest
from os.path import abspath, dirname, join

# Only run these tests when explicitly enabled to avoid flaky CI/network issues.
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_REAL_OPENAI_TESTS"),
    reason="Set RUN_REAL_OPENAI_TESTS=1 to run integration tests against the real OpenAI client.",
)

# Ensure project root is on sys.path when running via pytest from subfolders
sys.path.insert(0, abspath(join(dirname(__file__), "..")))
from ai.gpt import AI_GPT


def test_generate_reply_real_client_contains_cairo():
    # Require an API key to be present; otherwise skip gracefully.
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping real-client integration test.")
    if not os.getenv("GPT_MODEL"):
        pytest.skip("GPT_MODEL is not set; skipping real-client integration test.")

    ai = AI_GPT()
    messages = [{"role": "user", "content": "what is the capitol of Egypt?"}]
    reply = ai.generate_reply(messages)

    assert isinstance(reply, str)
    assert "cairo" in reply.lower()
