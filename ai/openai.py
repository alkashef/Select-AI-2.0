"""OpenAI + MCP backend (scaffold).

Current state: minimal scaffold with a no-op generate_reply. Cleaned of
unused imports and attributes pending a future implementation.
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import AI, Message
from openai import OpenAI
from config import get_openai_config
from logger import ChatLogger

class AI_OpenAI(AI):
    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.model = cfg.get("model", "gpt-4o")
        self.api_key = cfg.get("api_key", "")
        self.tools_catalog = cfg.get("tools_catalog", [])
        self.mcp = cfg.get("mcp_client", None)
        self._client = self._build_openai_client(self.api_key)
        key_suffix=self.api_key[-6:] if self.api_key else ""
        ChatLogger().event("ai_gpt.init", model=self.model, key_suffix=key_suffix)

    def generate_reply(self, messages: List[Message], context: Dict | None = None) -> str:
        """Generate an assistant reply using OpenAI with MCP tool calls."""
        #raise NotImplementedError
        return "NOT IMPLEMENTED YET"

    def _complete(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the OpenAI Chat Completions API once and return the raw response.
        Should pass `self.tools_catalog` and `tool_choice="auto"`.
        """
        # TODO: call self._client.chat.completions.create(...)
        raise NotImplementedError

    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """Return True if the model response contains any tool calls."""
        # TODO: inspect response.choices[0].message.tool_calls
        raise NotImplementedError

    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and return a list of tool call dicts from the response.

        Each item should include at least: id, function.name, function.arguments
        (arguments as a JSON string or already-parsed dict depending on caller).
        """
        # TODO: parse response and return a normalized list
        raise NotImplementedError

    def _dispatch_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Invoke the MCP tool via `self.mcp.call_tool` and return text content.

        Contract: the Teradata MCP client returns either a dict or a string.
        This method should coerce that into a string suitable for a `tool`
        message content in OpenAI format.
        """
        # TODO: name = ..., args = ..., result = self.mcp.call_tool(name, args)
        # TODO: if dict, return json.dumps(result)
        raise NotImplementedError

    def _append_tool_exchange(
        self,
        messages: List[Dict[str, Any]],
        tool_call_id: str,
        tool_result_text: str,
    ) -> None: