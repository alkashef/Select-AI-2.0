"""OpenAI + MCP client backend using HTTP transport via ``mcp_use``.

This backend uses OpenAI Chat Completions for responses and can execute tool
calls through an MCP server. It exposes a single meta tool to the model,
"mcp_invoke", which takes a target MCP tool name and JSON arguments. When the
model calls this tool, we dispatch the request to the MCP server over HTTP and
feed the result back into the conversation before asking OpenAI for a final
answer.

Environment variables:
- MCP_TRANSPORT: must be "http".
- MCP_URL: base URL for the MCP server (e.g., "http://localhost:8001/mcp").
- MCP_API_KEY: optional bearer token for authentication (if required).
"""

from __future__ import annotations
import asyncio
import json
import os
from typing import Any, Dict, List
from config import get_openai_config
from logger import ChatLogger
from .base import AI, Message
from .mcp_client import MCPClient


class AI_MCP_OpenAI(AI):
    """OpenAI backend with MCP tool-call dispatch.

    Conversation flow:
    1) Send user/system history to OpenAI along with a single meta tool "mcp_invoke".
    2) If the model calls the tool, invoke the requested MCP tool with provided args.
    3) Add the tool result to the conversation and ask OpenAI again for the final reply.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key: str = cfg["api_key"]
        self.model: str = cfg["model"]
        self.client = cfg["client"]

        transport = os.getenv("MCP_TRANSPORT", "http").strip().lower()
        if transport != "http":
            raise ValueError("MCP_TRANSPORT must be set to 'http' for AI_MCP_OpenAI")

        base_url = os.getenv("MCP_URL", "").strip()
        if not base_url:
            raise ValueError("MCP_URL environment variable is required for MCP transport")

        headers: Dict[str, str] = {}
        api_key = os.getenv("MCP_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._mcp: MCPClient = MCPClient(base_url=base_url, headers=headers or None)

        try:
            ChatLogger().event(
                "ai_mcp_openai.init",
                model=self.model,
                key_suffix=self.api_key[-6:] if self.api_key else "",
                transport=transport,
                url=base_url,
            )
        except Exception:
            pass

    def _build_openai_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        chat_messages: List[Dict[str, str]] = []
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
        return chat_messages

    def _openai_tools_schema(self) -> List[Dict[str, Any]]:
        # Expose a single meta tool that can invoke MCP server tools
        return [
            {
                "type": "function",
                "function": {
                    "name": "mcp_invoke",
                    "description": (
                        "Call a Teradata MCP tool by name with JSON arguments. "
                        "Use this when the user asks for database operations or analytics supported by MCP."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "description": "MCP tool name to invoke"},
                            "arguments": {"type": "object", "description": "JSON arguments for the tool"},
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def generate_reply(self, messages: List[Message], context: Dict | None = None) -> str:
        if not messages:
            return ""

        # Build base messages
        chat_messages = self._build_openai_messages(messages)
        if not chat_messages:
            return ""

        # Temperature from env
        try:
            temperature = float(os.getenv("GPT_TEMPERATURE", "0").strip())
        except ValueError:
            temperature = 0.0

        tools = self._openai_tools_schema()

        # First call with tool availability
        try:
            ChatLogger().event("ai_mcp_openai.call", model=self.model, msgs=str(len(chat_messages)))
        except Exception:
            pass

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Prepare to dispatch to MCP and then ask OpenAI again
            assistant_msg = {
                "role": "assistant",
                "content": getattr(msg, "content", None) or "",
            }
            chat_messages.append(assistant_msg)

            # Ensure MCP session
            async def _ensure_mcp_connected() -> None:
                await self._mcp.connect()

            asyncio.run(_ensure_mcp_connected())

            for call in tool_calls:
                name = getattr(call.function, "name", "") if getattr(call, "function", None) else ""
                args_text = getattr(call.function, "arguments", "{}") if getattr(call, "function", None) else "{}"
                try:
                    args_obj = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                except Exception:
                    args_obj = {}

                if name == "mcp_invoke":
                    tool_name = str(args_obj.get("tool_name", "")).strip()
                    tool_args = args_obj.get("arguments") or {}

                    async def _do_call() -> str:
                        return await self._mcp.call_tool(tool_name, tool_args)

                    result_text = asyncio.run(_do_call())
                else:
                    result_text = f"[mcp-error] Unknown tool function: {name}"

                # Append tool result per OpenAI format
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(call, "id", ""),
                        "content": result_text,
                    }
                )

            # Ask OpenAI again for a final message post tool results
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=temperature,
            )

            msg = resp.choices[0].message

            # Close MCP session (best-effort)
            async def _close_mcp() -> None:
                if self._mcp is not None:
                    await self._mcp.close()

            try:
                asyncio.run(_close_mcp())
            except Exception:
                pass

        return getattr(msg, "content", "") or ""
