"""OpenAI + MCP backend.

Implements an orchestrator that:
- Uses OpenAI Chat Completions to plan an action (answer directly or call MCP).
- Calls MCP tools over streamable HTTP when needed (read-only).
- Summarizes results and returns a one-shot reply including SQL when used.

Environment (from config/.env):
- OPENAI_API_KEY / GPT_MODEL handled by config.get_openai_config()
- MCP_URL: http://localhost:8001/mcp/ (required for MCP usage)
- MCP_TIMEOUT: per tool call timeout in seconds (default 120)
- TD_NAME: default database when user doesn't specify

Logging:
- [LLM=>MCP] tool name + payload (sanitized)
- [MCP=>LLM] tool response (truncated head+tail for debug)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from config import get_openai_config
from logger import ChatLogger
from .base import AI, Message

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _truncate_head_tail(text: str, keep: int = 4000) -> str:
    """Keep the first and last `keep` chars; indicate truncation in the middle."""
    if len(text) <= keep * 2:
        return text
    omitted = len(text) - keep * 2
    return f"{text[:keep]}\n... [truncated {omitted} chars] ...\n{text[-keep:]}"


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences around JSON blocks.

    Handles patterns like ```json ... ``` or ``` ... ``` and trims whitespace.
    """
    if not text:
        return text
    t = text.strip()
    # ```json ... ``` or ``` … ```
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
        # Optional language tag (e.g., json)
        t = re.sub(r"^json\s+", "", t, flags=re.IGNORECASE)
    return t.strip()


def _split_qualified(name: str, default_db: str = "") -> Tuple[str, str]:
    """Split a possibly qualified identifier into (database, table).

    - Accepts forms like "DB.TABLE", "DB.TABLE.MORE" (use first two), or just "TABLE".
    - Strips quotes and whitespace. Falls back to default_db if database is missing.
    """
    if not name:
        return (default_db.strip(), "")
    n = name.strip().strip('"').strip("'")
    parts = [p for p in re.split(r"\.", n) if p]
    if len(parts) >= 2:
        return (parts[0].strip(), parts[1].strip())
    return (default_db.strip(), parts[0].strip())


class AI_OpenAI(AI):
    """OpenAI-backed chat that can call MCP tools over HTTP."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.client = cfg["client"]

        # MCP configuration
        self.mcp_url = self._normalize_mcp_url(os.getenv("MCP_URL", "").strip())
        try:
            self.mcp_timeout = float(os.getenv("MCP_TIMEOUT", "120").strip())
        except ValueError:
            self.mcp_timeout = 120.0
        self.default_db = os.getenv("TD_NAME", "").strip()

        try:
            ChatLogger().event(
                "ai_openai.init", model=self.model, mcp_url=self.mcp_url or "(unset)", default_db=self.default_db or "(unset)"
            )
        except Exception:
            pass

    @staticmethod
    def _normalize_mcp_url(url: str) -> str:
        if not url:
            return url
        u = url.strip()
        if not u.endswith("/"):
            u += "/"
        # Ensure path ends with /mcp/
        if not u.rstrip("/").endswith("/mcp"):
            u = u + "mcp/"
        return u

    # ----------------------------- LLM Planning ----------------------------- #
    def _plan_action(self, messages: list[Message]) -> Dict[str, Any]:
        """Ask the LLM to plan the next action.

        Returns a dict like:
        {"action":"answer|list_tables|describe_table|preview_table|read_query",
         "args": {...}, "reason": "..."}
        """
        # Keep prompt small; last user message is most important
        last_user = next((m for m in reversed(messages) if m.get("role") == "user" and m.get("content")), None)
        user_text = last_user.get("content", "") if last_user else ""

        sys_prompt = (
            "You are a database assistant with tool access via MCP. "
            "Choose exactly ONE action: list_tables, describe_table, preview_table, read_query, "
            "dq_univariate, dq_missing, dq_distinct, or answer. "
            "Return STRICT JSON ONLY (no code fences) with fields: action, args (object), reason. "
            "Rules: read-only; prefer terminal actions. For counts/aggregates like 'how many' or 'number of', "
            "choose read_query and provide a complete SQL statement (e.g., SELECT COUNT(*) FROM DB.TABLE). "
            "If no DB specified, you may use the default database."
        )

        # Minimal history: one system + latest user
        chat_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ]

        # Use JSON mode when supported; fall back otherwise
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=0,
            )
        content = _strip_code_fences((resp.choices[0].message.content or "").strip())
        try:
            plan = json.loads(content)
            if not isinstance(plan, dict):
                raise ValueError("plan not dict")
        except Exception:
            # Fallback: treat as direct answer
            plan = {"action": "answer", "args": {"text": content}, "reason": "fallback"}
        return plan

    # ----------------------------- MCP Calls -------------------------------- #
    async def _with_mcp(self, coro_func):
        """Helper to run an MCP coroutine with a connected HTTP session."""
        if not self.mcp_url:
            raise RuntimeError("MCP_URL is not set; cannot call MCP tools.")
        client = MultiServerMCPClient({
            "mcp": {"url": self.mcp_url, "transport": "streamable_http"}
        })
        try:
            async with client.session("mcp") as session:
                tools = await load_mcp_tools(session)
                tools_by = {t.name: t for t in tools}
                return await coro_func(tools_by)
        except Exception as e:
            raise RuntimeError(f"Failed to connect/load MCP tools at {self.mcp_url}: {e}")

    async def _call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Tuple[str, Any]:
        """Call a named MCP tool with payload, returning (name, result)."""
        async def _runner(tools_by):
            if tool_name not in tools_by:
                raise RuntimeError(f"MCP tool not found: {tool_name}")
            tool = tools_by[tool_name]
            # Log outgoing payload
            try:
                ChatLogger().log("[LLM=>MCP]", f"{tool_name} { _json_dumps(payload) }")
            except Exception:
                pass
            try:
                res = await asyncio.wait_for(tool.ainvoke(payload), timeout=self.mcp_timeout)
            except Exception as e:
                raise RuntimeError(f"Tool '{tool_name}' invocation failed: {e}")
            # Log incoming response (head+tail truncation for debug visibility)
            try:
                text = _json_dumps(res)
                ChatLogger().log("[MCP=>LLM]", _truncate_head_tail(text))
            except Exception:
                pass
            return res

        result = await self._with_mcp(_runner)
        return tool_name, result

    # ---------------------------- Public API -------------------------------- #
    def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        if not messages:
            return ""

        try:
            ChatLogger().event("ai_openai.call", msgs=str(len(messages)))
        except Exception:
            pass

        plan = self._plan_action(messages)
        action = str(plan.get("action", "answer")).strip()
        args = plan.get("args") or {}

        # Route actions
        try:
            if action == "answer":
                text = str(args.get("text", "")).strip()
                return text

            if action == "list_tables":
                db = str(args.get("database") or self.default_db or "").strip()
                if not db:
                    return "Please specify a database name."
                tool, payload = "base_tableList", {"database_name": db}
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "describe_table":
                # Accept qualified table (e.g., DB.TABLE) or separate fields
                raw_tbl = str(args.get("table") or args.get("obj_name") or "").strip()
                db = str(args.get("database") or self.default_db or "").strip()
                db, tbl = _split_qualified(raw_tbl, db)
                if not (db and tbl):
                    return "Please provide both database and table name."
                tool, payload = "base_columnDescription", {"database_name": db, "obj_name": tbl}
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "preview_table":
                raw_tbl = str(args.get("table") or args.get("table_name") or "").strip()
                db = str(args.get("database") or self.default_db or "").strip()
                db, tbl = _split_qualified(raw_tbl, db)
                if not (db and tbl):
                    return "Please provide both database and table name."
                tool, payload = "base_tablePreview", {"database_name": db, "table_name": tbl}
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "read_query":
                sql = str(args.get("sql") or "").strip()
                if not sql:
                    return "Please provide a SQL statement to run."
                tool, payload = "base_readQuery", {"sql": sql}
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "dq_univariate":
                db = str(args.get("database") or self.default_db or "").strip()
                tbl = str(args.get("table") or args.get("table_name") or "").strip()
                col = str(args.get("column") or args.get("column_name") or "").strip()
                if not (db and tbl and col):
                    return "Please provide database, table, and column for univariate stats."
                tool, payload = "qlty_univariateStatistics", {
                    "database_name": db,
                    "table_name": tbl,
                    "column_name": col,
                }
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "dq_missing":
                db = str(args.get("database") or self.default_db or "").strip()
                tbl = str(args.get("table") or args.get("table_name") or "").strip()
                if not (db and tbl):
                    return "Please provide database and table for missing values analysis."
                tool, payload = "qlty_missingValues", {
                    "database_name": db,
                    "table_name": tbl,
                }
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            if action == "dq_distinct":
                db = str(args.get("database") or self.default_db or "").strip()
                tbl = str(args.get("table") or args.get("table_name") or "").strip()
                col = str(args.get("column") or args.get("column_name") or "").strip()
                if not (db and tbl and col):
                    return "Please provide database, table, and column for distinct category analysis."
                tool, payload = "qlty_distinctCategories", {
                    "database_name": db,
                    "table_name": tbl,
                    "column_name": col,
                }
                name, res = asyncio.run(self._call_tool(tool, payload))
                return self._summarize_result(messages, tool=name, payload=payload, result=res)

            # Unknown action -> answer fallback
            return str(args.get("text", "")).strip()
        except Exception as e:
            try:
                ChatLogger().event("ai_openai.tool.error", error=f"{e.__class__.__name__}: {e}")
            except Exception:
                pass
            return f"Tool execution failed: {e}"

    def _summarize_result(self, messages: list[Message], tool: str, payload: Dict[str, Any], result: Any) -> str:
        """Ask LLM to summarize the tool result; include SQL if present."""
        sql = payload.get("sql")
        sys_prompt = (
            "You are a helpful data assistant. Summarize the provided tool result. "
            "When SQL is provided, include the SQL first, then a concise natural-language summary. "
            "Favor readable bullet points; avoid dumping raw JSON unless explicitly asked."
        )
        content_blocks = []
        if sql:
            content_blocks.append(f"SQL:\n{sql}")
        content_blocks.append(f"Result JSON (possibly truncated):\n{_truncate_head_tail(_json_dumps(result), keep=3000)}")
        usr = "\n\n".join(content_blocks)

        chat_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Completed tool {tool}; failed to summarize: {e}. Here is raw result:\n{_truncate_head_tail(_json_dumps(result))}"
