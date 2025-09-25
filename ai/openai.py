"""OpenAI + MCP backend with simplified intent gate and MCP execution.

Chat replies are human-readable summaries; raw plan/results are written to logs.
Assumptions:
- Single active database from env TD_NAME (database_name is auto-injected).
- Always ignore tables: "user_query" plus any in IGNORED_TABLES.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config import Config, get_openai_config
from logger import ChatLogger
from .base import AI, Message
from .openai_prompts import tools_gate


class AI_OpenAI(AI):
    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.client = cfg["client"]
        self.mcp_url = os.getenv("MCP_URL", "").strip()
        self.default_db = os.getenv("TD_NAME", "").strip()
        self.schema_path = os.getenv("SCHEMA_SNAPSHOT", os.path.join("config", "schema_snapshot.json"))
        self.tools_catalog_path = os.getenv("MCP_TOOLS_CATALOG", os.path.join("ai", "mcp_tools.yml"))
        self.ignored_tables = self._parse_ignored_tables(os.getenv("IGNORED_TABLES", ""))
        self._metadata_loaded = False
        self.metadata: Dict[str, Any] = {"database": self.default_db, "tables": [], "columns": {}, "samples": {}}

    def generate_reply(self, messages: List[Message], context: Dict | None = None) -> str:
        if not messages:
            return ""
        self._ensure_metadata()
        self._ensure_logging_enabled()

        user_text = self._last_user_text(messages) or ""
        try:
            ChatLogger().log("user", user_text)
        except Exception:
            pass

        tool_lines = self._tool_lines_from_yaml()
        schema_text = self._build_schema_text()
        boundary = "This request is not about the database. I can only help with database questions."
        system_prompt = tools_gate(boundary, tool_lines, schema_text)

        plan = self._llm_plan(system_prompt, user_text)
        try:
            ChatLogger().event("openai.plan", json=json.dumps(plan or {}))
        except Exception:
            pass

        if plan is None:
            reply = "I couldn't interpret the request for database actions right now. Please try rephrasing."
            try:
                ChatLogger().log("ai", reply)
            except Exception:
                pass
            return reply

        if not plan.get("related_to_db"):
            msg = plan.get("message") or boundary
            try:
                ChatLogger().log("ai", msg)
            except Exception:
                pass
            return msg

        if not self.mcp_url:
            tool_names = ", ".join([t.get("name", "") for t in (plan.get("tools") or []) if t.get("name")]) or "(none)"
            reply = (
                "I figured this is a database request and planned to use these tools: "
                f"{tool_names}. But I can't run them because MCP_URL isn't configured. "
                "Please set MCP_URL and try again."
            )
            try:
                ChatLogger().log("ai", reply)
            except Exception:
                pass
            return reply

        try:
            results = self._execute_plan(plan)
        except Exception as e:
            reply = f"I planned database actions but execution failed: {e}"
            try:
                ChatLogger().event("openai.exec.error", error=str(e))
                ChatLogger().log("ai", reply)
            except Exception:
                pass
            return reply

        try:
            ChatLogger().event("openai.results", json=self._safe_json({"plan": plan, "results": results}))
        except Exception:
            pass

        reply = self._render_text_response(plan, results)
        try:
            ChatLogger().log("ai", reply)
        except Exception:
            pass
        return reply

    def warmup(self) -> None:
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        if self._metadata_loaded:
            return
        try:
            if os.path.exists(self.schema_path):
                with open(self.schema_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self.metadata.update(data)
        except Exception:
            pass
        self._metadata_loaded = True

    def _ensure_logging_enabled(self) -> None:
        try:
            cfg = Config.load()
            cfg.log_enabled = True
            ChatLogger._CFG = cfg  # type: ignore[attr-defined]
        except Exception:
            pass

    @staticmethod
    def _parse_ignored_tables(val: str) -> set[str]:
        s = {"user_query"}
        for part in (val or "").split(","):
            p = part.strip()
            if p:
                s.add(p)
        return s

    def _build_schema_text(self) -> str:
        tables: List[str] = list(self.metadata.get("tables") or [])
        cols: Dict[str, List[str]] = self.metadata.get("columns") or {}
        lines: List[str] = []
        for t in tables:
            if not t or t in self.ignored_tables:
                continue
            col_list = cols.get(t) or []
            col_str = ", ".join(col_list)
            lines.append(f"Table: {t}")
            lines.append(f"Columns: {col_str}")
        return "\n".join(lines)

    def _tool_lines_from_yaml(self) -> List[str]:
        path = self.tools_catalog_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            tools = data.get("tools") or []
            lines = []
            for t in tools:
                name = (t.get("name") or "").strip()
                desc = (t.get("description") or "").strip()
                if name:
                    lines.append(f"{name}: {desc}")
            return lines
        except Exception:
            return []

    @staticmethod
    def _last_user_text(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return messages[-1].get("content", "") if messages else ""

    def _llm_plan(self, system_prompt: str, user_text: str) -> Optional[Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0)
            content = getattr(resp.choices[0].message, "content", "") or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lstrip().lower().startswith("json"):
                    content = content.split("\n", 1)[1] if "\n" in content else ""
            return json.loads(content)
        except Exception:
            return None

    def _execute_plan(self, plan: Dict[str, Any]) -> Any:
        tools_plan = plan.get("tools") or []
        if not isinstance(tools_plan, list) or not tools_plan:
            return []
        try:
            return asyncio.run(self._execute_plan_async(tools_plan))
        except RuntimeError:
            return self._run_in_loop(self._execute_plan_async(tools_plan))

    @staticmethod
    def _run_in_loop(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    async def _execute_plan_async(self, tools_plan: List[Dict[str, Any]]) -> Any:
        client = MultiServerMCPClient({
            "mcp_server": {
                "url": self.mcp_url,
                "transport": "streamable_http",
            }
        })
        async with client.session("mcp_server") as session:
            tools = await load_mcp_tools(session)
            by_name: Dict[str, Any] = {getattr(t, "name", ""): t for t in tools}
            results: List[Any] = []
            for step in tools_plan:
                name = (step.get("name") or "").strip()
                args = step.get("args") or {}
                if not name or name not in by_name:
                    results.append({"tool": name, "error": "tool_not_found"})
                    continue
                blocked = self._is_blocked_args(args)
                if blocked:
                    results.append({"tool": name, "error": "ignored_table", "detail": blocked})
                    continue
                args = self._with_defaults(name, args)
                try:
                    res = await by_name[name].ainvoke(args)
                except Exception as e:
                    res = {"tool": name, "error": str(e)}
                results.append(res)
            return results

    def _is_blocked_args(self, args: Dict[str, Any]) -> Optional[str]:
        for key in ("table", "table_name", "obj_name"):
            val = (args.get(key) or "").strip() if isinstance(args.get(key), str) else None
            if val and val in self.ignored_tables:
                return f"{key} '{val}' is ignored"
        return None

    def _with_defaults(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        needs_db = any(prefix in tool_name for prefix in ("base_", "qlty_", "dba_"))
        if needs_db and self.default_db:
            args["database_name"] = self.default_db
        return args

    @staticmethod
    def _safe_json(obj: Any, max_len: int = 5000) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def _render_text_response(self, plan: Dict[str, Any], results: Any) -> str:
        return "Request executed. Details are recorded in the log file."

    
