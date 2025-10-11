"""GPT agent that can optionally call MCP tools before responding."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from openai import OpenAI

from .config import Settings, load_settings
from .logger import logger
from .mcp import MCPClient


class Message(TypedDict, total=False):
    role: str
    content: str
    ts: str
    chart: Any


class Agent:
    async def generate_reply(self, history: List[Message]) -> tuple[str, bool]:  # pragma: no cover - interface
        raise NotImplementedError


_SYSTEM_FALLBACK = (
    "You are Select AI, an assistant for business intelligence questions.\n\n"
    "When you need database-backed answers, respond with JSON exactly like:\n\n"
    '{ "tool": "tool_name", "arguments": {"key": "value"} }\n\n'
    "Only request one tool per turn. When you have all data, respond with plain text."
)


@dataclass
class GPTMCPAgent(Agent):
    settings: Settings
    mcp: MCPClient
    client: OpenAI
    system_prompt: str
    tool_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "GPTMCPAgent":
        settings = settings or load_settings()
        system_prompt = _load_system_prompt(settings.paths.repo_root / "system_prompt.txt")
        mcp = MCPClient(settings)
        client = OpenAI()
        logger.event("agent.init", backend="gpt")
        return cls(settings=settings, mcp=mcp, client=client, system_prompt=system_prompt)

    async def generate_reply(self, history: List[Message]) -> tuple[str, bool]:
        messages = self._convert_history(history)
        steps = 0

        while True:
            reply = await self._ask_gpt(messages)

            tool_request = self._maybe_parse_tool_request(reply)
            if tool_request is None:
                return reply, False

            steps += 1
            if steps > self.settings.mcp.max_steps:
                logger.event("agent.tool.max_steps", steps=steps)
                return "[error] Tool invocation limit reached.", False

            tool_name = tool_request["tool"]
            resolved_name, guidance = await self._resolve_tool(tool_name)
            if resolved_name is None:
                logger.event("agent.tool.unknown", requested=tool_name)
                if guidance:
                    messages.append({"role": "assistant", "content": guidance})
                else:
                    messages.append({"role": "assistant", "content": f"Tool `{tool_name}` is unavailable."})
                continue
            tool_name = resolved_name
            arguments = self._apply_defaults(tool_request.get("arguments", {}))
            arguments = self._sanitize_arguments(tool_name, arguments)
            logger.event("agent.tool.request", tool=tool_name)

            tool_result = await self.mcp.call_tool(tool_name, arguments)

            direct_reply = await self._maybe_direct_response(tool_name, tool_result, arguments)
            if direct_reply:
                return direct_reply, False

            messages.append({"role": "assistant", "content": json.dumps(tool_request)})

            formatted = self._format_tool_result(tool_name, tool_result)
            if formatted and formatted.startswith("Tool `"):
                fallback = await self._handle_tool_failure(tool_name, arguments)
                if fallback:
                    return fallback, False

            if formatted:
                messages.append({"role": "assistant", "content": formatted})
            else:
                fallback = await self._handle_tool_failure(tool_name, arguments)
                if fallback:
                    return fallback, False
                messages.append({"role": "assistant", "content": json.dumps(tool_result)})

    async def _ask_gpt(self, messages: List[Dict[str, str]]) -> str:
        def _call() -> str:
            response = self.client.chat.completions.create(
                model=self.settings.openai.model,
                messages=[{"role": "system", "content": self.system_prompt}, *messages],
                temperature=0.2,
                timeout=self.settings.openai.timeout,
            )
            choice = response.choices[0].message  # type: ignore[index]
            return choice.content or ""

        return await asyncio.to_thread(_call)

    @staticmethod
    def _convert_history(history: List[Message]) -> List[Dict[str, str]]:
        output: List[Dict[str, str]] = []
        for item in history:
            role = item.get("role", "assistant")
            role = "assistant" if role != "user" else "user"
            content = item.get("content", "")
            if content:
                output.append({"role": role, "content": content})
        return output

    @staticmethod
    def _maybe_parse_tool_request(reply: str) -> Optional[Dict[str, Any]]:
        text = reply.strip()
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None
        if "tool" not in parsed:
            return None
        if not isinstance(parsed["tool"], str):
            return None
        return parsed

    async def _ensure_tool_catalog(self) -> None:
        if self.tool_cache:
            return
        await self._refresh_tool_catalog()

    async def _refresh_tool_catalog(self) -> None:
        catalog = await self.mcp.list_tools()
        self.tool_cache = {entry["name"]: entry for entry in catalog}

    async def _resolve_tool(self, requested: str) -> tuple[Optional[str], Optional[str]]:
        name = requested.strip()
        if not name:
            return None, "Tool name missing."

        await self._ensure_tool_catalog()
        lower_map = {tool.lower(): tool for tool in self.tool_cache}

        alias = self._tool_aliases().get(name.lower())
        if alias:
            if alias not in self.tool_cache:
                await self._refresh_tool_catalog()
            if alias in self.tool_cache:
                return alias, None

        if name in self.tool_cache:
            return name, None

        if name.lower() in lower_map:
            return lower_map[name.lower()], None

        for tool_name in self.tool_cache:
            if name.lower() in tool_name.lower():
                return tool_name, None

        guidance = await self._tool_guidance_message(name)
        return None, guidance

    def _apply_defaults(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(arguments)
        default_db = "BANK_DB"

        db_value: Optional[str] = None
        for key in ("database_name", "database", "db", "schema"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                db_value = value.strip()
                break

        if not db_value:
            db_value = default_db

        args["database_name"] = db_value

        # Keep aliases in sync for downstream tools expecting different keys
        for key in ("database", "db", "schema"):
            args[key] = db_value

        return args

    def _sanitize_arguments(self, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        info = self.tool_cache.get(tool)
        if not info:
            return arguments

        schema = info.get("input_schema")
        if not isinstance(schema, dict):
            return arguments

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return arguments

        allowed = set(properties.keys())
        if not allowed:
            return arguments

        aliases = self._argument_aliases(tool)
        cleaned: Dict[str, Any] = {}
        for key, value in arguments.items():
            canonical = aliases.get(key, key)
            if canonical in allowed:
                cleaned[canonical] = value

        # Ensure required arguments are present if defaults were removed by alias mapping
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in cleaned and key in arguments:
                    cleaned[key] = arguments[key]

        return cleaned or arguments

    def _format_tool_result(self, tool: str, result: Dict[str, Any]) -> Optional[str]:
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return None

        status = str(parsed.get("status", "") or "").lower()
        if status and not self._is_success_status(status):
            error = parsed.get("error") or parsed.get("message") or parsed
            return f"Tool `{tool}` returned an error: {error}"

        data = parsed.get("results")
        if data is None:
            return None

        return self._summarize_results(tool, data)

    def _summarize_results(self, tool: str, data: Any) -> Optional[str]:
        if isinstance(data, dict):
            if not data:
                return "No results returned."
            lines = [f"**{tool} results:**"]
            for key, value in data.items():
                value = self._filter_user_query(value)
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2)
                else:
                    value_str = value
                lines.append(f"- {key}: {value_str}")
            return "\n".join(lines)

        if isinstance(data, list):
            data = self._filter_user_query(data)
            if not data:
                return "No results returned."
            lines = [f"**{tool} results:**"]
            if all(isinstance(item, dict) for item in data):
                keys = sorted({k for item in data for k in item.keys()})
                for item in data:
                    parts = ", ".join(f"{k}={item.get(k)}" for k in keys if k in item)
                    lines.append(f"- {parts}")
            else:
                for item in data:
                    lines.append(f"- {item}")
            return "\n".join(lines)

        return f"**{tool} result:** {data}"

    def _filter_user_query(self, data: Any) -> Any:
        if isinstance(data, list):
            filtered = []
            for item in data:
                if isinstance(item, dict):
                    if any(str(value).strip().lower() == "user_query" for value in item.values()):
                        continue
                    filtered.append(dict(item))
                elif isinstance(item, str):
                    if item.strip().lower() == "user_query":
                        continue
                    filtered.append(item)
                else:
                    filtered.append(item)
            return filtered
        if isinstance(data, dict):
            if any(str(value).strip().lower() == "user_query" for value in data.values()):
                return {}
            return data
        if isinstance(data, str) and data.strip().lower() == "user_query":
            return ""
        return data

    def _tool_aliases(self) -> Dict[str, str]:
        return {
            "list_tables": "base_tableList",
            "table_list": "base_tableList",
            "tables": "base_tableList",
            "list_table": "base_tableList",
            "columns": "sql_schema_reader",
            "list_columns": "sql_schema_reader",
        }

    def _argument_aliases(self, tool: str) -> Dict[str, str]:
        base_aliases = {
            "database": "database_name",
            "db": "database_name",
            "schema": "database_name",
        }

        if tool in {"sql_schema_reader"}:
            return {
                **base_aliases,
                "table": "table_name",
            }

        return base_aliases

    async def _tool_guidance_message(self, requested: str) -> str:
        await self._refresh_tool_catalog()
        lines = [
            f"Tool `{requested}` is not available. Here are the tools currently exposed by the MCP server:",
        ]
        for name, info in sorted(self.tool_cache.items()):
            description = info.get("description") or ""
            schema = info.get("input_schema") or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required", []) if isinstance(schema, dict) else []
            parts: List[str] = []
            if description:
                parts.append(description)
            if props:
                arg_bits = []
                for key, value in props.items():
                    piece = key
                    if isinstance(value, dict) and value.get("type"):
                        piece += f":{value['type']}"
                    if key in required:
                        piece += " (required)"
                    arg_bits.append(piece)
                if arg_bits:
                    parts.append("args: " + ", ".join(arg_bits))
            lines.append(f"- {name}: {'; '.join(parts) if parts else 'no schema provided'}")
        lines.append("Please choose the appropriate tool name from the list above.")
        return "\n".join(lines)

    async def _maybe_direct_response(
        self,
        tool: str,
        result: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Optional[str]:
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return None

        status = str(parsed.get("status", "")).lower()
        results = parsed.get("results")

        if tool in {"list_tables", "base_tableList"} and self._is_success_status(status):
            database = arguments.get("database_name", "BANK_DB")
            message = self._format_table_list(result, database)
            if message:
                return message

        if tool == "sql_schema_reader" and self._is_success_status(status):
            table = arguments.get("table") or arguments.get("table_name")
            message = self._format_column_list(result, table)
            if message:
                return message

        return None

    def _rows_from_results(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            if "rows" in data:
                rows = data.get("rows")
                columns = data.get("columns") or data.get("column_names")
                if isinstance(rows, list):
                    processed: List[Dict[str, Any]] = []
                    for row in rows:
                        if isinstance(row, dict):
                            processed.append(dict(row))
                        elif isinstance(row, list):
                            row_dict: Dict[str, Any] = {}
                            if isinstance(columns, list):
                                for idx, value in enumerate(row):
                                    key = str(columns[idx]) if idx < len(columns) else f"col_{idx}"
                                    row_dict[key] = value
                            else:
                                for idx, value in enumerate(row):
                                    row_dict[f"col_{idx}"] = value
                            processed.append(row_dict)
                        else:
                            processed.append({"value": row})
                    return processed
            if "data" in data and isinstance(data["data"], list):
                return self._rows_from_results({"rows": data["data"]})
            if "tables" in data and isinstance(data["tables"], list):
                return self._rows_from_results({"rows": data["tables"]})
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return [dict(item) for item in data]
        return []

    def _format_table_list(self, result: Dict[str, Any], database: str) -> Optional[str]:
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return None
        if not self._is_success_status(str(parsed.get("status", "") or "").lower()):
            return None

        rows = self._rows_from_results(parsed.get("results"))
        rows = self._filter_user_query(rows)
        if not rows:
            return f"No tables found in {database}."

        names: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if not isinstance(value, str):
                    continue
                if "table" in key.lower():
                    candidate = value.strip()
                    if candidate and candidate.lower() != "user_query":
                        names.append(candidate)
        if not names:
            for row in rows:
                for value in row.values():
                    if isinstance(value, str) and value.strip() and value.lower() != "user_query":
                        names.append(value.strip())

        if not names:
            return None

        unique: List[str] = []
        seen = set()
        for name in names:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(name)

        lines = [f"Tables in {database}:"]
        lines.extend(f"- {name}" for name in unique)
        return "\n".join(lines)

    def _format_column_list(self, result: Dict[str, Any], table: Optional[str]) -> Optional[str]:
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return None
        if not self._is_success_status(str(parsed.get("status", "") or "").lower()):
            return None

        rows = self._rows_from_results(parsed.get("results"))
        if not rows:
            return None

        title = f"Columns in {table}" if table else "Columns"
        lines = [title + ":"]
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = (
                row.get("column_name")
                or row.get("ColumnName")
                or row.get("columnname")
                or row.get("name")
            )
            datatype = (
                row.get("data_type")
                or row.get("Datatype")
                or row.get("dataType")
                or row.get("type")
            )
            nullable = row.get("nullable") or row.get("Nullable")
            desc = row.get("comment") or row.get("Comment") or row.get("description")

            parts: List[str] = []
            if name:
                parts.append(str(name))
            if datatype:
                parts.append(str(datatype))
            if nullable is not None:
                nullable_str = str(nullable).lower()
                parts.append("nullable" if nullable_str in {"y", "yes", "true", "1"} else "not nullable")
            if desc:
                parts.append(str(desc))

            if parts:
                lines.append("- " + ", ".join(parts))

        return "\n".join(lines) if len(lines) > 1 else None

    async def _handle_tool_failure(self, tool: str, arguments: Dict[str, Any]) -> Optional[str]:
        if tool not in {"list_tables", "base_tableList"}:
            return None

        database = arguments.get("database_name", "BANK_DB")
        sql = (
            "SELECT TableName FROM DBC.TablesV "
            "WHERE DatabaseName = :database ORDER BY TableName"
        )

        # Requirement: do not fallback to sql_tool anymore; just surface the error
        return None

    @staticmethod
    def _is_success_status(status: str) -> bool:
        normalized = status.strip().lower()
        return normalized in {"success", "ok", "okay", "succeeded"}


def _load_system_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return _SYSTEM_FALLBACK
