"""Async helpers for talking to the Teradata MCP server via stdio."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import Settings, load_settings
from .logger import logger


@dataclass
class MCPState:
    session: ClientSession | None = None
    exit_stack: AsyncExitStack | None = None


class MCPClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._state = MCPState()
        self._connect_lock = asyncio.Lock()

    async def _ensure_connection(self) -> ClientSession:
        async with self._connect_lock:
            if self._state.session is not None:
                return self._state.session

            self._state.exit_stack = AsyncExitStack()
            stdio = await self._state.exit_stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=self._settings.mcp.command[0],
                        args=self._settings.mcp.command[1:],
                        cwd=str(self._settings.mcp.cwd),
                        env=self._settings.mcp.env,
                    )
                )
            )
            read, write = stdio
            session = await self._state.exit_stack.enter_async_context(ClientSession(read, write))

            try:
                await asyncio.wait_for(session.initialize(), timeout=15)
            except asyncio.TimeoutError as exc:
                await self.close()
                msg = "Timed out while initializing MCP session"
                logger.event("mcp.init.timeout", message=msg)
                raise RuntimeError(msg) from exc

            self._state.session = session
            logger.event("mcp.connected", command=" ".join(self._settings.mcp.command))
            return session

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        session = await self._ensure_connection()
        response = await session.call_tool(name=name, arguments=arguments or {})

        payload = ""
        if isinstance(response.content, list):
            payload = "".join(getattr(item, "text", "") for item in response.content)
        elif response.content:  # type: ignore[truthy-bool]
            payload = str(response.content)

        parsed: Any | None = None
        if payload:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None

        logger.event("mcp.tool", tool=name, bytes=len(payload))

        return {
            "tool": name,
            "raw": payload,
            "parsed": parsed,
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        session = await self._ensure_connection()
        response = await session.list_tools()
        catalog: List[Dict[str, Any]] = []
        for tool in response.tools:
            catalog.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "input_schema", None),
                }
            )
        return catalog

    async def close(self) -> None:
        if self._state.exit_stack is not None:
            await self._state.exit_stack.aclose()
        self._state = MCPState()
        logger.event("mcp.closed")
