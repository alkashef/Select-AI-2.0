"""HTTP MCP client wrapper using mcp_use."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from mcp_use import connect_http
from logger import ChatLogger


class MCPClient:
    """Async MCP client that connects to a remote MCP server over HTTP."""

    def __init__(self, base_url: str, headers: Dict[str, str] | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._timeout = timeout
        self._sse_timeout = sse_read_timeout
        self._connector: Optional[HttpConnector] = None
        self._session: Optional[MCPSession] = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        ChatLogger().event("mcp_use.connect", transport="http", url=self._base_url)
        connector_kwargs: Dict[str, Any] = {"base_url": self._base_url}
        if self._headers:
            connector_kwargs["headers"] = self._headers
        if self._timeout is not None:
            connector_kwargs["timeout"] = self._timeout
        if self._sse_timeout is not None:
            connector_kwargs["sse_read_timeout"] = self._sse_timeout

        self._connector = HttpConnector(**connector_kwargs)
        session = MCPSession(self._connector)
        await session.initialize()
        self._session = session

    async def close(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.disconnect()
        finally:
            if self._connector is not None:
                # Ensure underlying connector shuts down if disconnect didn't already
                disconnect = getattr(self._connector, "disconnect", None)
                if disconnect is not None:
                    result = disconnect()
                    if asyncio.iscoroutine(result):
                        await result
        self._session = None
        self._connector = None

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("MCP client not connected")
        result = await self._session.call_tool(name, arguments)
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
            return str(result)
