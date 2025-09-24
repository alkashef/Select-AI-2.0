"""OpenAI + MCP backend (minimal, clean rebuild).

Temporary minimal implementation to resolve indentation issues while keeping
public API stable for tests. Expand incrementally after compile check passes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from config import get_openai_config
from .base import AI, Message


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
        self.tools_catalog_path = os.getenv("MCP_TOOLS_CATALOG", os.path.join("config", "mcp_tools.yml"))
        self._metadata_loaded = False
        self.metadata: Dict[str, Any] = {"database": self.default_db, "tables": [], "columns": {}, "samples": {}}
        self._dq_cache: Dict[Tuple[str, str, str], str] = {}

    # Public API
    def generate_reply(self, messages: List[Message], context: Dict | None = None) -> str:
        if not messages:
            return ""
        return "Phase 1 online. Ask about schema or DQ."

    # Minimal helpers used in other parts of the file/tests later
    def warmup(self) -> None:
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        if self._metadata_loaded:
            return
        # Best-effort: use any existing snapshot; otherwise keep defaults
        try:
            if os.path.exists(self.schema_path):
                with open(self.schema_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self.metadata.update(data)
        except Exception:
            pass
        self._metadata_loaded = True
