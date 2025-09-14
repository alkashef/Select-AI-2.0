"""Helpers for MCP HTTP test scripts.

- load_env_from_config: loads config/.env if present
- print_mcp_result: pretty-prints common MCP tool outputs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_env_from_config() -> None:
    """Load environment variables from config/.env if it exists."""
    env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def print_mcp_result(result: Any) -> None:
    """Normalize and print MCP tool results consistently.

    Supports dict/JSON strings, or list of objects with a `.text` field.
    """
    if result is None:
        print("(no result)")
        return

    # Dicts print as pretty JSON
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
        return

    # Strings may be JSON
    if isinstance(result, str):
        try:
            obj = json.loads(result)
            print(json.dumps(obj, indent=2))
        except json.JSONDecodeError:
            print(result)
        return

    # Lists of text-bearing objects
    try:
        text = getattr(result[0], "text", None) if result else None
        if text:
            try:
                obj = json.loads(text)
                print(json.dumps(obj, indent=2))
            except json.JSONDecodeError:
                print(text)
        else:
            print(str(result))
    except Exception:
        print(str(result))
