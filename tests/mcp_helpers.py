"""Helpers for MCP HTTP test scripts.

- load_env_from_config: loads config/.env if present
- print_mcp_result: pretty-prints common MCP tool outputs
- load_yaml_config: reads a YAML config file safely
- resolve_mcp_url: derive MCP URL from CLI arg, YAML, or env
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import os

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None


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


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML file as a dict.

    Returns empty dict if file missing or YAML library unavailable.
    Raises ValueError if YAML root is not a mapping.
    """
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def resolve_mcp_url(arg_url: Optional[str], cfg: Optional[Dict[str, Any]] = None, default: str = "http://localhost:8001/mcp/") -> str:
    """Resolve MCP URL from CLI arg, YAML config, or environment.

    Priority: arg_url > cfg['mcp_url'] > env MCP_URL > default.
    """
    cfg = cfg or {}
    return arg_url or cfg.get("mcp_url") or os.getenv("MCP_URL", default)
