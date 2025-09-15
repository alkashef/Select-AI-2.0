"""Deprecated wrapper: use `tests.test_mcp_diag` instead.

This script now delegates to `tests.test_mcp_diag` to avoid duplication.
It accepts the same common flags and then runs the diagnostics, which
also lists tools.

Usage (Windows cmd):
    python -m tests.test_mcp_diag --config tests\mcp_from_app.yml
    python -m tests.test_mcp_diag --url http://localhost:8001/mcp/
"""

from __future__ import annotations

import argparse
import asyncio

from tests.mcp_helpers import load_env_from_config
from tests.test_mcp_diag import main as diag_main


async def main() -> int:
    load_env_from_config()
    parser = argparse.ArgumentParser(description="Deprecated: use tests.test_mcp_diag (this delegates to it)")
    parser.add_argument("--config", default="tests/mcp_from_app.yml", help="From-app HTTP spawn config")
    parser.add_argument("--url", default=None, help="If provided, skip spawn and use this MCP URL")
    _ = parser.parse_args()  # Keep interface stable; diag will parse again.
    return await diag_main()  # Delegate to diagnostics (lists tools too)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
