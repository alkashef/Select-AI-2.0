"""Deprecated: this script has been removed.

Use diagnostics instead:
  python -m tests.test_mcp_diag --config tests\mcp_from_app.yml
"""

from __future__ import annotations

import asyncio

from tests.mcp_helpers import load_env_from_config
from tests.test_mcp_diag import main as diag_main


async def main() -> int:
    load_env_from_config()
    # Delegate to diagnostics to avoid duplication
    return await diag_main()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
