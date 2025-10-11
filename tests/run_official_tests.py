#!/usr/bin/env python3
"""Convenience wrapper to execute the official Teradata MCP test suite.

This script loads environment variables from the project .env file and then
invokes the documented test runner with the correct server command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def build_command() -> list[str]:
    database_uri = os.getenv("DATABASE_URI")
    if not database_uri:
        print("✗ DATABASE_URI is not set. Update .env or export it before running.", file=sys.stderr)
        sys.exit(1)

    profile = os.getenv("PROFILE", "tester")
    logging_level = os.getenv("LOGGING_LEVEL", "INFO")

    server_command = (
        "uv run teradata-mcp-server "
        f"--logging_level {logging_level} "
        f"--profile {profile} "
        f"--database_uri {database_uri}"
    )

    cmd = [
    "uv",
    "run",
    "python",
    "tests/run_mcp_tests.py",
    server_command,
    "tests/core_test_cases.json",
    "--verbose",
    ]

    additional_args = sys.argv[1:]
    if additional_args:
        cmd.extend(additional_args)

    return cmd


def main() -> None:
    command = build_command()

    print("Executing official MCP test suite:\n  " + " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
