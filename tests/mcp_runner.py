from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Dict, Tuple

from tests.mcp_helpers import load_env_from_config, load_yaml_config


def _wait_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def spawn_http_server(config_path: str) -> Tuple[subprocess.Popen, str]:
    """Spawn teradata-mcp-server (HTTP) using uvx and return (proc, mcp_url).

    Requires DATABASE_URI in config/.env or in YAML under env.DATABASE_URI.
    """
    load_env_from_config()
    cfg = load_yaml_config(config_path)
    server_cfg, env_cfg = cfg.get("server", {}), cfg.get("env", {})
    command, cmd_args = server_cfg.get("command", "uvx"), server_cfg.get("args", [])
    cwd = server_cfg.get("cwd")

    # Determine port and MCP URL
    try:
        port = int(cmd_args[cmd_args.index("--mcp_port") + 1])
    except Exception:
        port = 8001
    mcp_url = cfg.get("client", {}).get("mcp_url") or f"http://localhost:{port}/mcp/"

    env = os.environ.copy()
    dburi = env_cfg.get("DATABASE_URI") or env.get("DATABASE_URI")
    if not dburi:
        raise RuntimeError("DATABASE_URI is required in config/.env or tests/mcp_from_app.yml")
    env["DATABASE_URI"] = dburi
    for key in ("LOGMECH", "ENCRYPTDATA", "CHARSET"):
        val = env_cfg.get(key) or env.get(key)
        if val:
            env[key] = str(val)

    print("Command:", command, *cmd_args)
    print("MCP URL:", mcp_url)
    print("DATABASE_URI:", dburi)
    proc = subprocess.Popen([command, *cmd_args], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if not _wait_port("localhost", port, timeout=30):
        tail = ""
        try:
            tail = proc.stdout.read()[-2000:] if proc.stdout else ""
        except Exception:
            pass
        proc.terminate()
        raise RuntimeError(f"MCP server failed to open port {port}. Output tail:\n{tail}")

    return proc, mcp_url


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
    except Exception:
        pass
