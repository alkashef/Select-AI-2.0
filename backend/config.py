"""Runtime configuration helpers for the Streamlit + MCP prototype."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class LoggingSettings:
    enabled: bool
    file: Path
    level: str


@dataclass(frozen=True)
class OpenAISettings:
    model: str
    timeout: int


@dataclass(frozen=True)
class MCPSettings:
    command: List[str]
    cwd: Path
    env: dict[str, str]
    max_steps: int


@dataclass(frozen=True)
class AppPaths:
    repo_root: Path
    charts_dir: Path


@dataclass(frozen=True)
class Settings:
    logging: LoggingSettings
    openai: OpenAISettings
    mcp: MCPSettings
    paths: AppPaths


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)


def _bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_command(cmd: str) -> List[str]:
    return [part for part in cmd.split(" ") if part]


def load_settings() -> Settings:
    repo_root = _REPO_ROOT
    charts_dir = repo_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(os.getenv("LOG_FILE", "./logs/log.txt"))
    if not log_file.is_absolute():
        log_file = repo_root / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    logging_settings = LoggingSettings(
        enabled=_bool(os.getenv("LOG_ENABLED", "true")),
        file=log_file,
        level=log_level,
    )

    openai_settings = OpenAISettings(
        model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
    )

    database_uri = os.getenv("DATABASE_URI")
    profile = os.getenv("PROFILE", "tester")
    logging_level = log_level

    server_cmd = os.getenv("MCP_COMMAND")
    if server_cmd:
        command = _split_command(server_cmd)
    else:
        command = [
            "uv",
            "run",
            "teradata-mcp-server",
            "--logging_level",
            logging_level,
            "--profile",
            profile,
        ]
        if database_uri:
            command.extend(["--database_uri", database_uri])

    env = {**os.environ}
    env.setdefault("MCP_TRANSPORT", "stdio")

    mcp_settings = MCPSettings(
        command=command,
        cwd=repo_root,
        env=env,
        max_steps=int(os.getenv("MAX_STEPS", "25")),
    )

    return Settings(
        logging=logging_settings,
        openai=openai_settings,
        mcp=mcp_settings,
        paths=AppPaths(repo_root=repo_root, charts_dir=charts_dir),
    )
