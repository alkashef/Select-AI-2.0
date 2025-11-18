"""Configuration helper to load project .env and expose settings.

This module exposes a small ``Config`` class that loads ``config/.env``
via python-dotenv and provides a typed representation of a couple of
commonly used settings (logging enabled, log file path). The loader
intentionally raises when the expected .env file is missing to make
configuration errors explicit during startup.
"""

from pathlib import Path
from dotenv import load_dotenv

import os
from typing import Optional

from constants import ENV_PATH


class Config:
    """Strict loader for environment-backed settings from ``config/.env``.

    - Requires config/.env to exist, else raises FileNotFoundError.
    - Uses python-dotenv to load variables into the process environment.
    """
    
    def __init__(self, env_path: Path, log_enabled: bool, log_file: Path) -> None:
        self.env_path = env_path
        self.log_enabled = log_enabled
        self.log_file = log_file
    
    @staticmethod
    def _env_bool(name: str, default: str = "false") -> bool:
        """Parse a boolean-ish environment variable value.

        Recognizes 1/true/yes/on (case-insensitive) as truthy values.
        """
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}
    
    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "Config":
        """Load and return a Config instance.

        If the expected ``ENV_PATH`` does not exist a FileNotFoundError is
        raised. The loader uses ``load_dotenv(..., override=False)`` so
        that existing environment variables keep precedence.
        """
        if not ENV_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {ENV_PATH}")
        # Load .env, but do not override existing environment variables by default
        # so that OS/envvars (e.g., pytest monkeypatch) take precedence.
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    
        log_enabled = cls._env_bool("LOG_ENABLED", "false")
        log_file = Path(os.getenv("LOG_FILE", "logs/log.txt").strip())
        return cls(env_path=ENV_PATH, log_enabled=log_enabled, log_file=log_file)