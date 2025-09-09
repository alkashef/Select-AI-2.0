from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """Helper to strictly load and expose environment-backed settings from config/.env.

    - Requires config/.env to exist, else raises FileNotFoundError.
    - Uses python-dotenv to load variables into the process environment.
    """

    def __init__(self, env_path: Path, log_enabled: bool, log_file: Path) -> None:
        self.env_path = env_path
        self.log_enabled = log_enabled
        self.log_file = log_file

    @staticmethod
    def _env_bool(name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "Config":
        base = base_dir or Path(__file__).parent
        env_path = base / "config" / ".env"
        if not env_path.exists():
            raise FileNotFoundError(f"Config file not found: {env_path}")

        load_dotenv(dotenv_path=env_path)

        log_enabled = cls._env_bool("LOG_ENABLED", "false")
        log_file = Path(os.getenv("LOG_FILE", "log.txt").strip())
        return cls(env_path=env_path, log_enabled=log_enabled, log_file=log_file)


def get_openai_config(base_dir: Optional[Path] = None) -> dict:
    """Load OpenAI settings from config/.env and return a ready client + settings.

    Returns a dict with keys: {"api_key", "model", "client"}.
    Raises FileNotFoundError if config/.env is missing, or RuntimeError if the
    required OPENAI_API_KEY is not set.
    """
    # Ensure .env is loaded and exists (reuses Config side-effect to load)
    Config.load(base_dir=base_dir)

    # Import locally to avoid hard dependency for non-OpenAI flows
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or config/.env")

    model = os.getenv("GPT_MODEL", "gpt-4o").strip() or "gpt-4o"

    # Optional timeout to avoid hanging calls
    try:
        timeout = int(os.getenv("OPENAI_TIMEOUT", "20").strip())
    except ValueError:
        timeout = 20

    # Optional advanced settings
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    organization = os.getenv("OPENAI_ORG", "").strip() or None
    project = os.getenv("OPENAI_PROJECT", "").strip() or None

    client_kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    if organization:
        client_kwargs["organization"] = organization
    if project:
        client_kwargs["project"] = project

    client = OpenAI(**client_kwargs)
    return {"api_key": api_key, "model": model, "client": client}


def get_ai_backend(base_dir: Optional[Path] = None) -> str:
    """Return AI_BACKEND from env (defaults to 'gpt'), ensuring .env is loaded."""
    Config.load(base_dir=base_dir)
    return os.getenv("AI_BACKEND", "gpt").strip().lower() or "gpt"
