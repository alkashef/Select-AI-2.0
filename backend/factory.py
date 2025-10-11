"""Factory helpers to create AI backends."""

from __future__ import annotations

from .agent import Agent, GPTMCPAgent
from .config import load_settings


async def get_ai() -> Agent:
    settings = load_settings()
    return await GPTMCPAgent.create(settings)
