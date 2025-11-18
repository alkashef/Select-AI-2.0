
"""Typed state definitions used to pass data between agents.

These simple TypedDicts document the expected keys and types used by
the multi-agent orchestrator and agents when exchanging information.
"""

from typing import TypedDict, List, Dict, Optional


class BaseState(TypedDict):
    """Minimal base state stored and passed between agents.

    Fields
    ------
    user_query:
        The original user query string.
    messages:
        Conversation history as a list of dict-like messages.
    response:
        Optional textual response produced by an agent.
    """
    user_query: str
    messages: List[Dict]
    response: Optional[str]