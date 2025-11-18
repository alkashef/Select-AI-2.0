"""Base classes and types used by agent implementations.

This module defines a lightweight TypedDict for chat messages and an
abstract BaseAgent class that concrete agents must implement. The
BaseAgent centralizes common configuration read from environment
variables (e.g. verbosity and iteration limits) and defines the
factory-style ``create`` constructor as well as a callable interface
``__call__`` that receives and returns a ``MultiAgentState``.
"""

from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from typing import Self
from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Optional

from states import MultiAgentState


class Message(TypedDict, total=False):
    """Typed representation of a single chat message.

    Fields
    ------
    role:
        The role of the message author (e.g. 'user', 'ai', 'assistant', 'system').
    content:
        The textual content of the message.
    ts:
        Optional ISO-8601 timestamp string when the message was created.
    """
    role: Literal["user", "ai", "assistant", "system"]
    content: str
    ts: Optional[str]


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement the asynchronous factory ``create`` which
    should prepare any heavy resources (tools, adapters) and return an
    instance. They must also implement ``__call__`` which executes the
    agent's logic for a given ``MultiAgentState`` and returns an updated
    state.

    Common attributes provided by this base class:
    - llm: a LangChain language model instance
    - memory: conversation memory used across agents
    - max_iterations: agent loop limit (from env MAX_ITERATIONS)
    - verbose: enable verbose logging (from env VERBOSE)
    - return_intermediate_steps: whether the agent executor should return steps
    - tools, agent_executor: placeholders for tool/open-agent plumbing
    """

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory, system_prompt: str) -> None:
        super().__init__()
        self.llm = llm
        self.memory = memory
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 30))
        self.verbose = eval(os.getenv("VERBOSE", False))
        self.return_intermediate_steps = eval(os.getenv("RETURN_INTERMEDIATE_STEPS", False))

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.tools = None
        self.agent_executor = None

    @classmethod
    @abstractmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        """Asynchronously create and configure an agent instance.

        Parameters
        ----------
        llm:
            A LangChain BaseLanguageModel used by the agent for LLM calls.
        memory:
            A LangChain chat memory instance shared across agents.

        Returns
        -------
        Self
            A fully configured instance of the agent subclass.

        Notes
        -----
        Implementations should perform any IO-bound setup here (tool
        construction, adapters) and return the ready-to-use agent.
        """
        ...

    @abstractmethod
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Execute the agent's logic for the provided state.

        Parameters
        ----------
        state:
            The current MultiAgentState to be processed by the agent.

        Returns
        -------
        MultiAgentState
            The updated state after the agent has run.

        Notes
        -----
        Implementations should avoid side-effects outside of the
        returned state unless explicitly required (e.g., writing files).
        """
        ...