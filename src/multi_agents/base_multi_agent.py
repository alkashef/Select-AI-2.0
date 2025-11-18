
"""Abstract base for multi-agent orchestrators.

Multi-agent orchestrators implement a small contract: build the state
graph, run the loop for a user query, and provide a visualization
helper. This base class documents that contract.
"""

from abc import ABC, abstractmethod

from states import BaseState


class BaseMultiAgent(ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def _build_graph(self) -> None:
        """Construct internal graph of agent nodes and transitions."""
        ...

    @abstractmethod
    async def run(self, user_query: str) -> BaseState:
        """Execute the multi-agent flow for the given user query.

        Should return the final BaseState produced by the orchestrator.
        """
        ...

    @abstractmethod
    def visualize(self) -> None:
        """Emit or save a visual representation of the agent graph.

        Implementations may write images to disk or return in-memory
        objects. This is an optional convenience for debugging.
        """
        ...