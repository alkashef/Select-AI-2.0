"""Concrete orchestrator that wires Manager, Teradata and Plot agents.

This implementation uses `langgraph` to build a small state graph where
the manager decides which agent to call next. The graph cycles until a
terminal END state is reached. The `run` method prepares an initial
MultiAgentState and triggers the compiled app asynchronously.
"""

from langgraph.graph import StateGraph, END
from langchain.memory.chat_memory import BaseChatMemory

from typing import override

from states import MultiAgentState
from multi_agents import BaseMultiAgent
from constants import LANGGRAPH_GRPAH_IMAGE_PATH
from agents import (
    PlotAgent,
    ManagerAgent,
    TeradataAgent
)


class MultiAgent(BaseMultiAgent):

    def __init__(
        self,
        memory: BaseChatMemory,
        manager_agent: ManagerAgent,
        plot_agent: PlotAgent,
        teradata_agent: TeradataAgent,
    ) -> None:
        super().__init__()
        self.memory = memory
        self.manager_agent = manager_agent
        self.plot_agent = plot_agent
        self.teradata_agent = teradata_agent

        self.graph = StateGraph(MultiAgentState)
        self._build_graph()
        self.app = self.graph.compile()

    def route_decision(self, state: MultiAgentState) -> str:
        """Decide next node name from the current state.

        If the state contains ``done`` or an unknown decision the function
        returns the graph END token.
        """
        if state.get("done"):
            return END

        decision = state.get("manager_decision", "")
        if "teradata" in decision:
            return "teradata"
        elif "plot" in decision:
            return "plot"
        else:
            return END

    @override
    def _build_graph(self) -> None:
        """Construct the state graph connecting agents.

        This internal helper registers nodes and conditional edges used
        by the compiled state machine.
        """

        self.graph.add_node("manager", self.manager_agent)
        self.graph.add_node("teradata", self.teradata_agent)
        self.graph.add_node("plot", self.plot_agent)

        self.graph.add_conditional_edges(
            "manager", 
            self.route_decision,
            {
                "teradata": "teradata",
                "plot": "plot",
                END: END
            }
        )

        self.graph.add_edge("teradata", "manager")
        self.graph.add_edge("plot", "manager")
        self.graph.set_entry_point("manager")

    @override
    async def run(self, user_query: str) -> MultiAgentState:
        """Run the multi-agent graph for a user query and return final state.

        Parameters
        ----------
        user_query:
            The natural language query from the user to process.

        Returns
        -------
        MultiAgentState
            The final state after the orchestrator has completed.
        """
        if not isinstance(user_query, str):
            raise ValueError(f"User query must be a string.")
        
        state = MultiAgentState(
            is_plot=False,
            user_query=user_query,
            messages=self.memory.load_memory_variables({})["chat_history"]
        )

        final_state = await self.app.ainvoke(state)

        return final_state
    
    @override
    def visualize(self):
        """Save a PNG rendering of the compiled state graph to disk."""
        with open(LANGGRAPH_GRPAH_IMAGE_PATH, "wb") as f:
            f.write(self.app.get_graph().draw_mermaid_png())