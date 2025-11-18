"""Extended state used specifically by the MultiAgent orchestrator.

Contains fields used to coordinate between Manager, Teradata and Plot
agents such as whether a plot was produced, the manager explanation,
SQL snippets produced by the Teradata agent, and agent-specific
responses.
"""

from typing import Optional

from states import BaseState


class MultiAgentState(BaseState):
    """State container for the MultiAgent orchestrator.

    This class extends BaseState with fields that multiple agents use to
    coordinate work and record intermediate outputs.

    Parameters
    ----------
        is_plot (bool): True if the workflow produced a visualization/plot.
        explanation (Optional[str]): Human-readable explanation or reasoning
            produced by the manager agent.
        sql_queries (Optional[str]): SQL snippets or full queries generated
            by the Teradata agent; may contain multiple queries separated
            by newlines.
        manager_decision (Optional[str]): Final decision or selected action
            chosen by the manager agent (e.g., "execute", "refine", "abort").
        td_agent_response (Optional[str]): Raw or processed response returned
            by the Teradata agent (errors, execution results, or logs).
        plot_agent_response (Optional[str]): Output from the plot agent,
            such as a path/URL to the generated plot or a brief summary.
    """
    is_plot: bool
    explanation: Optional[str]
    sql_queries: Optional[str]
    manager_decision: Optional[str]
    td_agent_response: Optional[str]
    plot_agent_response: Optional[str]