"""Manager agent implementation.

The ManagerAgent acts as the decision-maker: given the user query and
previous agent outputs it decides whether the multi-agent flow should
query Teradata, create a plot, or finish. It uses a tool-calling agent
executor under the hood and parses the (expected) JSON decision payload
returned by the LLM.
"""

from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent

import json
from typing import Self
from typing_extensions import override

from agents import BaseAgent
from modules.logger import logger
from states import MultiAgentState
from constants import MANAGER_AGENT_SYSTEM_PROMPT_PATH


class ManagerAgent(BaseAgent):
    """Agent responsible for routing and high-level decisions.

    Expected behaviour
    ------------------
    - Build a system prompt from `MANAGER_AGENT_SYSTEM_PROMPT_PATH`.
    - Use a tool-calling agent executor (tools are empty for now).
    - When invoked, include the user query and other agents' outputs
      in the prompt, then parse a JSON decision structure from the LLM
      response. The parsed decision sets ``state['manager_decision']``.
    """

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        with open(str(MANAGER_AGENT_SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
            system_prompt = f.read()

        super().__init__(llm, memory, system_prompt)

    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        """Asynchronously create a configured ManagerAgent instance.

        Parameters
        ----------
        llm:
            LangChain language model used by this agent.
        memory:
            Shared conversation memory.

        Returns
        -------
        ManagerAgent
            The initialized manager agent ready for use.
        """
        self = cls(llm, memory)
        self.tools = []

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps
        )

        return self

    @override
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Run the manager logic and update the provided state.

        Parameters
        ----------
        state:
            The current MultiAgentState containing the user query and
            any previous agent outputs.

        Returns
        -------
        MultiAgentState
            The updated state with fields set such as ``manager_decision``,
            ``response``, and ``explanation``.
        """
        logger.log("[Agent]", "manager")
        user_query = f"User Query:\n{state['user_query']}"
        td_agent_response = state.get("td_agent_response", None)
        plot_agent_response = state.get("plot_agent_response", None)

        if td_agent_response is not None:
            user_query += f"\n\nTeradata Agent Response:\n{td_agent_response}"
        if plot_agent_response is not None:
            user_query += f"\n\nPlot Agent Response:\n{plot_agent_response}"

        response = await self.agent_executor.ainvoke(
            {"input": user_query},
        )

        decision, message, explanation = None, None, None
        try:
            response = response["output"]
            response = response.replace("```json", "")
            response = response.replace("}\n```", "")
            response = response.replace("}```", "")
            response = json.loads(response)
            decision = response["decision"].lower()
            message = response["message"]
            explanation = response["explanation"]
        except:
            if isinstance(response, dict):
                text = response.get("output", "")
            else:
                text = str(response)

            decision = "done"
            message = text
            explanation = text

        if "teradata" in decision:
            state["manager_decision"] = "teradata"
        elif "plot" in decision:
            state["manager_decision"] = "plot"
        elif "done" in decision:
            state["manager_decision"] = "done"
        else:
            state["manager_decision"] = "done"

        state["response"] = message if message is not None else response["output"]
        state["explanation"] = explanation if explanation is not None else None
        state["messages"].append({"role": "manager", "content": state["response"]})

        logger.log("[Manager Decision]", decision)
        logger.log(f"[Manager Explanation]", explanation)
        logger.log("[Manager Agent Output]", str(response))

        return state