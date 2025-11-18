"""Plot agent implementation.

The PlotAgent is responsible for generating visualizations. It uses a
Python AST REPL tool to execute plotting code in a sandboxed manner and
writes resulting images to the configured charts path. When the agent
executes a plotting tool it sets ``state['is_plot'] = True`` so the UI
can load the generated image.
"""

from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from typing import Self
from string import Template
from typing_extensions import override

from agents import BaseAgent
from modules.logger import logger
from states import MultiAgentState
from constants import CHARTS_PATH, PLOT_AGENT_SYSTEM_PROMPT_PATH


class PlotAgent(BaseAgent):
    """Agent responsible for generating charts and visualization artifacts.

    The system prompt is templated with the ``CHARTS_PATH`` so the LLM can
    reference where to write images. The agent exposes the usual
    asynchronous ``create`` factory and a callable ``__call__``.
    """

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        with open(str(PLOT_AGENT_SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
            content = Template(f.read())

        system_prompt = content.safe_substitute(
            charts_path=CHARTS_PATH
        )
        super().__init__(llm, memory, system_prompt)


    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        """Create a PlotAgent and prepare the Python execution tool.

        Parameters
        ----------
        llm:
            LangChain language model used by this agent.
        memory:
            Shared conversation memory.

        Returns
        -------
        PlotAgent
            Initialized PlotAgent with Python execution tool available.
        """
        self = cls(llm, memory)
        self.tools = [PythonAstREPLTool()]

        agent = create_tool_calling_agent(llm=llm, tools=self.tools, prompt=self.prompt)
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
        """Invoke the plotting agent and update state with results.

        Parameters
        ----------
        state:
            Current MultiAgentState containing manager explanation.

        Returns
        -------
        MultiAgentState
            Updated state with ``is_plot`` possibly set to True and
            ``plot_agent_response`` populated with agent output.
        """
        logger.log("[Agent]", "plot")
        explanation = state.get("explanation", None)
        input_message = f"Manager Request: {explanation}"

        response = await self.agent_executor.ainvoke(
            {"input": input_message},
        )
        if "intermediate_steps" in response and len(response["intermediate_steps"]) > 0:
            action, _ = response["intermediate_steps"][0]
            logger.log("[Used Tool]", action.tool)
            state["is_plot"] = True

        state["plot_agent_response"] = response["output"]
        logger.log("[Plot Agent Output]", response["output"])

        return state