"""Teradata agent implementation.

This agent is responsible for interacting with the Teradata MCP server
via the `mcp_use` client and the LangChainAdapter. It detects SQL
queries produced as intermediate tool outputs and collects them into
``state['sql_queries']`` for display in the UI.
"""

from mcp_use import MCPClient
from dotenv import load_dotenv
from mcp_use.agents.adapters import LangChainAdapter
from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os
import re
import json
import textwrap
from string import Template
from typing import Self, Union
from typing_extensions import override

from agents import BaseAgent
from modules.logger import logger
from states import MultiAgentState
from constants import TERADATA_AGENT_SYSTEM_PROMPT_PATH, CHARTS_PATH, ENV_PATH

load_dotenv(ENV_PATH)


class TeradataAgent(BaseAgent):
    """Agent that executes queries against Teradata via MCP.

    The TeradataAgent configures an MCP client and creates LangChain
    tools from it using ``LangChainAdapter``. When the agent runs it
    inspects intermediate tool outputs (MCP observations) and, if SQL
    is present, formats and returns it attached to the multi-agent
    state so it can be shown in the UI.
    """

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        self.mcp_config = {
            "mcpServers": {
                "teradata": {
                    "command": "uvx",
                    "args": ["teradata-mcp-server"],
                    "env": {
                        "TD_NAME": os.getenv("TD_NAME"),
                        "TD_HOST": os.getenv("TD_HOST"),
                        "TD_USER": os.getenv("TD_USER"),
                        "TD_PASSWORD": os.getenv("TD_PASSWORD"),
                        "TD_PORT": os.getenv("TD_PORT"),
                        "MCP_TRANSPORT": os.getenv("MCP_TRANSPORT"),
                        "DATABASE_URI": os.getenv("DATABASE_URI"),
                    }
                }
            }
        }
        with open(str(TERADATA_AGENT_SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
            content = Template(f.read())

        system_prompt = content.safe_substitute(
            database_name=os.getenv("TD_NAME"),
            charts_path=CHARTS_PATH
        )
        super().__init__(llm, memory, system_prompt)

        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.adapter = LangChainAdapter()

    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        """Create the TeradataAgent and prepare MCP-backed tools.

        Parameters
        ----------
        llm:
            LangChain language model used by this agent.
        memory:
            Shared conversation memory.

        Returns
        -------
        TeradataAgent
            A fully configured teradata agent ready to call MCP tools.
        """
        self = cls(llm, memory)
        self.tools = await self.adapter.create_tools(self.client)

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps,
        )

        return self

    def _process_intermediate_logs(self, response) -> Union[str, None]:
        """Extract SQL statements from MCP intermediate tool outputs.

        Parameters
        ----------
        response:
            The raw agent response dictionary returned by the agent executor.

        Returns
        -------
        Optional[str]
            Markdown-formatted SQL block when SQL queries are detected;
            otherwise None.
        """
        found_sql = False
        sql_message = ["\n\n**SQL Commands:**\n"]

        intermediate_steps = response.get("intermediate_steps", [])
        intermediate_steps_len = len(intermediate_steps)
        for i, (action, observation) in enumerate(intermediate_steps, start=1):
            logger.log(f"[Step {i}/{intermediate_steps_len}]", "")
            logger.log("[Used Tool]", action.tool)

            if not isinstance(observation, list):
                match = re.search(r"text='(.*)'", observation)
                if match:
                    observation = match.group(1).encode("utf-8").decode("unicode_escape")

            try:
                obs_json = json.loads(observation) if observation else {}
            except (json.JSONDecodeError, TypeError):
                obs_json = {}

            status = obs_json.get("status")
            if status:
                logger.log("[MCP Status]", status)
                if status == "success":
                    results = obs_json.get("results")
                    logger.log("[MCP Results]", str(results))
                    sql_query = obs_json.get("metadata", {}).get("sql")
                    if sql_query:
                        found_sql = True
                        logger.log("[MCP SQL Query]", sql_query)
                        wrapped_sql = textwrap.fill(sql_query, width=100)
                        sql_message.append(f"\n```sql\n{wrapped_sql}\n```\n")

        final_sql_messages = "\n".join(sql_message) if found_sql else None

        return final_sql_messages

    @override
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Invoke the Teradata agent and attach responses and SQL to state.

        Parameters
        ----------
        state:
            The current MultiAgentState containing the manager explanation.

        Returns
        -------
        MultiAgentState
            Updated state containing ``td_agent_response`` and optionally
            ``sql_queries`` when SQL was produced.

        Notes
        -----
        If an exception occurs during execution the exception is stored
        in ``state['td_agent_response']`` and the state is returned.
        """
        logger.log("[Agent]", "teradata")
        explanation = state.get("explanation", None)
        input_message = f"Manager Request:\n{explanation}"

        try:
            response = await self.agent_executor.ainvoke(
                {"input": input_message},
            )
        except ValueError as e:
            logger.log("[Error]", f"{e}")
            state["td_agent_response"] = e
            return state

        state["td_agent_response"] = response["output"]
        sql_messages = self._process_intermediate_logs(response)
        state["sql_queries"] = sql_messages

        logger.log("[Teradata Agent Output]", response["output"])

        return state