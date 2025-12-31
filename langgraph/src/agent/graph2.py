"""Fixed LangGraph Agent with DeepSeek + Tool"""

from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import SystemMessage, HumanMessage

from agent.llm import chatdeepseek
from agent.tools import add
from langchain.agents import create_agent
graph = create_agent(chatdeepseek, tools=[add], system_prompt="Your are a helpful assistant.", name="New Graph")