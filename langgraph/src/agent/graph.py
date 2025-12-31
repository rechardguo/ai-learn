"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from agent.llm import llm_chatdeepseek
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
from agent.tools import add

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str

class State(MessagesState):
    # 可选：保留 query 作为额外字段（通常不需要）
    pass

# @dataclass
# class State:
#     """Input state for the agent.

#     Defines the initial structure of incoming data.
#     See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
#     """

#     query: str = "example"

#上面的写法等同于下面的写法
# class State:
#     def __init__(self, query: str = "example"):
#         self.query = query

# async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
#     """Process input and returns output.

#     Can use runtime context to alter behavior.
#     """
#     return {
#         "changeme": "output from call_model. "
#         f"Configured with {(runtime.context or {}).get('my_configurable_param')}"
#     }



async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    llm = llm_chatdeepseek
    llm.bind_tools([add])
    # 添加系统提示（可选）
    messages = [SystemMessage(content="You are a helpful assistant."),state.query] 
    
    # 调用 LLM
    response = await llm.ainvoke(messages)
    
    return {"messages": [response]}  # 必须返回消息列表

# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
