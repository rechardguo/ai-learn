import os
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
    "bar",
    AIMessage(content="Sydney is in Australia, UTC+10"),
    AIMessage(content="Current time in Sydney is 10:00 AM"),
    AIMessage(content="Weather in San Francisco is sunny, 72°F"),
]))

response = model.invoke("hello")
print(response)
print("----")
response = model.invoke("hello, again!")
print(response)
print("----")

from langchain.agents import create_agent
os.path.abspath('some/relative/path')
agent = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver()
)

config = RunnableConfig({"configurable": {"thread_id": "1"}})

# First invocationresult
result = agent.invoke({"messages": [HumanMessage(content="I live in Sydney, Australia.")]}, config=config)
print(result)
print("----")
# Second invocation: the first message is persisted (Sydney location), so the model returns GMT+10 time
result = agent.invoke({"messages": [HumanMessage(content="What's my local time?")]}, config=config)
print(result)
print("----")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config
)
print(result)