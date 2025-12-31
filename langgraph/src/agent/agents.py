from typing import TypedDict
from langchain.agents import create_agent
from agent.llm import chatdeepseek
from agent.tools import add, multiply
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.messages import SystemMessage, HumanMessage

model = chatdeepseek

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent_1 = create_agent(model, 
                       tools=[add,multiply], 
                       middleware=[handle_tool_errors],
                       system_prompt=SystemMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": "You are an AI assistant tasked with analyzing literary works.",
                                },
                                {
                                    "type": "text",
                                    "text": "<the entire contents of 'Pride and Prejudice'>",
                                    "cache_control": {"type": "ephemeral"}
                                }
                            ]
                        ))


class Context(TypedDict):
    user_role: str
    
from langchain.agents.middleware import dynamic_prompt, ModelRequest
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent_1 = create_agent(model, 
                       tools=[add,multiply], 
                       middleware=[handle_tool_errors, user_role_prompt],
                       context_schema=Context)