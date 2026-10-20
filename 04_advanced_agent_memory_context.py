import requests
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from hf_setup import get_hf_chat_model

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool
def get_weather(city: str):
    """Return weather information for a given city."""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

@tool
def locate_user(runtime: ToolRuntime[Context]):
    """Look up a user's city based on the context."""
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "Unknown"

chat_model = get_hf_chat_model(temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
    model=chat_model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant...",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}
context = Context(user_id="ABC123")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather like?"}]},
    config=config,
    context=context
)

print(response.structured_response.summary)
print(response.structured_response.temperature_celsius)
