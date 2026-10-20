import requests
from langchain.agents import create_agent
from langchain.tools import tool
from hf_setup import get_hf_chat_model

@tool
def get_weather(city: str):
    """Return weather information for a given city."""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

chat_model = get_hf_chat_model()

agent = create_agent(
    model=chat_model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant who always cracks jokes and is humorous while remaining helpful."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in Vienna?"}
    ]
})

print(response["messages"][-1].content)
