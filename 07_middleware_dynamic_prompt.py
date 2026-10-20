from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from hf_setup import get_hf_chat_model

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful and very concise assistant."
    
    match user_role:
        case "expert":
            return f"{base_prompt} Provide detailed technical responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple and basic."
        case "child":
            return f"{base_prompt} Explain everything as if you were literally talking to a 5-year-old."
        case _:
            return base_prompt

agent = create_agent(
    model=get_hf_chat_model(),
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain PCA"}]},
    context=Context(user_role="child")
)

print(response["messages"][-1].content)
