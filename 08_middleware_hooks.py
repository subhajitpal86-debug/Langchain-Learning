import time
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from hf_setup import get_hf_chat_model

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state, runtime):
        self.start_time = time.perf_counter()
        print("before agent triggered")

    def before_model(self, request, handler):
        print("before model triggered")
        return handler(request)
        
    def after_model(self, response):
        print("after model triggered")
        return response

    def after_agent(self, state):
        diff = time.perf_counter() - self.start_time
        print(f"after agent: {diff}")
        return state

agent = create_agent(
    model=get_hf_chat_model(),
    middleware=[HooksDemo()]
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain PCA"}]
})

print(response["messages"][-1].content)
