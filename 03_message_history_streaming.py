from langchain.messages import HumanMessage, AIMessage, SystemMessage
from hf_setup import get_hf_chat_model

model = get_hf_chat_model(temperature=0.7)

conversation = [
    SystemMessage(content="You are a helpful assistant for questions regarding programming."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is an interpreted programming language."),
    HumanMessage(content="When was it released?")
]

# Streaming example 
# (Note: Standard HF pipelines might batch output unless configured with a TextIteratorStreamer, 
# but the LangChain interface works the exact same way)
for chunk in model.stream(conversation):
    print(chunk.content, end="", flush=True)
