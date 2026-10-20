from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from hf_setup import get_hf_chat_model

# Replaced OpenAIEmbeddings with free local Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "I love apples.",
    "I enjoy oranges.",
    "I think pears taste very good.",
    "I hate bananas.",
    "I dislike raspberries.",
    "I despise mangoes.",
    "I love Linux.",
    "I hate Windows."
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search the small product/fruit database for information."
)

agent = create_agent(
    model=get_hf_chat_model(),
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant. Use the tool to retrieve context and answer in a concise way."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}
    ]
})

print(response["messages"][-1].content)
