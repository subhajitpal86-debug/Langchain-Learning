from hf_setup import get_hf_chat_model

# Initialize the standalone ChatHuggingFace model
model = get_hf_chat_model(temperature=0.1)

response = model.invoke("Hello, what is Python?")
print(response.content)
