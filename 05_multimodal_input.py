from langchain.messages import HumanMessage
from base64 import b64encode
from hf_setup import get_hf_chat_model

model = get_hf_chat_model()

# Example with URL
message = HumanMessage(content=[
    {"type": "text", "text": "Describe the contents of this image."},
    {"type": "image_url", "image_url": {"url": "https://neuralnine.com/logo.png"}}
])

# NOTE: Qwen2.5-1.5B-Instruct is a text-only model.
# Multimodal inputs require a Vision-Language Model (VLM).
# This code perfectly matches the video's LangChain structure, 
# but will raise a model-level exception unless you swap the model_id for a VLM.
try:
    response = model.invoke([message])
    print(response.content)
except Exception as e:
    print(f"Error: {e}\n(Reminder: Qwen2.5-1.5B is text-only. Use a multimodal model for image ingestion.)")
