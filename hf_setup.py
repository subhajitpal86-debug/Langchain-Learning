import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

def get_hf_chat_model(temperature=0.7):
    # 1. Load model and tokenizer locally
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # MODIFIED FOR CODESPACES (CPU ONLY & Memory Optimization)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Forces lower memory usage on CPU
        device_map="cpu",           # Explicitly target CPU
        low_cpu_mem_usage=True      # Optimizes memory allocation
    )

    # 2. Setup local pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=True if temperature > 0 else False
    )

    # 3. Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return ChatHuggingFace(llm=llm)
