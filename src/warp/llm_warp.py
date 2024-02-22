from datetime import timedelta
from typing import Union

import torch
import transformers
import whisper
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

qwen_model_list = ["Qwen1.5-7B-Chat-GPTQ-Int8"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_id = None


def exec_qwen_command(
    file_path="", output_dir="", language="", model_type="", format=""
):
    pass


def load_model(new_model_id):
    global model, model_id
    if model is None:
        model_id = new_model_id
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device)
    if model_id != new_model_id:
        model_id = new_model_id
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device)


def _test_llm():
    global model
    model_id = "touqir/Cyrax-7B"
    load_model(model_id)
    messages = [{"role": "user", "content": "Huggingface是什么,用中文回答?"}]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    print(outputs[0]["generated_text"])
