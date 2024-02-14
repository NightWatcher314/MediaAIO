from datetime import timedelta
import shutil
import sys
import time
import hashlib
import os
import subprocess
from typing import Union
import warnings
import torch
from tqdm import tqdm
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import urllib
import whisper
from config import base_models_dir

model_list = ["Qwen1.5-7B-Chat-GPTQ-Int8"]

qwen_models_dir = os.path.join(base_models_dir, "Qwen")
device = "cuda" if torch.cuda.is_available() else "cpu"


def whisper_command_warp(
    file_path,
    output_dir=None,
    language="Chinese",
    model="base",
    format="txt",
):
    command = f"whisper {file_path} --model {model} --language {language} --verbose True\
        --output_dir {output_dir} --output_format {format}"
    return command


def exec_whisper_command(
    file_path="", output_dir="", language="", model_type="", format=""
):
    _check_and_download_model(model_type)
    # model_path = _get_model_path(model_type)
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     model_path,
    #     torch_dtype=torch_dtype,
    #     low_cpu_mem_usage=True,
    #     use_safetensors=True,
    # )
    # model.to(device)
    # processor = AutoProcessor.from_pretrained(model_path)
    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     max_new_tokens=128,
    #     chunk_length_s=30,
    #     batch_size=16,
    #     return_timestamps=True,
    #     torch_dtype=torch_dtype,
    #     device=device,
    # )
    # result = pipe(
    #     file_path, generate_kwargs={"task": "transcribe", "language": language}
    # )
    # if format == "srt" and output_dir != "":
    #     result = _parse_result_to_srt(result)
    #     output_path = os.path.join(output_dir, "output.srt")
    #     with open(output_path, "w", encoding="utf-8") as srtFile:
    #         srtFile.write(result)


def _get_model_path(model_type):
    model_path = os.path.join(qwen_models_dir, model_type)
    return os.path.join(model_path, "snapshots", "model")


def _check_and_download_model(model_type):
    # model_path = os.path.join(qwen_models_dir, model_type)
    # if os.path.exists(model_path):
    #     return os.path.join(model_path, "snapshots", "model")
    # model_id = f"openai/whisper-{model_type}"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int8")
    # download_path = os.path.join(
    #     f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--openai--whisper-{model_type}"
    # )
    # shutil.move(os.path.join(download_path), model_path)
    # for dir in os.listdir(model_path):
    #     if dir == "snapshots":
    #         for ddir in os.listdir(os.path.join(model_path, dir)):
    #             shutil.move(
    #                 os.path.join(model_path, dir, ddir),
    #                 os.path.join(model_path, dir, "model"),
    #             )
    # del model, processor
    # return os.path.join(model_path, "snapshots", "model")
