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
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import urllib
import whisper
from config import base_models_dir

whisper_model_list = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
]
whisper_format_list = ["srt", "txt"]
whisper_language_list = ["Auto", "Chinese", "English"]
whisper_task_list = ["transcribe", "translate"]

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


async def exec_whisper_command(
    file_paths="", output_dir="", language="", model_type="", format=""
):
    model_id = f"openai/{model_type}"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    output_paths = []
    for file_path in file_paths:
        if language == "Auto":
            result = pipe(file_path, generate_kwargs={"task": "transcribe"})
        else:
            result = pipe(
                file_path, generate_kwargs={"task": "transcribe", "language": language}
            )
        output_path = os.path.join(output_dir, f"output.{format}")
        if format == "srt":
            result = _parse_result_to_srt(result)
            with open(output_path, "w", encoding="utf-8") as srtFile:
                srtFile.write(result)
        else:
            with open(output_path, "w", encoding="utf-8") as txtFile:
                txtFile.write(result["text"])
        output_paths.append(output_path)
    return output_paths


def _parse_result_to_srt(result):
    ret_result = ""
    for i, chunk in enumerate(result["chunks"]):
        startTime = str(0) + str(timedelta(seconds=int(chunk["timestamp"][0]))) + ",000"
        endTime = str(0) + str(timedelta(seconds=int(chunk["timestamp"][1]))) + ",000"
        text = chunk["text"]
        segmentId = i + 1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
        print(segment)
        ret_result += segment
    return ret_result
