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


whisper_models_dir = os.path.join(base_models_dir, "whisper")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


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
    global model
    _check_and_download_model(model_type)
    model_path = _get_model_path(model_type)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_path)
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
    result = pipe(
        file_path, generate_kwargs={"task": "transcribe", "language": language}
    )
    if format == "srt" and output_dir != "":
        result = _parse_result_to_srt(result)
        output_path = os.path.join(output_dir, "output.srt")
        with open(output_path, "w", encoding="utf-8") as srtFile:
            srtFile.write(result)


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


def _get_model_path(model_type):
    model_path = os.path.join(whisper_models_dir, model_type)
    return os.path.join(model_path, "snapshots", "model")


def _check_and_download_model(model_type):
    model_path = os.path.join(whisper_models_dir, model_type)
    if os.path.exists(model_path):
        return os.path.join(model_path, "snapshots", "model")
    model_id = f"openai/whisper-{model_type}"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    download_path = os.path.join(
        f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--openai--whisper-{model_type}"
    )
    shutil.move(os.path.join(download_path), model_path)
    for dir in os.listdir(model_path):
        if dir == "snapshots":
            for ddir in os.listdir(os.path.join(model_path, dir)):
                shutil.move(
                    os.path.join(model_path, dir, ddir),
                    os.path.join(model_path, dir, "model"),
                )
    del model, processor
    return os.path.join(model_path, "snapshots", "model")
