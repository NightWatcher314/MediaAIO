from datetime import timedelta
import os

from tqdm import tqdm
from config import logger
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

import utils.utils as utils

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


async def exec_whisper_command(
    file_paths="", output_dir="", language="", model_type="", format=""
):
    """
    执行 Whisper 命令的异步函数。

    参数：
    - file_paths: 要处理的文件路径列表。
    - output_dir: 输出文件的目录。
    - language: 语言设置。
    - model_type: 模型类型。
    - format: 输出文件的格式。

    返回：
    - output_paths: 输出文件的路径列表。
    """
    logger.info("开始执行 Whisper 命令。")
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
    for file_path in tqdm(file_paths):
        if utils.detect_file_type(file_path) == "video":
            file_path = utils.convert_video_to_audio(file_path)
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
    logger.info("Whisper 命令执行完毕。")
    return output_paths


def _parse_result_to_srt(result):
    """
    将结果解析为SRT格式的字幕文本。

    参数：
    result (dict): 包含结果信息的字典。

    返回：
    str: SRT格式的字幕文本。
    """
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
