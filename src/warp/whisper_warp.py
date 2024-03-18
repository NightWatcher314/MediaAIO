import asyncio
from datetime import timedelta
import os
from typing import List, Literal, LiteralString
from faster_whisper import WhisperModel
from tqdm import tqdm
from config import logger, base_models_dir
import utils.utils as utils

whisper_model_list = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v2"]
whisper_format_list = ["srt", "txt"]
whisper_language_list = ["Auto", "zh", "en"]
whisper_task_list = ["transcribe", "translate"]
whisper_compute_type_list = ("float16", "int8")


async def exec_whisper_command(
    file_paths=[],
    output_dir="",
    language="",
    model_type="",
    format="",
    compute_type="float16",
):
    logger.info("开始执行 Whisper 命令。")
    output_paths = []
    model = WhisperModel(model_type, device="cuda", compute_type=compute_type)
    for file_path in tqdm(file_paths):
        remove_flag = False
        if utils.detect_file_type(file_path) == "video":
            remove_flag = True
            file_path = utils.convert_video_to_audio(file_path)
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language=language if language != "Auto" else None,
        )
        output_path = os.path.join(
            output_dir, os.path.basename(file_path).split(".")[0] + f".{format}"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            if format == "srt":
                f.write(_parse_segments_to_srt(segments))
            else:
                f.write(_parse_segments_to_txt(segments))
        if remove_flag:
            utils.remove_file(file_path)
    logger.info("Whisper 命令执行完毕。")
    return output_paths


def _parse_segments_to_srt(segments):
    srt_str = ""
    for i, segment in enumerate(segments):
        start_time = utils.convert_timedelta_to_srt(timedelta(seconds=segment.start))
        end_time = utils.convert_timedelta_to_srt(timedelta(seconds=segment.end))
        srt_str += f"{i+1}\n{start_time} --> {end_time}\n{segment.text}\n\n"
    return srt_str


def _parse_segments_to_txt(segments):
    txt_str = ""
    for i, segment in enumerate(segments):
        txt_str += f"{segment.text}\n"
    return txt_str


def _test_whisper_exec():
    asyncio.run(
        exec_whisper_command(
            file_paths=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\audio\audio_copy.aac",
            ],
            output_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            language="zh",
            model_type="large-v3",
            format=whisper_format_list[0],
            compute_type="float16",
        )
    )
