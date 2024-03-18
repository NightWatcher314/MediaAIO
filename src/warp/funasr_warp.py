import asyncio
from datetime import timedelta
import os
from funasr import AutoModel
from tqdm import tqdm
from config import logger
import utils.utils as utils


funasr_model_list = ["paraformer-zh"]


async def exec_funasr_command(input_files="", output_dir="", output_format="srt"):
    """
    执行 FunASR 命令，将输入文件转换为指定格式的文本文件。

    参数：
    - input_files：输入文件的路径列表。
    - output_dir：输出文件的目录。
    - output_format：输出文件的格式，默认为 "srt"。

    返回：
    - output_paths：输出文件的路径列表。
    """
    logger.info("开始执行 FunASR 命令。")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2",
    )
    output_paths = []
    for input_file in tqdm(input_files):
        remove_flag = False
        if "video" in utils.detect_file_type(input_file):
            remove_flag = True
            input_file = utils.convert_video_to_audio(input_file)
        if "mp3" not in utils.detect_file_type(input_file):
            remove_flag = True
            input_file = utils.convert_audio_to_audio(input_file)
        res = model.generate(
            input=input_file,
            batch_size_s=300,
            hotword="魔搭",
        )
        input_file_name_no_ext = utils.get_file_name_without_ext(input_file)
        output_path = os.path.join(
            output_dir, f"{input_file_name_no_ext}.{output_format}"
        )
        if output_format == "srt":
            content = _parse_result_to_srt(res)
        if output_format == "txt":
            content = res[0]["text"]
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)
        output_paths.append(output_path)
        if remove_flag:
            utils.remove_file(input_file)
    logger.info("FunASR 命令执行完毕，结果位于 {}。".format(output_dir))
    return output_paths


def _test_exec_funasr_command():
    asyncio.run(
        exec_funasr_command(
            [r"C:\Users\night\Document\Code\MediaAIO\test_assets\audio\audio.aac"],
            r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
        )
    )


def _parse_result_to_srt(result):
    ret_result = ""
    for i, chunk in enumerate(result[0]["sentence_info"]):
        # startTime = utils.timedelta_to_srt_format(timedelta(seconds=chunk["start"]))
        # endTime = utils.timedelta_to_srt_format(timedelta(seconds=chunk["end"]))
        startTime = str(0) + str(timedelta(seconds=int(chunk["start"]))) + ",000"
        endTime = str(0) + str(timedelta(seconds=int(chunk["end"]))) + ",000"
        text = chunk["text"]
        segmentId = i + 1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
        ret_result += segment
    return ret_result
