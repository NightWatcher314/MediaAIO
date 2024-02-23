import asyncio
import os
import utils.utils as utils
from config import base_models_dir, base_work_dir, logger

rife_model_list = ["RIFE"]
script_dir = os.path.join(base_models_dir, "ECCV2022-RIFE")
inference_video_path = os.path.join(script_dir, "inference_video.py")


def _rife_video_command_warp(
    input_path="",
    output_path="",
    exp=2,
    scale=1,
    fp16=False,
    montage=False,
    fps=None,
):
    command = f"python {inference_video_path} --exp={exp} --scale={scale} --video='{input_path}' --output='{output_path}'"
    if fp16:
        command += " --fp16"
    if montage:
        command += " --montage"
    if fps is not None:
        command += f" --fps={fps}"
    return command


async def exec_rife_video_command(
    input_paths=[],
    output_dir="",
    exp=2,
    scale=1,
    fp16=False,
    montage=False,
    fps=None,
) -> list[str]:
    """
    执行 RIFE 视频命令并返回输出路径列表。

    参数：
    - input_paths：输入文件路径列表。
    - output_dir：输出目录路径。
    - exp：扩展倍数，默认为2。
    - scale：缩放比例，默认为1。
    - fp16：是否使用 FP16 模式，默认为False。
    - montage：是否进行蒙太奇，默认为False。
    - fps：输出视频的帧率，默认为None。

    返回：
    - 输出路径列表。

    异步函数。
    """
    logger.info("开始执行 RIFE 视频命令。")
    output_paths = []
    for input_path in input_paths:
        input_ext = input_path.split(".")[-1]
        input_name = input_path.split("/")[-1].split(".")[0]
        input_dir = os.path.dirname(input_path)
        if output_dir == "":
            output_dir = input_dir
        output_path = os.path.join(
            output_dir, f"{input_name}_rife_fps_{2**exp}x.{input_ext}"
        )
        output_paths.append(output_path)
        try:
            os.chdir(script_dir)
            command = _rife_video_command_warp(
                input_path, output_path, exp, scale, fp16, montage, fps
            )
            utils.run_util_complete(command)
        finally:
            os.chdir(base_work_dir)
    logger.info("RIFE 视频命令执行完成。")
    return output_paths


def test_video_execution():
    print("Start to test RIFE video execution.")
    asyncio.run(
        exec_rife_video_command(
            input_path="/home/night/Code/MediaAIO/test_assets/video_test_4X_112fps.mp4",
            fp16=True,
            uhd=True,
        )
    )
    print("Test done.")
    return 0
