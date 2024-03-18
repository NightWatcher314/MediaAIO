import asyncio
import shutil
import sys
import tempfile
from typing import List, Literal, Optional
from config import base_models_dir, base_work_dir, logger
import subprocess
import os
from typing import Literal
import utils.utils as utils


task_list = [
    "001_VRT_videosr_bi_REDS_6frames",
    "005_VRT_videodeblurring_DVD",
    "006_VRT_videodeblurring_GoPro",
    "007_VRT_videodeblurring_REDS",
    "008_VRT_videodenoising_DAVIS",
    "009_VRT_videofi_Vimeo_4frames",
]

script_dir = os.path.join(base_models_dir, "VRT")
inference_path = os.path.join(script_dir, "main_test_vrt.py")
results_dir = os.path.join(script_dir, "results")


async def exec_VRT_command(
    input_files: List[str] = [],
    output_dir: str = "",
    task: Literal[
        "001_VRT_videosr_bi_REDS_6frames",
        "005_VRT_videodeblurring_DVD",
        "006_VRT_videodeblurring_GoPro",
        "007_VRT_videodeblurring_REDS",
        "008_VRT_videodenoising_DAVIS",
        "009_VRT_videofi_Vimeo_4frames",
    ] = "",
    tile_patch: int = 128,
    tile_num: int = 32,
    tile_overlap_patch: int = 20,
    tile_overlap_num: int = 2,
    denoise: Literal[0, 10, 20, 30, 40, 50] = 0,
):
    logger.info("开始执行 VRT 命令。")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    utils.clear_dir(results_dir)
    output_paths = []
    tmp_dir = tempfile.TemporaryDirectory()
    for input_file in input_files:
        input_name = utils.get_file_name_without_ext(input_file)
        frames_dir, fps = utils.convert_video_to_frames(
            input_file, os.path.join(tmp_dir.name, input_name)
        )
        _spilt_frames(frames_dir)
        audio_path = utils.convert_video_to_audio(
            input_file, os.path.join(tmp_dir.name, f"{input_name}.mp3")
        )
        try:
            os.chdir(script_dir)
            command = f"python {inference_path} --task {task} --folder_lq {frames_dir} --tile {tile_num} {tile_patch} {tile_patch} \
            --tile_overlap {tile_overlap_num} {tile_overlap_patch} {tile_overlap_patch} --save_result"
            if task == "008_VRT_videodenoising_DAVIS":
                command += f" --sigma {denoise}"
            utils.run_util_complete(command)
            output_frames_dir = os.path.join(results_dir, os.listdir(results_dir)[0])
            output_path = os.path.join(output_dir, f"{input_name}_VRT.mp4")
            utils.convert_frames_to_video(output_frames_dir, output_path, fps)
            utils.merge_audio_to_video(output_path, audio_path, output_path)
            output_paths.append(output_path)
        finally:
            os.chdir(base_work_dir)
    return output_paths


def _spilt_frames(input_dir: str, batch_size: int = 50):
    input_images = os.listdir(input_dir)
    num_batches = len(input_images) // batch_size + 1
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(input_images))
        subfolder_path = os.path.join(input_dir, f"batch_{batch_num + 1}")
        os.makedirs(subfolder_path, exist_ok=True)
        # 移动图像文件到子文件夹
        for i in range(start_idx, end_idx):
            src = os.path.join(input_dir, input_images[i])
            dst = os.path.join(subfolder_path, input_images[i])
            shutil.move(src, dst)
    return 0


def _test_VRT_warp():
    asyncio.run(
        exec_VRT_command(
            input_files=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test.mp4",
            ],
            output_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            task=task_list[1],
            tile_num=12,
            tile_patch=256,
            denoise=25,
        )
    )
