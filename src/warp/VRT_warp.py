import asyncio
import tempfile
from typing import List, Literal, Optional
from config import base_models_dir, base_work_dir
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
    utils.clear_dir(results_dir)
    tmp_dir = tempfile.TemporaryDirectory()
    for input_file in input_files:
        utils.convert_video_to_frames(input_file, tmp_dir.name)
    os.chdir(script_dir)
    command = f"python {inference_path} --task {task} --folder_lq '{tmp_dir.name}' --tile {tile_num} {tile_patch} {tile_patch} \
    --tile_overlap {tile_overlap_num} {tile_overlap_patch} {tile_overlap_patch} --save_result"
    if task == "008_VRT_videodenoising_DAVIS":
        command += f" --sigma {denoise}"
    utils.run_util_complete(command)
    utils.copy_dir(results_dir, output_dir)
    os.chdir(base_work_dir)


def test_VRT_warp():
    asyncio.run(
        exec_VRT_command(
            input_files=[
                "/home/night/Code/MediaAIO/test_assets/video/a.mp4",
            ],
            output_dir="/home/night/Code/MediaAIO/test_assets",
            task=task_list[1],
            tile_num=12,
            tile_patch=256,
            denoise=25,
        )
    )
