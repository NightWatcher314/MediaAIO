import asyncio
import tempfile
from typing import List, Literal, Optional
from config import base_models_dir, base_work_dir, logger
import subprocess
import os
import utils.utils as utils

model_paths = {
    "real_sr": "model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x{}_GAN.pth",
    "real_sr_l": "model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x{}_GAN.pth",
    "gray_dn": "model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise{}.pth",
    "color_dn": "model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise{}.pth",
}
task_list = ["real_sr", "gray_dn", "color_dn"]

script_dir = os.path.join(base_models_dir, "SwinIR")
results_dir = os.path.join(script_dir, "results")


async def exec_swinir_command(
    input_files: List[str] = [],
    output_dir: str = "",
    task: Literal["real_sr", "gray_dn", "color_dn"] = "",
    tile: int = None,
    large_model: bool = False,
    scale: Literal[1, 2, 3, 4, 8] = 4,
    noise: Literal[15, 25, 50] = 15,
):
    logger.info("开始执行 SwinIR 命令。")
    utils.clear_dir(results_dir)
    tmp_dir = tempfile.TemporaryDirectory()
    utils.copy_files(input_files, tmp_dir.name)
    try:
        os.chdir(script_dir)
        match task:
            case "real_sr":
                save_dir = os.path.join(results_dir, f"swinir_{task}_x{scale}")
                model_path = model_paths["real_sr"].format(scale)
                if large_model:
                    save_dir = save_dir + "_large"
                    model_path = model_paths["real_sr_l"].format(scale)
                command = f"python main_test_swinir.py --task {task} --scale {scale} \
                    --model_path {model_path} \
                    --folder_lq {tmp_dir.name}"
                if large_model:
                    command += " --large_model"
            case "gray_dn":
                save_dir = os.path.join(results_dir, f"swinir_{task}_noise{noise}")
                command = f"python main_test_swinir.py --task {task} --noise {noise} \
                    --model_path {model_paths[task].format(noise)} \
                    --folder_gt {tmp_dir.name}"
            case "color_dn":
                save_dir = os.path.join(results_dir, f"swinir_{task}_noise{noise}")
                command = f"python main_test_swinir.py --task {task} --noise {noise} \
                    --model_path {model_paths[task].format(noise)} \
                    --folder_gt {tmp_dir.name}"
        if tile is not None:
            command += f" --tile {tile}"
        utils.run_util_complete(command)
        utils.copy_dir(save_dir, output_dir)
    finally:
        os.chdir(base_work_dir)
    logger.info("SwinIR 命令执行完毕。")
    return output_dir


def test_swinir_warp():
    asyncio.run(
        exec_swinir_command(
            input_files=[
                "/home/night/Code/MediaAIO/test_assets/image/22571708571302_.pic.jpg",
            ],
            output_dir="/home/night/Code/MediaAIO/test_assets",
            task=task_list[2],
            large_model=False,
            noise=25,
            tile=400,
            scale=4,
        )
    )
