import asyncio
import os
import subprocess
from typing import List, Literal

from tqdm import tqdm
import utils.utils as utils
from config import base_models_dir, base_work_dir, logger

nafnet_model_list = [
    "NAFNet-width64-GoPro",  # denoise
    "NAFNet-width32-GoPro",  # denoise
    "NAFNet-width64-SIDD",  # denoise
    "NAFNet-width32-SIDD",  # denoise
    "NAFNet-width64-REDS",  # deblur
]
script_dir = os.path.join(base_models_dir, "NAFNet")
inference_path = os.path.join(script_dir, "basicsr", "demo.py")


async def exec_nafnet_command(
    input_paths: List[str] = [],
    output_dir="",
    model: Literal[
        "NAFNet-width64-GoPro",
        "NAFNet-width32-GoPro",
        "NAFNet-width64-SIDD",
        "NAFNet-width32-SIDD",
        "NAFNet-width64-REDS",
    ] = "",
):
    logger.info("开始执行 NAFNet 命令。")
    output_paths = []
    for input_path in tqdm(input_paths):
        input_name, input_ext = os.path.splitext(os.path.basename(input_path))
        model_type = model.split("-")[-1]
        model_name = "-".join(model.split("-")[:-1])
        config_path = os.path.join(
            script_dir, "options", "test", f"{model_type}", f"{model_name}.yml"
        )
        if output_dir == "":
            input_dir = os.path.dirname(input_path)
            output_dir = input_dir
        output_path = os.path.join(output_dir, f"{input_name}_{model}{input_ext}")
        try:
            os.chdir(script_dir)
            command = f"poetry run python {inference_path} -opt {config_path} --input_path {input_path} --output_path {output_path}"
            utils.run_util_complete(command)
            output_paths.append(output_path)
        finally:
            os.chdir(base_work_dir)
    logger.info("NAFNet 命令执行完毕。")
    return output_paths


def _test_exec_nafnet():
    asyncio.run(
        exec_nafnet_command(
            input_paths=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\image\OIP.jpeg"
            ],
            output_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            model=nafnet_model_list[4],
        )
    )
    return 0
