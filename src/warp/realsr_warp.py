import asyncio
import os
import utils.utils as utils
from tqdm import tqdm
from typing import List, Literal
from config import base_models_dir, base_work_dir, logger, platform

realsr_model_list = [
    "models-DF2K_JPEG",
    "models-DF2K",
]


script_dir = os.path.join(base_models_dir, "realsr-ncnn-vulkan")
if platform.startswith("win"):
    inference_path = os.path.join(script_dir, "realsr-ncnn-vulkan.exe")
if platform.startswith("linux"):
    inference_path = os.path.join(script_dir, "realsr-ncnn-vulkan")


async def exec_realsr_command(
    input_paths: List[str] = [],
    output_dir: str = "",
    model: Literal["models-DF2K_JPEG", "models-DF2K"] = "models-DF2K_JPEG",
    output_format: Literal["png", "jpg", "webp"] = "png",
    scale=4,
    tile=0,
    tta=False,
) -> list[str]:
    logger.info("开始执行 realsr 命令。")
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for input_path in tqdm(input_paths):
        try:
            os.chdir(script_dir)
            input_name = utils.get_file_name_without_ext(input_path)
            output_path = os.path.join(
                output_dir, f"{input_name}_realsr.{output_format}"
            )
            command = f"{inference_path} -i {input_path} -o {output_path} -m {model} \
                -s {scale} -t {tile} -f {output_format} -g 1"
            if tta:
                command += " -x"
            utils.run_util_complete(command)
            output_paths.append(output_path)
        finally:
            os.chdir(base_work_dir)
    logger.info("RealESRGAN 命令执行完成。")
    return output_paths


def _test_exec_realsr_command():
    asyncio.run(
        exec_realsr_command(
            input_paths=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\image\OIP.jpeg",
            ],
            output_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            model=realsr_model_list[0],
            scale=4,
            tile=0,
            tta=True,
        )
    )
