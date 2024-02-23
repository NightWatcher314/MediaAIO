import asyncio
import os
import subprocess
import utils.utils as utils
from config import base_models_dir, base_work_dir

nafnet_model_list = [
    "NAFNet-width64-GoPro",
    "NAFNet-width64-SIDD",
    "NAFNet-width64-REDS",
]
script_dir = os.path.join(base_models_dir, "NAFNet")
inference_path = os.path.join(script_dir, "basicsr/demo.py")


async def exec_nafnet_command(input_paths=[], output_dir="", model=""):
    output_paths = []
    for input_path in input_paths:
        input_name, input_ext = os.path.splitext(input_path)
        model_type = model.split("-")[-1]
        model_name = "-".join(model.split("-")[:-1])
        config_path = os.path.join(
            script_dir, "options/test", f"{model_type}/{model_name}.yml"
        )
        if output_dir == "":
            input_dir = os.path.dirname(input_path)
            output_path = os.path.join(input_dir, f"{input_name}_{model}{input_ext}")
        else:
            output_path = os.path.join(output_dir, f"{input_name}_{model}{input_ext}")
        try:
            os.chdir(script_dir)
            command = f"poetry run python {inference_path} -opt '{config_path}' --input_path '{input_path}' --output_path '{output_path}'"
            utils.run_util_complete(command)
            output_paths.append(output_path)
        finally:
            os.chdir(base_work_dir)
    return output_paths


def test_exec_nafnet():
    asyncio.run(
        exec_nafnet_command(
            input_paths=[
                "/home/night/Code/MediaAIO/test_assets/image/22571708571302_.pic.jpg"
            ],
            output_dir="/home/night/Code/MediaAIO/test_assets",
            model="NAFNet-width64-REDS",
        )
    )
    return 0
