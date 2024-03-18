import asyncio
import os
import subprocess
import utils.utils as utils
from config import base_models_dir, base_work_dir, logger

reg_model_list = [
    "RealESRGAN_x4plus",
    "realesr-general-x4v3",
    "realesr-animevideov3",
    "RealESRGAN_x4plus_anime_6B",
    "RealESRNet_x4plus",
    "RealESRGAN_x2plus",
]

image_inference_path = os.path.join(
    base_models_dir, "Real-ESRGAN/inference_realesrgan.py"
)
video_inference_path = os.path.join(
    base_models_dir, "Real-ESRGAN/inference_realesrgan_video.py"
)
script_dir = os.path.join(base_models_dir, "Real-ESRGAN")


async def exec_realesrgan_command(
    input_paths=[],
    output_dir="",
    model="RealESRGAN_x4plus",
    scale=4,
    denoise=0.5,
    face_enhance=False,
    tile=0,
    fp32=False,
    task_type="video",
) -> list[str]:
    """
    执行 RealESRGAN 视频命令。

    参数：
    - input_paths：输入视频文件路径列表。
    - output_dir：输出目录。
    - model：模型名称，默认为 "RealESRGAN_x4plus"。
    - scale：放大倍数，默认为 4。
    - denoise：降噪强度，默认为 0.5。
    - face_enhance：是否进行人脸增强，默认为 False。
    - tile：分块大小，默认为 0。
    - fp32：是否使用 FP32 模式，默认为 False, 初始为 fp16。
    - task_type：任务类型，默认为 "video"。

    返回：
    - output_paths：输出视频文件路径列表。
    """
    logger.info("开始执行 RealESRGAN 命令。")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_paths = []
    for input_path in input_paths:
        input_name, input_ext = os.path.splitext(os.path.basename(input_path))
        output_file_name = f"{input_name}_reg.{input_ext}"
        if output_dir == "":
            output_dir = os.path.dirname(input_path)
        output_file_path = os.path.join(output_dir, output_file_name)
        output_paths.append(output_file_path)
        try:
            os.chdir(script_dir)
            if task_type == "video":
                command = f"python {video_inference_path} -i {input_path} -o {output_dir} -n {model} -s {scale} -t {tile}"
            else:
                command = f"python {image_inference_path} -i {input_path} -o {output_dir} -n {model} -s {scale} -t {tile}"
            if face_enhance:
                command += " --face_enhance"
            if fp32:
                command += " --fp32"
            if model == "realesr-general-x4v3":
                command += f" -dn {denoise} "
            utils.run_util_complete(command)
        finally:
            os.chdir(base_work_dir)
    logger.info("RealESRGAN 命令执行完成。")
    return output_paths


def test_exec_realesrgan_command():
    asyncio.run(
        exec_realesrgan_command(
            input_paths=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\a.mp4"
            ],
            output_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            model="RealESRGAN_x4plus_anime_6B",
            scale=4,
            denoise=0.5,
            face_enhance=True,
            tile=0,
            fp32=False,
        )
    )
