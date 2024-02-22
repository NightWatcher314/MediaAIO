import os
import subprocess
import sys
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
srcipt_dir = os.path.join(base_models_dir, "Real-ESRGAN")


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
    - fp32：是否使用 FP32 模式，默认为 False。
    - task_type：任务类型，默认为 "video"。

    返回：
    - output_paths：输出视频文件路径列表。
    """
    print("开始执行 RealESRGAN 视频命令\n\n")
    output_paths = []
    for input_path in input_paths:
        input_file_name = input_path.split("/")[-1]
        output_file_name = (
            f"{input_file_name.split('.')[0]}_out.{input_file_name.split('.')[-1]}"
        )
        output_file_path = os.path.join(output_dir, output_file_name)
        output_paths.append(output_file_path)
        try:
            os.chdir(srcipt_dir)
            if task_type == "video":
                command = f"python {video_inference_path} -i '{input_path}' -o '{output_dir}' -n {model} -s {scale} -t {tile}"
            else:
                command = f"python {image_inference_path} -i '{input_path}' -o '{output_dir}' -n {model} -s {scale} -t {tile}"
            if face_enhance:
                command += " --face_enhance"
            if fp32:
                command += " --fp32"
            if model == "realesr-general-x4v3":
                command += f" -dn {denoise} "
            print(command)
            p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            p.wait()
        finally:
            os.chdir(base_work_dir)
    print("\n\nRealESRGAN 命令执行完毕")
    return output_paths


def test_exec_realesrgan_command():
    p = exec_realesrgan_command(
        input_path=["/home/night/Code/MediaAIO/test_assets/video_test.mp4"],
        output_dir="/home/night/Code/MediaAIO/test_assets",
        model="RealESRGAN_x4plus",
        scale=4,
        denoise=0.5,
        face_enhance=True,
        tile=0,
        fp32=True,
    )
    for line in p.stdout:
        print(111)
        print(line.decode("utf-8"))
