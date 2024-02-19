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


def exec_realesrgan_video_command_in_subprocess(
    input_path="",
    output_dir="",
    model="RealESRGAN_x4plus",
    scale=4,
    denoise=0.5,
    face_enhance=False,
    tile=0,
    fp32=False,
) -> tuple[subprocess.Popen, str]:
    print("Start to exec RealESRGAN video command\n\n")
    input_file_name = input_path.split("/")[-1]
    output_file_name = (
        f"{input_file_name.split('.')[0]}_out.{input_file_name.split('.')[-1]}"
    )
    output_file_path = os.path.join(output_dir, output_file_name)
    try:
        os.chdir(srcipt_dir)
        command = f"python {video_inference_path} -i '{input_path}' -o '{output_dir}' -n {model} -s {scale} -t {tile}"
        if face_enhance:
            command += " --face_enhance"
        if fp32:
            command += " --fp32"
        if model == "realesr-general-x4v3":
            command += f" -dn {denoise} "
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    finally:
        os.chdir(base_work_dir)
    return p, output_file_path


def test_exec_realesrgan_video_command():
    p = exec_realesrgan_video_command_in_subprocess(
        input_path="/home/night/Code/MediaAIO/test_assets/video_test.mp4",
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
