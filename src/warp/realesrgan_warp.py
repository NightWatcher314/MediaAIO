import os
import torch
import subprocess
from config import base_models_dir, base_work_dir

reg_model_list = [
    "realesr-animevideov3",
    "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x2plus",
    "realesr-general-x4v3",
]

image_inference_path = os.path.join(
    base_models_dir, "Real-ESRGAN/inference_realesrgan.py"
)
video_inference_path = os.path.join(
    base_models_dir, "Real-ESRGAN/inference_realesrgan_video.py"
)
srcipt_dir = os.path.join(base_models_dir, "Real-ESRGAN")


def exec_realesrgan_command(
    input_path="",
    output_dir="",
    model="RealESRGAN_x4plus",
    scale=4,
    denoise=0.5,
    face_enhance=False,
    type="image",
):
    try:
        os.chdir(srcipt_dir)
        if type == "image":
            command = f"python {image_inference_path} -i {input_path} -o {output_dir} -n {model} -dn {denoise} -s {scale} "
            if face_enhance:
                command += " --face_enhance"

        elif type == "video":
            command = f"python {video_inference_path} -i {input_path} -o {output_dir} -n {model} -dn {denoise} -s {scale}"
            if face_enhance:
                command += " --face_enhance"
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line.decode("utf-8"))
    finally:
        os.chdir(base_work_dir)
