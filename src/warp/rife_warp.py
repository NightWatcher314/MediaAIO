import os
import torch
import subprocess
from config import base_models_dir, base_work_dir

inference_path = os.path.join(base_models_dir, "ECCV2022-RIFE/inference_video.py")
srcipt_dir = os.path.join(base_models_dir, "ECCV2022-RIFE")


def exec_rife_command(input_path="", output_dir="", exp=2, scale=4, task="video"):
    try:
        os.chdir(srcipt_dir)
        if task == "video":
            pass
    finally:
        os.chdir(base_work_dir)
