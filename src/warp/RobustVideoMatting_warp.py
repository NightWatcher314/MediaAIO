import os
import torch
import subprocess
from config import base_models_dir, base_work_dir


model_list = ["mobilenetv3", "resnet50"]

inference_path = os.path.join(base_models_dir, "RobustVideoMatting/inference.py")
srcipt_dir = os.path.join(base_models_dir, "RobustVideoMatting")
checkpoint_path_dict = {
    "mobilenetv3": os.path.join(
        base_models_dir, "RobustVideoMatting/weights/rvm_mobilenetv3.pth"
    ),
    "resnet50": os.path.join(
        base_models_dir, "RobustVideoMatting/weights/rvm_resnet50.pth"
    ),
}


def exec_rvm_command(
    input_path="",
    output_dir="",
    model="",
    output_mbps=4,
    output_type="video",
    output_alpha=False,
    output_foreground=False,
):
    try:
        os.chdir(srcipt_dir)
        output_path = os.path.join(output_dir, "composition.mp4")
        command = f"python inference.py --variant {model} --checkpoint {checkpoint_path_dict[model]} \
--device cuda --input-source {input_path} --output-type {output_type} --output-composition {output_path} \
--output-video-mbps {output_mbps}"
        if output_alpha:
            alpha_path = os.path.join(output_dir, "alpha.mp4")
            command += f" --output-alpha {alpha_path}"
        if output_foreground:
            foreground_path = os.path.join(output_dir, "foreground.mp4")
            command += f" --output-foreground {foreground_path}"
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        for line in p.stdout:
            print(line.decode("utf-8"))
    finally:
        os.chdir(base_work_dir)
