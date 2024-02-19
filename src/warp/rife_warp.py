import os
import subprocess
from config import base_models_dir, base_work_dir

rife_model_list = ["RIFE"]
script_dir = os.path.join(base_models_dir, "ECCV2022-RIFE")
inference_video_path = os.path.join(script_dir, "inference_video.py")


def _rife_video_command_warp(
    input_path="",
    output_path="",
    exp=2,
    scale=1,
    fp16=False,
    montage=False,
    fps=None,
):
    command = f"python {inference_video_path} --exp={exp} --scale={scale} --video='{input_path}' --output='{output_path}'"
    if fp16:
        command += " --fp16"
    if montage:
        command += " --montage"
    if fps is not None:
        command += f" --fps={fps}"
    return command


def exec_rife_video_command_in_subprocess(
    input_path="",
    output_path="",
    exp=2,
    scale=1,
    fp16=False,
    montage=False,
    fps=None,
) -> tuple[subprocess.Popen, str]:
    if output_path == "":
        input_ext = input_path.split(".")[-1]
        input_name = input_path.split("/")[-1].split(".")[0]
        input_dir = os.path.dirname(input_path)
        output_path = os.path.join(
            input_dir, f"{input_name}_rife_fps_{2**exp}x.{input_ext}"
        )
    try:
        os.chdir(script_dir)
        command = _rife_video_command_warp(
            input_path, output_path, exp, scale, fp16, montage, fps
        )
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    finally:
        os.chdir(base_work_dir)
    return p, output_path


def test_video_execution():
    print("Start to test RIFE video execution.")
    p = exec_rife_video_command_in_subprocess(
        input_path="/home/night/Code/MediaAIO/test_assets/video_test_4X_112fps.mp4",
        fp16=True,
        uhd=True,
    )
    p.wait()
    print("Test done.")
    return 0
