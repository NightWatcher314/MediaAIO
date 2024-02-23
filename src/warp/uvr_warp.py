import os
import torch
import subprocess
from config import base_models_dir, base_work_dir
from audio_separator.separator import Separator
import asyncio
import sys
from config import logger
from tqdm import tqdm
import utils.utils as utils


uvr_mdx_model_list = ["Reverb_HQ_By_FoxJoy.onnx", "UVR-MDX-NET-Inst_Main.onnx"]
uvr_vr_model_list = [
    "5_HP-Karaoke-UVR.pth",
    "3_HP-Vocal-UVR.pth",
    "2_HP-UVR.pth",
    "1_HP-UVR.pth",
    "UVR-De-Echo-Aggressive.pth",
    "UVR-DeNoise.pth",
    "UVR-DeEcho-DeReverb.pth",
]
weights_dir = os.path.join(base_models_dir, "audio-separator")


async def exec_uvr_command_gradio(
    model_filename="",
    output_format="mp3",
    primary_out_dir="",
    second_out_dir="",
    denoise=False,
    normalization=0.9,
    sample_rate=44100,
    single_stem=None,
    model_type="mdx",
    mdx_args={},
    vr_args={},
    audio_files=[],
):
    """
    执行 UVR 命令的异步函数。

    Args:
        model_filename (str, optional): 模型文件名。默认为空字符串。
        output_format (str, optional): 输出格式。默认为 "mp3"。
        primary_out_dir (str, optional): 主音频输出目录。默认为空字符串。
        second_out_dir (str, optional): 次音频输出目录。默认为空字符串。
        denoise (bool, optional): 是否进行降噪。默认为 False。
        normalization (float, optional): 归一化阈值。默认为 0.9。
        sample_rate (int, optional): 采样率。默认为 44100。
        single_stem (str, optional): 单音频输出路径。默认为 None。
        model_type (str, optional): 模型类型。默认为 "mdx"。
        mdx_args (dict, optional): MDX 参数。默认为空字典。
        vr_args (dict, optional): VR 参数。默认为空字典。
        audio_files (list, optional): 音频文件列表。默认为空列表。

    Returns:
        int: 返回值为 0。
    """
    logger.info("开始执行 UVR 命令。")
    if not os.path.exists(primary_out_dir):
        os.makedirs(primary_out_dir)
    if not os.path.exists(second_out_dir):
        os.makedirs(second_out_dir)
    for audio_file in tqdm(audio_files):
        primary_name = (
            os.path.basename(audio_file).split(".")[0]
            + "_primary."
            + output_format.lower()
        )
        secondary_name = (
            os.path.basename(audio_file).split(".")[0]
            + "_secondary."
            + output_format.lower()
        )
        primary_stem_output_path = os.path.join(primary_out_dir, primary_name)
        secondary_stem_output_path = os.path.join(second_out_dir, secondary_name)
        if model_type == "mdx":
            separator = Separator(
                model_file_dir=weights_dir,
                output_single_stem=single_stem,
                enable_denoise=denoise,
                normalization_threshold=normalization,
                primary_stem_output_path=primary_stem_output_path,
                secondary_stem_output_path=secondary_stem_output_path,
                mdx_params=mdx_args,
                sample_rate=sample_rate,
                output_format=output_format,
            )
        elif model_type == "vr":
            separator = Separator(
                model_file_dir=weights_dir,
                output_single_stem=single_stem,
                enable_denoise=denoise,
                normalization_threshold=normalization,
                primary_stem_output_path=primary_stem_output_path,
                secondary_stem_output_path=secondary_stem_output_path,
                vr_params=vr_args,
                sample_rate=sample_rate,
                output_format=output_format,
            )
        separator.load_model(model_filename)
        separator.separate(audio_file)
    logger.info("UVR 命令执行完成。")
    return 0
