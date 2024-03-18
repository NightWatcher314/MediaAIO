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


uvr_mdx_model_list = list(Separator().list_supported_model_files()["MDX"].values())
uvr_vr_model_list = list(Separator().list_supported_model_files()["VR"].values())
uvr_demucs_model_list = list(
    Separator().list_supported_model_files()["Demucs"].values()
)
weights_dir = os.path.join(base_models_dir, "audio-separator")


async def exec_uvr_command(
    audio_files=[],
    model_filename="",
    output_format="mp3",
    primary_out_dir="",
    second_out_dir="",
    normalization=0.9,
    sample_rate=44100,
    single_stem=None,
    mdx_args={},
    vr_args={},
):
    logger.info("开始执行 UVR 命令。")
    if not os.path.exists(primary_out_dir):
        os.makedirs(primary_out_dir)
    if not os.path.exists(second_out_dir):
        os.makedirs(second_out_dir)
    primary_out_paths = []
    second_out_paths = []
    if model_filename in uvr_mdx_model_list:
        model_type = "mdx"
    elif model_filename in uvr_vr_model_list:
        model_type = "vr"
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
        primary_out_paths.append(primary_stem_output_path)
        second_out_paths.append(second_out_paths)
        if model_type == "mdx":
            separator = Separator(
                model_file_dir=weights_dir,
                output_single_stem=single_stem,
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
                normalization_threshold=normalization,
                primary_stem_output_path=primary_stem_output_path,
                secondary_stem_output_path=secondary_stem_output_path,
                vr_params=vr_args,
                sample_rate=sample_rate,
                output_format=output_format,
            )
        separator.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(separator.torch_device)

        print(f"加载模型 {model_filename}")
        separator.load_model(model_filename)
        separator.separate(audio_file)
    logger.info("UVR 命令执行完成。")
    return primary_out_paths, second_out_paths


def _test_uvr_command():
    print(uvr_mdx_model_list)
    print(uvr_vr_model_list)
    asyncio.run(
        exec_uvr_command(
            audio_files=[
                r"C:\Users\night\Document\Code\MediaAIO\test_assets\audio\audio_copy.aac"
            ],
            model_filename="UVR-DeEcho-DeReverb.pth",
            output_format="mp3",
            primary_out_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
            second_out_dir=r"C:\Users\night\Document\Code\MediaAIO\test_assets\output",
        )
    )
