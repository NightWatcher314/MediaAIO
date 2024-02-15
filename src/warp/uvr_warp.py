import os
import torch
import subprocess
from config import base_models_dir, base_work_dir
from audio_separator.separator import Separator
import asyncio

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


# def _uvr_command_warp(
#     model_filename="",
#     output_format="mp3",
#     output_dir="",
#     denoise=False,
#     normalization=0.9,
#     single_stem="",
#     sample_rate=44100,
#     model_type="mdx",
#     mdx_args={},
#     vr_args={},
#     audio_file="",
# ):
#     model_file_dir = os.path.join(base_models_dir, "audio-separator")
#     command = f"audio-separator --model_filename {model_filename} --output_format {output_format} \
# --output_dir {output_dir} --normalization {normalization} --model_file_dir {model_file_dir} \
# --sample_rate {sample_rate}"
#     if denoise:
#         command += " --denoise"
#     if single_stem != "":
#         command += f" --single_stem {single_stem}"
#     if model_type == "mdx":
#         for key, value in mdx_args.items():
#             command += f" --{key} {value}"
#     elif model_type == "vr":
#         for key, value in vr_args.items():
#             command += f" --{key} {value}"
#     command += f" {audio_file}"
#     return command


async def exec_uvr_command(
    model_filename="",
    output_format="mp3",
    first_out_dir="",
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
    model_file_dir = os.path.join(base_models_dir, "audio-separator")
    for audio_file in audio_files:
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
        primary_stem_output_path = os.path.join(first_out_dir, primary_name)
        secondary_stem_output_path = os.path.join(second_out_dir, secondary_name)
        if model_type == "mdx":
            separator = Separator(
                model_file_dir=model_file_dir,
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
                model_file_dir=model_file_dir,
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
    return 0
