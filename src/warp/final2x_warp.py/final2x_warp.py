import json
import os

import torch


def final2x_command_warp(
    input_paths="",
    output_dir="",
    model="RealCUGAN-pro",
    model_scale=None,
    model_noise=None,
    target_scale=None,
    tta=False,
):
    json_str = _generate_config_json(
        input_paths, output_dir, model, model_scale, model_noise, target_scale, tta
    )
    # print(json_str)
    command = f"Final2x-core -j '{json_str}'"
    print(command)
    return f"input_path"


def _generate_config_json(
    input_paths="",
    output_dir="",
    model="RealCUGAN-pro",
    model_scale=None,
    model_noise=None,
    target_scale=None,
    tta=False,
):
    json_dict = {}
    if torch.cuda.is_available():
        json_dict["gpuid"] = 0
    else:
        json_dict["gpuid"] = -1
    json_dict["inputpath"] = input_paths
    json_dict["model"] = model
    json_dict["outputpath"] = output_dir
    if model_scale is not None:
        json_dict["modelscale"] = model_scale
    if model_noise is not None:
        json_dict["modelnoise"] = model_noise
    if target_scale is not None:
        json_dict["targetscale"] = target_scale
    json_dict["tta"] = tta
    # json_path = os.path.join(output_dir, "config.json")
    json_str = json.dumps(json_dict, indent=0).replace("\n", "")
    return json_str
