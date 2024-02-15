import os
import sys
from time import sleep
import gradio as gr
from utils.utils import Logger
import utils.utils as utils
from warp.uvr_warp import uvr_mdx_model_list, uvr_vr_model_list, exec_uvr_command

uvr_model_list = uvr_mdx_model_list + uvr_vr_model_list


def show_model_by_type(model_type):
    if model_type == "mdx":
        return gr.Dropdown(
            uvr_mdx_model_list,
            label="Model",
            interactive=True,
            value=uvr_mdx_model_list[0],
        )
    elif model_type == "vr":
        return gr.Dropdown(
            uvr_vr_model_list,
            label="Model",
            interactive=True,
            value=uvr_vr_model_list[0],
        )


def ui():
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            with gr.Column():
                file_uploadss = gr.Files(
                    label="Image File",
                    file_count="multiple",
                    file_types=["audio"],
                )
            with gr.Column():
                output_stem = gr.Dropdown(
                    choices=["instrumental", "vocals", "all"], label="Output Stem"
                )
                primary_file_path = gr.Textbox(label="Primary Audio Path")
                second_file_path = gr.Textbox(label="Second Audio Path")
                primary_file_output = gr.File(label="Output Audio", visible=False)
                second_file_output = gr.File(label="Output Audio", visible=False)
            with gr.Column():
                with gr.Row():
                    convert_btn = gr.Button(value="Convert")
                    stop_btn = gr.Button(value="Stop")
                with gr.Row():
                    logs_box = gr.TextArea(label="Logs", interactive=False)

        with gr.Accordion(label="Advanced Options", open=True):
            with gr.Row():
                with gr.Column():
                    model_type = gr.Dropdown(
                        choices=["mdx", "vr"],
                        label="Model-Type",
                        value="mdx",
                        interactive=True,
                    )
                    model = gr.Dropdown(
                        uvr_mdx_model_list,
                        label="Model",
                        value=uvr_mdx_model_list[0],
                        interactive=True,
                    )
                    model_type.change(show_model_by_type, model_type, model)
                    output_format = gr.Dropdown(
                        choices=["mp3", "wav", "ogg", "flac"],
                        label="Output Format",
                        value="wav",
                        interactive=True,
                    )
                with gr.Column():
                    enable_denoise = gr.Dropdown(
                        choices=["True", "False"],
                        label="Enable Denoise",
                        value="False",
                        interactive=True,
                    )
                    sample_rate = gr.Number(
                        label="Sample Rate",
                        value=44100,
                        step=1000,
                        interactive=True,
                        minimum=1000,
                        maximum=96000,
                    )
                    normalization_threshold = gr.Number(
                        label="Normalization Threshold",
                        value=0.9,
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        interactive=True,
                    )
                with gr.Column() as arch_args:
                    batch_size = gr.Number(label="Model Noise", value=0)
                    model_scale = gr.Number(label="Model Scale", value=4)
                    target_scale = gr.Number(label="Target Scale", value=2)
            # with gr.Row():
            #     model = gr.Dropdown(
            #         uvr_model_list,
            #         label="Model",
            #         value=uvr_model_list[0],
            #         interactive=True,
            #     )
            #     model_noise = gr.Number(label="Model Noise", value=0)
            # with gr.Row():
            #     model_scale = gr.Number(label="Model Scale", value=4)
            #     target_scale = gr.Number(label="Target Scale", value=2)
        # demo.load(logger.read_logs, None, logs_box, every=1)

    return demo
