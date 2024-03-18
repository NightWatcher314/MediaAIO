import asyncio
import os
import gradio as gr
from config import logger
import utils.utils as utils
from warp.uvr_warp import uvr_mdx_model_list, uvr_vr_model_list, exec_uvr_command

task_name = "uvr"
block_name = "UVR"
uvr_model_list = uvr_mdx_model_list + uvr_vr_model_list
event_loop = asyncio.new_event_loop()
task_token = ""
exec_logs_dir = ""


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


def on_stop_click():
    event_loop.stop()


def on_convert_click(
    model_type,
    model_filename,
    enable_denoise,
    normalization_threshold,
    output_format,
    sample_rate,
    mdx_segment_size,
    mdx_overlap,
    mdx_batch_size,
    mdx_hop_length,
    vr_batch_size,
    vr_window_size,
    vr_aggression,
    vr_high_end_process,
    vr_enable_tta,
    vr_enable_post_process,
    vr_post_process_threshold,
    file_uploads,
    output_stem,
    primary_file_dir,
    second_file_dir,
    gr_process=gr.Progress(),
):
    global task_token, exec_logs_dir, event_loop
    task_token = utils.get_task_token(task_name)
    exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token)
    is_diff_dir = primary_file_dir != second_file_dir
    if primary_file_dir == "Default":
        primary_file_dir = os.path.join(exec_logs_dir, "primary")
    if second_file_dir == "Default":
        second_file_dir = os.path.join(exec_logs_dir, "second")
    print("Start to execute uvr command.")
    output_stem = output_stem if output_stem != "all" else None
    mdx_args = {
        "hop_length": mdx_hop_length,
        "segment_size": mdx_segment_size,
        "overlap": mdx_overlap,
        "batch_size": mdx_batch_size,
    }
    vr_args = {
        "batch_size": vr_batch_size,
        "window_size": vr_window_size,
        "aggression": vr_aggression,
        "high_end_process": vr_high_end_process,
        "enable_tta": vr_enable_tta,
        "enable_post_process": vr_enable_post_process,
        "post_process_threshold": vr_post_process_threshold,
    }

    asyncio.set_event_loop(event_loop)
    event_loop.run_until_complete(
        exec_uvr_command(
            audio_files=file_uploads,
            model_filename=model_filename,
            output_format=output_format,
            primary_out_dir=primary_file_dir,
            second_out_dir=second_file_dir,
            denoise=enable_denoise,
            normalization=normalization_threshold,
            sample_rate=sample_rate,
            single_stem=output_stem,
            model_type=model_type,
            mdx_args=mdx_args,
            vr_args=vr_args,
        )
    )

    if is_diff_dir:
        primary_zip_file = os.path.join(primary_file_dir, "primary.zip")
        second_zip_file = os.path.join(second_file_dir, "second.zip")
        utils.zip_dir(primary_file_dir, primary_zip_file)
        utils.zip_dir(second_file_dir, second_zip_file)
        return gr.File(
            [primary_zip_file, second_zip_file], label="Download", visible=True
        )
    else:
        zip_file = os.path.join(exec_logs_dir, "uvr.zip")
        utils.zip_dir(exec_logs_dir, zip_file)
        return gr.File(zip_file, label="Download", visible=True)


def ui():
    print("Start to render uvr block.")
    with gr.Blocks() as demo:
        with gr.Row(variant="compact"):
            with gr.Column():
                file_uploads = gr.Files(
                    label="Audio File",
                    file_count="multiple",
                    file_types=[".mp3", ".wav", ".ogg", ".flac"],
                )
                file_download = gr.File(label="Download", visible=False)
            with gr.Column():
                output_stem = gr.Dropdown(
                    choices=["instrumental", "vocals", "all"],
                    label="Output Stem",
                    value="all",
                )
                primary_file_dir = gr.Textbox(
                    label="Primary Audio Dir", value="Default"
                )
                second_file_dir = gr.Textbox(label="Second Audio Dir", value="Default")
            with gr.Column():
                with gr.Row():
                    convert_btn = gr.Button(value="Convert")
                    stop_btn = gr.Button(value="Stop")
                with gr.Row():
                    logs_area = gr.TextArea(label="Logs", max_lines=7)
                    # demo.load(logger.read_logs, None, logs_area, every=1)

        with gr.Accordion(label="Advanced Options", open=True):
            with gr.Row():
                with gr.Column():
                    model_type = gr.Dropdown(
                        choices=["mdx", "vr"],
                        label="Model-Type",
                        value="mdx",
                        interactive=True,
                    )
                    model_filename = gr.Dropdown(
                        uvr_mdx_model_list,
                        label="Model",
                        value=uvr_mdx_model_list[0],
                        interactive=True,
                    )
                    model_type.change(show_model_by_type, model_type, model_filename)
                with gr.Column():
                    enable_denoise = gr.Dropdown(
                        choices=["True", "False"],
                        label="Enable Denoise",
                        value="False",
                        interactive=True,
                    )
                    normalization_threshold = gr.Number(
                        label="Normalization Threshold",
                        value=0.9,
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        interactive=True,
                    )
                with gr.Column():
                    output_format = gr.Dropdown(
                        choices=["mp3", "wav", "ogg", "flac"],
                        label="Output Format",
                        value="wav",
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
        with gr.Accordion(label="Mdx Arch Options", open=False):
            with gr.Row():
                with gr.Column():
                    mdx_segment_size = gr.Number(label="Segment Size", value=256)
                    mdx_overlap = gr.Number(label="Overlap", value=0.25)
                with gr.Column():
                    mdx_batch_size = gr.Number(label="Batch Size", value=1)
                    mdx_hop_length = gr.Number(label="Hop Length", value=1024)
        with gr.Accordion(label="Vr Arch Options", open=False):
            with gr.Row():
                with gr.Column():
                    vr_enable_tta = gr.Checkbox(label="Enable TTA", value=False)
                    vr_enable_post_process = gr.Checkbox(
                        label="Enable Post Process", value=False
                    )
                    vr_high_end_process = gr.Checkbox(
                        label="High End Process", value=False
                    )
                with gr.Column():
                    vr_batch_size = gr.Number(label="Batch Size", value=16)
                    vr_window_size = gr.Number(label="Window Size", value=512)
                with gr.Column():
                    vr_aggression = gr.Number(label="Aggression", value=5)
                    vr_post_process_threshold = gr.Number(
                        label="Post Process Threshold", value=0.2
                    )

        convert_btn.click(
            on_convert_click,
            [
                model_type,
                model_filename,
                enable_denoise,
                normalization_threshold,
                output_format,
                sample_rate,
                mdx_segment_size,
                mdx_overlap,
                mdx_batch_size,
                mdx_hop_length,
                vr_batch_size,
                vr_window_size,
                vr_aggression,
                vr_high_end_process,
                vr_enable_tta,
                vr_enable_post_process,
                vr_post_process_threshold,
                file_uploads,
                output_stem,
                primary_file_dir,
                second_file_dir,
            ],
            file_download,
        )
        stop_btn.click(on_stop_click)
    return demo
