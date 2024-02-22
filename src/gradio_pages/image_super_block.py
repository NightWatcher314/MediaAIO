import asyncio
import os
import gradio as gr
from config import logger
import utils.utils as utils
from warp.realesrgan_warp import (
    reg_model_list,
    exec_realesrgan_command,
)

super_model_list = reg_model_list
task_name = "image_super"
task_token = ""
exec_logs_dir = ""
logs_box_value = utils.StrWarp()
event_loop = asyncio.new_event_loop()


def on_convert_click(input_files, model, scale, denoise, face_enhance, tile, fp32):
    global task_token, event_loop, logs_box_value, exec_logs_dir
    if not input_files:
        gr.Error("Please upload a file.")
        return "", ""
    asyncio.set_event_loop(event_loop)
    task_token = utils.get_task_token(task_name)
    exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token)
    input_dir, output_dir = utils.generate_io_dir(exec_logs_dir)
    input_files = utils.copy_files(input_files, input_dir)
    output_paths = event_loop.run_until_complete(
        exec_realesrgan_command(
            input_paths=input_files,
            output_dir=output_dir,
            model=model,
            scale=scale,
            denoise=denoise,
            face_enhance=face_enhance,
            tile=tile,
            fp32=fp32,
            task_type="image",
        )
    )
    zipfile = utils.zip_dir(output_dir, os.path.join(exec_logs_dir, "output.zip"))
    return output_paths, gr.File(zipfile, label="Download Zip", visible=True)


def ui():
    with gr.Blocks() as demo:
        with gr.Accordion(label="Image Upload"):
            with gr.Row():
                with gr.Column():
                    image_files = gr.Files(label="Upload Images", file_types=["image"])
                    with gr.Row():
                        convert_btn = gr.Button(value="Convert")
                        stop_btn = gr.Button(value="Stop")
                    download_file = gr.File(label="Download Zip", visible=False)
                with gr.Column():
                    image_outputs = gr.Gallery(label="Output Images")
        with gr.Accordion(label="Parameters"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        label="Model",
                        choices=super_model_list,
                        value=super_model_list[0],
                    )
                    scale = gr.Slider(label="Scale", value=4)

                with gr.Column():
                    enable_fp32 = gr.Checkbox(value=False, label="Enable FP32")
                    enable_face_enhance = gr.Checkbox(
                        value=False, label="Enable Face Enhance"
                    )
                    denoise = gr.Slider(
                        label="Denoise",
                        value=0.5,
                    )
                    tile = gr.Slider(
                        label="Tile Size",
                        value=0,
                    )
        convert_btn.click(
            on_convert_click,
            [
                image_files,
                model,
                scale,
                denoise,
                enable_face_enhance,
                tile,
                enable_fp32,
            ],
            [image_outputs, download_file],
        )
    return demo
