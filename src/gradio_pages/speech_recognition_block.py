import os
import sys
from time import sleep
import gradio as gr
import asyncio
from config import base_work_dir
from warp.whisper_warp import (
    whisper_model_list,
    whisper_language_list,
    whisper_format_list,
    exec_whisper_command,
)
from warp.funasr_warp import funasr_model_list, exec_funasr_command
import utils.utils as utils

os.chdir(base_work_dir)

block_name = "Speech Recognition"
sr_model_list = whisper_model_list + funasr_model_list
sr_language_list = whisper_language_list
sr_format_list = whisper_format_list
task_name = "speech_recognition"
task_token = ""
exec_logs_dir = ""
logs_box_value = utils.StrWarp()
event_loop = asyncio.new_event_loop()


def on_stop_click():
    event_loop.stop()


def single_on_convert_click(file_upload, model, language, output_format):
    global task_token, event_loop, logs_box_value, exec_logs_dir
    if not file_upload:
        gr.Error("Please upload a file.")
        return "", gr.File(visible=False)
    asyncio.set_event_loop(event_loop)
    task_token = utils.get_task_token(task_name)
    exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token)
    input_format = file_upload.split(".")[-1]
    utils.copy_file(file_upload, exec_logs_dir, "input." + input_format)
    input_path = os.path.join(exec_logs_dir, "input." + input_format)
    output_dir = os.path.join(exec_logs_dir)
    print("Starting process")
    if model in whisper_model_list:
        output_path = event_loop.run_until_complete(
            exec_whisper_command(
                file_paths=[input_path],
                output_dir=output_dir,
                language=language,
                model_type=model,
                format=output_format,
            )
        )[0]
    else:
        output_path = event_loop.run_until_complete(
            exec_funasr_command(
                input_files=[input_path],
                output_dir=output_dir,
                output_format=output_format,
            )
        )[0]
    print("Finished process")
    output_text = utils.read_text_file(output_path)
    return output_text, gr.File(output_path, visible=True)


def batch_on_convert_click(file_uploads, model, language, output_format):
    global task_token, event_loop, logs_box_value
    if not file_uploads:
        gr.Error("Please upload a file.")
        return "", gr.File(visible=False)
    asyncio.set_event_loop(event_loop)
    task_token = utils.get_task_token(task_name)
    exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token)
    input_dir, output_dir = utils.generate_io_dir(exec_logs_dir)
    input_paths = []
    for file_upload in file_uploads:
        input_paths.append(utils.copy_file(file_upload, input_dir))
    print("Starting process")
    if model in whisper_model_list:
        output_paths = event_loop.run_until_complete(
            exec_whisper_command(
                file_paths=input_paths,
                output_dir=output_dir,
                language=language,
                model_type=model,
                format=output_format,
            )
        )
    else:
        output_paths = event_loop.run_until_complete(
            exec_funasr_command(
                input_files=input_paths,
                output_dir=output_dir,
                output_format=output_format,
            )
        )
    print("Finished process")
    output_text = ""
    for output_path in output_paths:
        output_text += utils.read_text_file(output_path) + "\n"
    zip_file_path = utils.zip_dir(output_dir, os.path.join(exec_logs_dir, "output.zip"))
    return output_text, gr.File(zip_file_path, visible=True)


def ui():
    with gr.Blocks() as demo:
        with gr.Accordion(label="Single File"):
            with gr.Row():
                with gr.Column():
                    single_file_upload = gr.Audio(label="Upload Audio", type="filepath")
                    single_download_file = gr.File(
                        label="Download File", type="filepath", visible=False
                    )
                with gr.Column():
                    with gr.Row():
                        single_convert_btn = gr.Button(value="Convert")
                        single_stop_btn = gr.Button(value="Stop")
                    with gr.Row():
                        single_output_text = gr.TextArea(
                            value="",
                            label="Output Text",
                            lines=8,
                            max_lines=10,
                            visible=True,
                        )
        with gr.Accordion(label="Batch File"):
            with gr.Row():
                with gr.Column():
                    batch_file_upload = gr.Files(label="Upload Audio", type="filepath")
                    batch_download_file = gr.File(
                        label="Download File", type="filepath", visible=False
                    )
                with gr.Column():
                    with gr.Row():
                        batch_convert_btn = gr.Button(value="Convert")
                        batch_stop_btn = gr.Button(value="Stop")
                    with gr.Row():
                        batch_output_text = gr.TextArea(
                            value="",
                            label="Output Text",
                            lines=8,
                            max_lines=10,
                            visible=True,
                        )

        with gr.Accordion(label="Parameters"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        label="Model", choices=sr_model_list, value=sr_model_list[1]
                    )
                    language = gr.Dropdown(
                        label="Language",
                        choices=sr_language_list,
                        value=sr_language_list[0],
                    )
                with gr.Column():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=sr_format_list,
                        value=sr_format_list[0],
                    )
        single_convert_btn.click(
            single_on_convert_click,
            [single_file_upload, model, language, output_format],
            [single_output_text, single_download_file],
        )
        single_stop_btn.click(on_stop_click)
        batch_convert_btn.click(
            batch_on_convert_click,
            [batch_file_upload, model, language, output_format],
            [batch_output_text, batch_download_file],
        )
    return demo
