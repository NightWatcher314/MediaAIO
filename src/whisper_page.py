import os
import sys
from time import sleep
import gradio as gr
from utils.utils import Logger
import audio.whisper as sr
import utils.utils as utils
from utils.process_manager import ProcessManager as PM

model_list = ["base", "medium", "large"]
format_list = ["txt", "vtt", "srt", "json", "tsv", "all"]
language_list = ["Chinese", "English"]
task_list = ["transcribe", "translate"]


task_type = "audio"
task_name = "whisper"
task_token = ""
process_hash = ""
output_path = ""
logs_dir = utils.get_logs_dir(task_type, task_name)
logger = Logger(logs_dir)
exec_logs_dir = ""
pm = PM()


def show_file(file):
    if file is None:
        return None
    return file


def on_stop_click():
    process = pm.get_process_by_hash(process_hash)
    process.terminate()


def on_convert_click(file_upload, model, language, output_format):
    global task_token, process_hash, output_path
    if not file_upload:
        gr.Error("Please upload a file.")
        return gr.Textbox("Please upload a file."), gr.File(visible=False)
    task_token = utils.get_task_token(task_type, task_name)
    exec_logs_dir = utils.get_exec_logs_dir(task_type, task_name, task_token)
    input_format = file_upload.name.split(".")[-1]
    utils.save_gradio_file(file_upload, exec_logs_dir, "input." + input_format)
    input_path = os.path.join(exec_logs_dir, "input." + input_format)
    output_dir = os.path.join(exec_logs_dir)
    command = sr.whisper_command_warp(
        input_path, output_dir, language, model, output_format
    )
    process_hash = pm.exec(command, task_token)
    p = pm.get_process_by_hash(process_hash)
    pm.print_process_stdout_and_wait(p)
    pm.delete_process_by_hash(process_hash)
    if p.poll() != 0:
        gr.Error("Conversion Stop.")
        return gr.Textbox("Conversion Stop."), gr.File(visible=False)
    output_path = os.path.join(output_dir, f"input.{output_format}")
    output_text = utils.read_text_file(output_path)
    return output_text, gr.File(output_path, visible=True)


def ui():
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Audio File",
                        file_types=["audio"],
                        scale=1,
                    )
                    audio_display = gr.Audio(scale=1)
                    file_upload.change(show_file, file_upload, audio_display)
                with gr.Column():
                    output = gr.Textbox(label="Output Text")
                    download_file = gr.File(
                        label="Download File", type="filepath", visible=False
                    )
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        label="Model", choices=model_list, value=model_list[0]
                    )
                    language = gr.Dropdown(
                        label="Language", choices=language_list, value=language_list[0]
                    )
                with gr.Column():
                    output_format = gr.Dropdown(
                        label="Output Format", choices=format_list, value=format_list[0]
                    )
        with gr.Row():
            with gr.Column():
                convert_btn = gr.Button(value="Convert")
            with gr.Column():
                stop_btn = gr.Button(value="Stop")
        logs_box = gr.Textbox(label="Logs", max_lines=10)
        # demo.load(logger.read_logs, None, logs_box, every=1)
        convert_btn.click(
            on_convert_click,
            [file_upload, model, language, output_format],
            [output, download_file],
        )

        stop_btn.click(on_stop_click)

    return demo
