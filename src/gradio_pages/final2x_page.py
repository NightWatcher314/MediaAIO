import os
import sys
from time import sleep
import gradio as gr
from utils.utils import Logger
import image.final2x_warp.final2x_warp as final2x_warp
import utils.utils as utils
from utils.process_manager import ProcessManager as PM

model_list = ["ESRGAN", "RealSR", "SRGAN", "RRDB_ESRGAN"]

task_type = "video"
task_name = "super_resolution"
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


def on_convert_click(file_upload, model):
    global task_token, process_hash, output_path
    # if not file_upload:
    #     gr.Error("Please upload a file.")
    #     return
    # task_token = utils.get_task_token(task_type, task_name)
    # exec_logs_dir = utils.get_exec_logs_dir(task_type, task_name, task_token)
    # input_format = file_upload.name.split(".")[-1]
    # utils.save_gradio_file(file_upload, exec_logs_dir, "input." + input_format)
    # input_path = os.path.join(exec_logs_dir, "input." + input_format)
    # output_dir = os.path.join(exec_logs_dir)
    # command = final2x.final2x_command_warp()
    # process_hash = pm.exec(command, task_token)
    # p = pm.get_process_by_hash(process_hash)
    # while p.poll() is None:
    #     for line in p.stdout:
    #         print(line.decode("utf-8"))
    # pm.delete_process_by_hash(process_hash)
    # if p.poll() != 0:
    #     gr.Error("Conversion Stop.")
    #     return gr.Textbox("Conversion Stop."), gr.File(visible=False)
    # output_path = os.path.join(output_dir, f"input.{output_format}")
    # output_text = utils.read_text_file(output_path)
    # return output_text, gr.File(output_path, visible=True)


def ui():
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    file_uploadss = gr.File(
                        label="Image File",
                        file_count="multiple",
                        file_types=["image"],
                    )
                with gr.Column():
                    file_output = gr.Gallery(label="Output Image")

        with gr.Group():
            with gr.Row():
                model = gr.Dropdown(
                    model_list, label="Model", value=model_list[0], interactive=True
                )
                model_noise = gr.Number(label="Model Noise", value=0)
            with gr.Row():
                model_scale = gr.Number(label="Model Scale", value=4)
                target_scale = gr.Number(label="Target Scale", value=2)
        with gr.Row():
            convert_btn = gr.Button(value="Convert")
            # convert_btn.click(
            #     on_convert_click,
            #     [file_upload],
            # )
            stop_btn = gr.Button(value="Stop")
            stop_btn.click(on_stop_click)
        logs_box = gr.Textbox(label="Logs")
        # demo.load(logger.read_logs, None, logs_box, every=1)

    return demo
